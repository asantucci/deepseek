import torch
from contextlib import nullcontext
import os
from training_config import TrainingConfig
from torch.utils.data import DataLoader
import json
from deepseek import DeepSeekModelForCausalLM, DeepSeekConfig
import bitsandbytes as bnb
import math
import torch.nn as nn

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def get_model_config(model_config_path: str):
    with open(model_config_path, "r") as f:
        config = json.load(f)
    return config


def get_wandb_config(training_config: TrainingConfig):
    config = {
        "batch_size": training_config.batch_size,
        "learning_rate": training_config.learning_rate,
        "use_fused_adamw": training_config.adamw_use_fused,
    }
    config.update(get_model_config(training_config.model_config_path))
    return config


def get_model(model_config_path: str):
    config = get_model_config(model_config_path)
    model = DeepSeekModelForCausalLM(DeepSeekConfig(**config))
    total_params, activated_params = model.get_total_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Activated parameters: {activated_params:,}")
    print(f"Activated parameters ratio: {activated_params / total_params:.2%}")
    return model


def configure_optimizers(
    model, weight_decay, learning_rate, betas, fused, use_eight_bit_optimizer=False
):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwis    no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    if use_eight_bit_optimizer:
        # fuse is not supported
        optimizer = bnb.optim.AdamW8bit(optim_groups, lr=learning_rate, betas=betas)
    else:
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=fused
        )
    return optimizer


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def evalaute(
    model: nn.Module,
    eval_iters: int,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    cur_train_iter: iter,
    cur_eval_iter: iter,
    device: str,
):
    model.eval()
    loss_iters = torch.zeros(eval_iters)
    losses = {}
    for i in range(eval_iters):
        eval_batch, cur_eval_iter = get_next_batch(eval_dataloader, cur_eval_iter)
        _, loss, _ = model(
            x = eval_batch["input_ids"].to(device),
            targets = eval_batch["labels"].to(device),
            past_key_value = None,
            attention_mask = eval_batch["attention_mask"].to(device),
        )
        loss_iters[i] = loss.item()
    losses["eval"] = loss_iters.mean()
    for i in range(eval_iters):
        train_batch, cur_train_iter = get_next_batch(train_dataloader, cur_train_iter)
        _, loss, _ = model(
            x = train_batch["input_ids"].to(device),
            targets = train_batch["labels"].to(device),
            past_key_value = None,
            attention_mask = train_batch["attention_mask"].to(device),
        )
        loss_iters[i] = loss.item()
    losses["train"] = loss_iters.mean()
    model.train()
    return losses, cur_train_iter, cur_eval_iter


def get_next_batch(dataloader, current_iter):
    try:
        batch = next(current_iter)
    except StopIteration:
        # Reset the iterator
        current_iter = iter(dataloader)
        batch = next(current_iter)

    return batch, current_iter


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        training_config: TrainingConfig,
    ):
        global ptdtype
        self.training_config = training_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ctx = (
            nullcontext()
            if training_config.device == "cpu"
            # mixed precision training
            else torch.amp.autocast(
                device_type=training_config.device, dtype=ptdtype[training_config.dtype]
            )
        )

    def train(self):
        if self.training_config.wandb_log:
            import wandb

            wandb.init(
                project=self.training_config.wandb_project,
                name=self.training_config.wandb_run_name,
                config=get_wandb_config(self.training_config),
            )
        model = get_model(self.training_config.model_config_path)
        model.to(self.training_config.device)
        optimizer = configure_optimizers(
            model,
            self.training_config.adamw_weight_decay,
            self.training_config.learning_rate,
            (self.training_config.adamw_beta1, self.training_config.adamw_beta2),
            self.training_config.adamw_use_fused,
            self.training_config.use_eight_bit_optimizer,
        )
        optimizer.zero_grad(set_to_none=True)
        best_val_loss = 1e9
        iter_num = 0
        if self.training_config.resume:
            checkpoint = torch.load(
                os.path.join(self.training_config.out_dir, "ckpt.pt")
            )
            config = checkpoint["model_config"]
            model = DeepSeekModelForCausalLM(config)
            model.to(self.training_config.device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_val_loss = checkpoint["best_val_loss"]
            iter_num = checkpoint["iter_num"]

        cur_train_iter = iter(self.train_dataloader)
        cur_eval_iter = iter(self.eval_dataloader)
        while iter_num < self.training_config.max_train_steps:
            # determine and set the learning rate for this iteration
            lr = (
                get_lr(
                    iter_num,
                    self.training_config.warmup_iters,
                    self.training_config.lr_decay_iters,
                    self.training_config.learning_rate,
                    self.training_config.min_learning_rate,
                )
                if self.training_config.decay_lr
                else self.training_config.learning_rate
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            for _ in range(self.training_config.gradient_accumulation_steps):
                batch, cur_train_iter = get_next_batch(
                    self.train_dataloader, cur_train_iter
                )
                with self.ctx:
                    _, train_loss, _ = model(
                        x = batch["input_ids"].to(self.training_config.device),
                        targets = batch["labels"].to(self.training_config.device),
                        past_key_value = None,
                        attention_mask = batch["attention_mask"].to(self.training_config.device),
                    )
                train_loss = (
                    train_loss / self.training_config.gradient_accumulation_steps
                )
                train_loss.backward()
            if self.training_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.training_config.grad_clip
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if (iter_num + 1) % self.training_config.eval_interval == 0:
                losses, cur_train_iter, cur_eval_iter = evalaute(
                    model,
                    self.training_config.eval_iters,
                    self.train_dataloader,
                    self.eval_dataloader,
                    cur_train_iter,
                    cur_eval_iter,
                    self.training_config.device,
                )
                if self.training_config.wandb_log:
                    wandb.log(
                        {
                            "Step": iter_num,
                            "Train Loss": losses["train"],
                            "Val Loss": losses["eval"],
                            "Learning Rate": lr,
                        }
                    )
                if losses["eval"] < best_val_loss:
                    best_val_loss = losses["eval"]
                    if iter_num > 0:
                        checkpoint = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_config": model.config,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "training_config": self.training_config,
                        }
                        torch.save(
                            checkpoint,
                            os.path.join(
                                self.training_config.out_dir,
                                self.training_config.checkpoint_path,
                            ),
                        )
                print(
                    f"step {iter_num+1}: train loss: {losses['train']:.4f}, val loss: {losses['eval']:.4f}"
                )
            iter_num += 1
