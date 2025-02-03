import torch
import torch.nn as nn
from moe import MoE, FeedForward
from mla import MultiHeadLatentAttention
from config import DeepSeekConfig
import torch.nn.functional as F
from typing import Optional
from mla import KVCache
import json

class Block(nn.Module):
    def __init__(self, config: DeepSeekConfig, block_idx: int):
        super().__init__()
        self.self_attn = MultiHeadLatentAttention(config, layer_idx=block_idx)
        self.mlp = (
            MoE(config)
            if block_idx >= config.first_k_dense_replace
            else FeedForward(config.d_model, config.mlp_hidden_dimension)
        )
        self.input_layernorm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.d_model, eps=config.rms_norm_eps
        )

    def forward(self, x: torch.tensor, past_key_value: Optional[KVCache] = None):
        """
        args:
            x: (B, T, d_model)
            past_key_value (KVCache, optional): when it is None, KV cache will not be used
        return:
            x: (B, T, d_model)
            past_key_value (KVCache, optional): None or updated KVCache.
        """
        identity = x
        x, past_key_value = self.self_attn(self.input_layernorm(x), past_key_value)
        if past_key_value is not None:
            print(f"past_key_value shape: {past_key_value.key_cache[0].shape}")
        x = identity + x
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, past_key_value


class DeepSeekModel(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(config, i) for i in range(config.num_layers)]
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.max_position_embeddings = config.max_position_embeddings
        self.init_weight_std = config.init_weight_std
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.tensor,
        past_key_value: Optional[KVCache] = None,
    ) -> torch.tensor:
        """
        Args:
            x: (B, T)
            target: (B, T)
        return: (B, T, vocab_size)
        """
        B, T = x.shape
        assert (
            T <= self.max_position_embeddings
        ), f"Sequence length {T} cannot exceed block size {self.max_position_embeddings}"

        x = self.embed_tokens(x)
        x = self.dropout(x)
        for layer in self.layers:
            x, past_key_value = layer(x, past_key_value)
        return self.norm(x), past_key_value


class DeepSeekModelForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.topk = config.topk
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.model = DeepSeekModel(config)
        self.config = config
        # initialize weights
        self.init_weight_std = config.init_weight_std
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.init_weight_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_weight_std)
            
    def get_total_parameters(self):
        total_params = 0
        activated_params = 0
        routed_moe_module_name = "mlp.experts"
        activated_routed_moe_module_name = [
            f"{routed_moe_module_name}.{i}" for i in range(self.topk)
        ]

        def is_activated_routed_moe_module(name: str):
            for activated_routed_moe_module in activated_routed_moe_module_name:
                if name.find(activated_routed_moe_module) != -1:
                    return True
            return False

        for name, param in self.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                if not name.find(
                    routed_moe_module_name
                ) == -1 or is_activated_routed_moe_module(name):
                    activated_params += param.numel()
        return total_params, activated_params

    def forward(
        self,
        x: torch.tensor,
        targets: torch.tensor = None,
        past_key_value: Optional[KVCache] = None,
    ) -> torch.tensor:
        x, past_key_value = self.model(x, past_key_value)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # for inference, only the last token logits is used for prediction the next token
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss, past_key_value
    
    @torch.no_grad()
    def generate(self, input: torch.tensor, max_length: int, temperature: float = 1.0):
        x = input
        kv_cache = KVCache(self.config.num_layers)
        for _ in range(max_length):
            # use kv cache
            logits, _, kv_cache = self(x, past_key_value=kv_cache)
            # [B, vocab_size]
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = next_token
            input = torch.cat([input, next_token], dim=1)
        return input


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    config = DeepSeekConfig(**config)
    input = torch.randint(0, config.vocab_size, (2, 2)).to(config.device)
    targets = torch.randint(0, config.vocab_size, (2, 2)).to(config.device)
    model = DeepSeekModelForCausalLM(config).to(config.device)
    output, loss, past_key_value = model(input, targets)
    print(f"when targets is not None: output shape: {output.shape}, loss: {loss}")
    targets = None
    model.eval()
    past_key_value = KVCache(config.num_layers)
    output, loss, past_key_value = model(input, targets, past_key_value)
    print("-" * 100)
    print("When targets is None")
    print(f"output shape: {output.shape}")
    print(f"loss: {loss}")
    print(f"KV Cache shape: {past_key_value.key_cache[0].shape}")
    print("-" * 100)
    print("State dict")
    sd = model.state_dict()
    sd_keys = sd.keys()
    for key in sd_keys:
        print(f"{key}: {sd[key].shape}")
