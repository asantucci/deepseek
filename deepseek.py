import torch
import torch.nn as nn
from moe import MoE, FeedForward
from mla import MultiHeadLatentAttention
from config import DeepSeekConfig
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, config: DeepSeekConfig, block_idx: int):
        super().__init__()
        self.self_attn = MultiHeadLatentAttention(config)
        self.mlp = (
            MoE(config)
            if block_idx >= config.first_k_dense_replace
            else FeedForward(config.d_model, config.mlp_hidden_dimension)
        )
        self.input_layernorm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.d_model, eps=config.rms_norm_eps
        )

    def forward(self, x: torch.tensor):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DeepSeekModel(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(config, i) for i in range(config.num_layers)]
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.init_weight_std = config.init_weight_std
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x: torch.tensor, targets: torch.tensor = None) -> torch.tensor:
        """
        Args:
            x: (B, T)
            target: (B, T)
        return: (B, T, vocab_size)
        """
        B, T = x.shape
        assert (
            T <= self.block_size
        ), f"Sequence length {T} cannot exceed block size {self.block_size}"

        x = self.embed_tokens(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DeepSeekModelForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.model = DeepSeekModel(config)
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

    def forward(self, x: torch.tensor, targets: torch.tensor = None) -> torch.tensor:
        x = self.model(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # for inference, only the last token logits is used for prediction the next token
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss


if __name__ == "__main__":
    config = DeepSeekConfig(
        d_model=1024,
        nheads=128,
        block_size=512,
        dropout=0.0,
        device="cuda",
        use_kv_cache=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        rope_head_dim=64,
        nope_head_dim=128,
        num_shared_experts=1,
        num_routed_experts=1,
        moe_hidden_dimension=20,
        mlp_hidden_dimension=20,
        topk=1,
        topk_norm_epsilon=1e-9,
        rms_norm_eps=1e-6,
        normalized_moe_gates=True,
        expert_load_balance_factor=0.01,
        num_layers=1,
        vocab_size=10000,
        init_weight_std=0.006,
        first_k_dense_replace=1,
    )
    input = torch.randint(0, config.vocab_size, (2, 2)).to(config.device)
    targets = torch.randint(0, config.vocab_size, (2, 2)).to(config.device)
    model = DeepSeekModelForCausalLM(config).to(config.device)
    # output, loss = model(input, targets)
    # print(f"when targets is not None: output shape: {output.shape}, loss: {loss}")
    targets = None
    model.eval()
    output, loss = model(input, targets)
    # print(f"when targets is None: output shape: {output.shape}, loss: {loss}")
    sd = model.state_dict()
    sd_keys = sd.keys()
    for key in sd_keys:
        print(f"{key}: {sd[key].shape}")
