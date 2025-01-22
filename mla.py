import torch
import torch.nn as nn
from torch.nn import RMSNorm
import numpy as np
import torch.nn.functional as F
from config import DeepSeekConfig


# https://arxiv.org/abs/2104.09864
def apply_original_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rope to the q and k

    Args:
        q: the query tensor of shape (batch_size, nheads, seq_len, rope_head_dim)
        k: the key tensor of shape (batch_size, nheads, seq_len, rope_head_dim)
        positions: the positions of the input tensor
        cos: the cosine part of the rope embedding, shape (seq_len, rope_head_dim / 2)
        sin: the sine part of the rope embedding, shape (seq_len, rope_head_dim / 2)
    Returns:
        q and k after applying rope
    """
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    cos = cos[positions]
    sin = sin[positions]
    q_rotated = torch.zeros_like(q)
    k_rotated = torch.zeros_like(k)
    # in deepseek implementation, x_even * cos_embed - x_odd * sin_embed is first half
    # x_odd * cos_embed + x_even * sin_embed is second half
    # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py#L339
    q_rotated[..., 0::2] = q_even * cos - q_odd * sin
    q_rotated[..., 1::2] = q_odd * cos + q_even * sin
    k_rotated[..., 0::2] = k_even * cos - k_odd * sin
    k_rotated[..., 1::2] = k_odd * cos + k_even * sin
    return q_rotated, k_rotated


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py#L339
def apply_deepseek_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rope to the q and k.

    The difference between the original rope and the deepseek rope is that the deepseek rope

    Args:
        q: the query tensor of shape (batch_size, nheads, seq_len, rope_head_dim)
        k: the key tensor of shape (batch_size, nheads, seq_len, rope_head_dim)
        positions: the positions of the input tensor
        cos: the cosine part of the rope embedding, shape (seq_len, rope_head_dim)
        sin: the sine part of the rope embedding, shape (seq_len, rope_head_dim)
    Returns:
        q and k after applying rope
    """
    cos = cos[positions]
    sin = sin[positions]

    b, h, s, d = q.shape
    # transform q and k so that the even indices elements are the first half
    # the odd indices elements are the second half
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    # the sequence length for q and k might be different due to kv cache
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated


"""
Get the cos and sin for the rope embeddings
"""


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        rope_head_dim: int,
        max_content_length: int,
        base: int = 10000,
        device: str = "cuda",
    ):
        super().__init__()
        self.rope_head_dim = rope_head_dim
        self.max_content_length = max_content_length
        self.base = base
        self.device = device

        freqs = 1.0 / (
            base
            ** (torch.arange(0, rope_head_dim, 2).float().to(device) / rope_head_dim)
        )
        self.register_buffer("freqs", freqs, persistent=False)
        self.max_seq_len_cached = None
        self._set_cos_sin_cache(
            seq_len=max_content_length, device=self.freqs.device, dtype=self.freqs.dtype
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.freqs.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()

        # transformation for Q
        self.q_head_dim = config.rope_head_dim + config.nope_head_dim
        self.q_lora_rank = config.q_lora_rank
        if config.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(config.d_model, config.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank,
                config.nheads * self.q_head_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                config.d_model, config.nheads * self.q_head_dim, bias=False
            )

        # tranformation for K and V
        self.kv_a_proj_with_mqa = nn.Linear(
            config.d_model, config.kv_lora_rank + config.rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.nheads * (config.nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.nheads * config.v_head_dim, config.d_model, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)

        self.nheads = config.nheads
        self.rope_head_dim = config.rope_head_dim
        self.nope_head_dim = config.nope_head_dim
        self.block_size = config.block_size
        self.rope_base = config.rope_base
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.use_kv_cache = config.use_kv_cache
        self.cache_kv_lora = None
        self.cache_k_rope = None
        self._init_rope()
        self.register_buffer(
            "rope_embeddings",
            self.rope_embedding(
                config.block_size,
                config.rope_head_dim,
                base=10000,
                device=torch.device(config.device),
            ),
        )

    def _init_rope(self):
        self.rope_embeddings_1 = RotaryEmbedding(
            self.rope_head_dim,
            self.block_size,
            base=self.rope_base,
        )

    def rope_embedding(
        self,
        block_size: int,
        rope_head_dim: int,
        base: int = 10000,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Generate the rope embeddings where the number of rows is the block size
        and the number of columns is the rope head dimension

        Args:
            block_size: max sequence length or context length
            rope_head_dim: the rope head dimension
            base: the base frequency
        Returns:
            rope_embeddings: the rope embeddings of shape (block_size, rope_head_dim)
        """
        dim_indices = torch.arange(0, rope_head_dim, 2).to(
            device
        )  # lenght of rope_head_dim / 2
        freqs = 1 / (base ** (dim_indices / rope_head_dim))
        positions = torch.arange(block_size).to(device)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(
            0
        )  # (block_size, rope_head_dim)
        embeddings = torch.zeros((block_size, rope_head_dim)).to(
            device
        )  # (block_size, rope_head_dim)
        embeddings[:, 0::2] = torch.cos(angles)
        embeddings[:, 1::2] = torch.sin(angles)
        return embeddings

    def apply_rope(
        self, x: torch.Tensor, positions: torch.Tensor, rope_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rope to the input tensor

        Args:
            x: the input tensor of shape (batch_size, seq_len, rope_head_dim) or (batch_size, nheads, seq_len, rope_head_dim)
            positions: the positions of the input tensor (seq_len)
            rope_embeddings: the rope embeddings of shape (max_seq_len, rope_head_dim)
        Returns:
            the rotated tensor of shape (batch_size, seq_len, rope_head_dim)
        """
        assert (
            x.shape[1] <= rope_embeddings.shape[0]
        ), "The sequence length of x must be less than or equal to the sequence length of rope_embeddings"
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos_embed = rope_embeddings[positions, 0::2]
        sin_embed = rope_embeddings[positions, 1::2]
        x_rotated = torch.zeros_like(x)
        # in deepseek implementation, x_even * cos_embed - x_odd * sin_embed is first half
        # x_odd * cos_embed + x_even * sin_embed is second half
        # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py#L339
        x_rotated[..., 0::2] = x_even * cos_embed - x_odd * sin_embed
        x_rotated[..., 1::2] = x_odd * cos_embed + x_even * sin_embed
        return x_rotated

    def forward(self, x):
        if self.use_kv_cache:
            return self.forward_with_kv_cache(x)
        raise NotImplementedError("Non KV cache is not supported")

    def forward_with_kv_cache(self, x):
        # if the cache is not empty which means it is autoregressive token generation
        # instead of context, we only need the last token to compute the q, k, v
        if self.cache_k_rope is not None:
            x = x[:, [-1], :]
            positions = torch.tensor([x.shape[1] - 1]).to(x.device)
        else:
            positions = torch.arange(x.shape[1]).to(x.device)
        B, T = x.shape[0], x.shape[1]
        # transform Q
        if self.q_lora_rank is not None:
            q = self.q_a_proj(x)  # B, T, q_lora_rank
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)
        else:
            q = self.q_proj(x)
        q = q.view(B, -1, self.nheads, self.q_head_dim)
        q.transpose_(1, 2)  # B, nheads, T, rope_head_dim + nope_head_dim
        # q_nope: B, nheads, T, nope_head_dim
        # q_rope: B, nheads, T, rope_head_dim
        q_nope, q_rope = torch.split(
            q, [self.nope_head_dim, self.rope_head_dim], dim=-1
        )
        q_rope = self.apply_rope(
            q_rope, positions, self.rope_embeddings
        )  # B, T, nheads, rope_head_dim
        q = torch.cat(
            (q_nope, q_rope), dim=-1
        )  # B, T, nheads, rope_head_dim + nope_head_dim

        # transform K and V
        # B, T, kv_lora_rank + rope_head_dim
        kv_compressed = self.kv_a_proj_with_mqa(x)
        # kv_compressed: B, T, kv_lora_rank
        # k_shared_rope: B, T, rope_head_dim
        kv_compressed, k_shared_rope = kv_compressed.split(
            [self.kv_lora_rank, self.rope_head_dim], dim=-1
        )
        k_shared_rope = self.apply_rope(
            k_shared_rope, positions, self.rope_embeddings
        )  # B, T, rope_head_dim
        if self.cache_kv_lora is None:
            self.cache_kv_lora = kv_compressed
            self.cache_k_rope = k_shared_rope
        else:
            self.cache_kv_lora = torch.cat((self.cache_kv_lora, kv_compressed), dim=1)
            self.cache_k_rope = torch.cat((self.cache_k_rope, k_shared_rope), dim=1)
        kv = self.kv_b_proj(self.kv_a_layernorm(self.cache_kv_lora)).view(
            B, -1, self.nheads, self.nope_head_dim + self.v_head_dim
        )  # B, T, nheads, (nope_head_dim + v_head_dim)
        k_nope, v = torch.split(kv, [self.nope_head_dim, self.v_head_dim], dim=-1)
        k_shared_rope = self.cache_k_rope.view(B, -1, 1, self.rope_head_dim).expand(
            -1, -1, self.nheads, -1
        )  # B, T, nheads, rope_head_dim
        k = torch.cat(
            (k_nope, k_shared_rope), dim=-1
        )  # B, T, nheads, rope_head_dim + nope_head_dim

        k = k.transpose(1, 2)  # B, nheads, T, rope_head_dim + nope_head_dim
        v = v.transpose(1, 2)  # B, nheads, T, v_head_dim

        # flash attention v1
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(B, -1, self.nheads * self.v_head_dim)
        )
        output = self.o_proj(output)
        return self.dropout(output)


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
        rope_base=10000,
        v_head_dim=128,
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
        first_k_dense_replace=0,
    )
    mla = MultiHeadLatentAttention(config)
    mla = mla.to(config.device)
    input = torch.randn(2, 2, 1024).to(config.device)
    output = mla(input)
    print(f"MLA output shape: {output.shape}")
    print("Add another token")
    new_token_ermbedding = torch.randn(2, 1, 1024).to(config.device)
    input = torch.cat((input, new_token_ermbedding), dim=1)
    output = mla(input)
    print(f"MLA output shape: {output.shape}")
