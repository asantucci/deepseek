import torch
import torch.nn as nn
from torch.nn import RMSNorm
import numpy as np
import torch.nn.functional as F
from config import DeepSeekConfig
from typing import Optional


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


class KVCache(object):
    def __init__(self, num_layers: int):
        self.key_cache = [None] * num_layers
        self.value_cache = [None] * num_layers

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ):
        # key_states and value_states are of shape [bsz, num_heads, seq_len, head_dim]
        past_keys = self.key_cache[layer_idx]
        past_values = self.value_cache[layer_idx]
        if past_keys is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # concatenate along the sequence length dimension
            self.key_cache[layer_idx] = torch.cat((past_keys, key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((past_values, value_states), dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_cache_length(self, layer_idx: int):
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
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
        self.q_lora_rank = config.q_lora_rank
        self.v_head_dim = config.v_head_dim
        self._init_rope()

    def _init_rope(self):
        self.rope = RotaryEmbedding(
            self.rope_head_dim,
            self.block_size,
            base=self.rope_base,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
    ):
        B, q_len = x.shape[:2]

        if self.q_lora_rank is not None:
            q = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(q)
        else:
            q = self.q_proj(x)
        # B, nheads, q_len, rope_head_dim + nope_head_dim
        q = q.view(B, q_len, self.nheads, self.q_head_dim).transpose(1, 2)
        # q_nope: B, nheads, q_len, nope_head_dim
        # q_rope: B, nheads, q_len, rope_head_dim
        q_nope, q_rope = torch.split(
            q, [self.nope_head_dim, self.rope_head_dim], dim=-1
        )

        # B, q_len, kv_lora_rank + rope_head_dim
        kv_compressed = self.kv_a_proj_with_mqa(x)
        # kv_compressed: B, q_len, kv_lora_rank
        # k_rope: B, q_len, rope_head_dim
        kv_compressed, k_rope = kv_compressed.split(
            [self.kv_lora_rank, self.rope_head_dim], dim=-1
        )
        # add head dimension to k_rope. This k_rope is shared across all heads
        k_rope = k_rope.view(B, 1, q_len, self.rope_head_dim)

        # B, nheads, T, (nope_head_dim + v_head_dim)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(kv_compressed))
            .view(B, -1, self.nheads, self.nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, value_states = torch.split(
            kv, [self.nope_head_dim, self.v_head_dim], dim=-1
        )

        # apply rope to k_rope and q_rope
        # first get the seq_len including cache before this token
        past_seq_len = 0
        if past_key_value is not None:
            past_seq_len = past_key_value.get_cache_length(self.layer_idx)

        positions = torch.arange(
            past_seq_len, q_len + past_seq_len, device=x.device, dtype=torch.long
        )
        kv_seq_len = q_len + past_seq_len
        cos, sin = self.rope(value_states, seq_len=kv_seq_len)
        # q_rope: B, nheads, q_len, rope_head_dim
        # k_rope: B, 1, q_len, rope_head_dim
        q_rope, k_rope = apply_deepseek_rope(q_rope, k_rope, positions, cos, sin)

        # concatenate q/k_rope and q/k_nope
        query_states = q_rope.new_empty(B, self.nheads, q_len, self.q_head_dim)
        query_states[..., : self.nope_head_dim] = q_nope
        query_states[..., self.nope_head_dim :] = q_rope

        key_states = k_rope.new_empty(B, self.nheads, q_len, self.q_head_dim)
        key_states[..., : self.nope_head_dim] = k_nope
        key_states[..., self.nope_head_dim :] = k_rope

        # update kv cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        # when the q_len = kv_seq_len, the attention mask is casual mask
        # otherwise, the query can attend to all past tokens, so the attention bias is all 0
        attn_mask = None
        if q_len == kv_seq_len:
            attn_mask = torch.tril(
                torch.ones(q_len, q_len, device=x.device), diagonal=0
            ).bool()
        # B, nheads, q_len, v_head_dim
        output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attn_mask
        )
        # B, q_len, nheads * v_head_dim
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(B, -1, self.nheads * self.v_head_dim)
        )
        output = self.o_proj(output)
        return output, past_key_value


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
    mla = MultiHeadLatentAttention(config, layer_idx=0)
    mla = mla.to(config.device)
    input = torch.randn(2, 2, 1024).to(config.device)
    output, _ = mla(input)
    print(f"MLA output shape: {output.shape}")
    print("Add another token")
    new_token_ermbedding = torch.randn(2, 1, 1024).to(config.device)
    input = torch.cat((input, new_token_ermbedding), dim=1)
    output, _ = mla(input)
    print(f"MLA output shape: {output.shape}")
    
    # use kv cache
    mla = MultiHeadLatentAttention(config, layer_idx=0).to(config.device)
    kv_cache = KVCache(config.num_layers)
    output, kv_cache = mla(input, kv_cache)
    print(f"MLA output shape: {output.shape}")
    print(f"KV cache shape: {kv_cache.key_cache[0].shape}")
    new_token_ermbedding = torch.randn(2, 1, 1024).to(config.device)
    output, kv_cache = mla(new_token_ermbedding, kv_cache)
    print(f"MLA output shape: {output.shape}")
    print(f"KV cache shape: {kv_cache.key_cache[0].shape}")
