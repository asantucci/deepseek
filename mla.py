import torch
import torch.nn as nn
from torch.nn import RMSNorm
import numpy as np
import torch.nn.functional as F
from config import DeepSeekConfig
from typing import Optional
import math
import json


# https://arxiv.org/abs/2104.09864
# look at https://github.com/kingoflolz/mesh-transformer-jax/blob/f2aa66e0925de6593dcbb70e72399b97b4130482/mesh_transformer/layers.py#L144
# to see if we could use this implementation
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
        max_position_embeddings: int,
        base: int = 10000,
        device: str = "cuda",
    ):
        super().__init__()
        self.rope_head_dim = rope_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device

        inv_freqs = 1.0 / (
            base
            ** (torch.arange(0, rope_head_dim, 2).float().to(device) / rope_head_dim)
        )
        self.register_buffer("inv_freqs", inv_freqs, persistent=False)
        self.max_seq_len_cached = None
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freqs.device,
            dtype=self.inv_freqs.dtype,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freqs.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# proposed by reddit user /u/kaiokendev
# https://www.reddit.com/r/LocalLLaMA/comments/14fgjqj/a_simple_way_to_extending_context_to_8k/
# concurrent work from meta: https://arxiv.org/abs/2306.15595
class PositinalInterpolationRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        rope_head_dim: int,
        max_position_embeddings: int,
        base: int = 10000,
        device: str = "cuda",
        scaling_factor: float = 1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(rope_head_dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freqs.dtype)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class NTKAwareRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        rope_head_dim: int,
        max_position_embeddings: int,
        base: int = 10000,
        device: str = "cuda",
        alpha: float = 1.0,
    ):
        base = base * alpha ** (rope_head_dim / (rope_head_dim - 2))
        super().__init__(rope_head_dim, max_position_embeddings, base, device)


class DynamicNTKAwareScalingRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        rope_head_dim: int,
        max_position_embeddings: int,
        base: int = 10000,
        device: str = "cuda",
        scaling_factor: float = 1.0,
    ):
        super().__init__(rope_head_dim, max_position_embeddings, base, device)
        self.scaling_factor = scaling_factor

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            # (self.scaling_factor * seq_len / self.max_position_embeddings)- (self.scaling_factor - 1)
            # is the same as
            # scaling_factor * (seq_len - max_position_embeddings) / max_position_embeddings + 1
            # which makes more sense and easier to understand
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.rope_head_dim / (self.rope_head_dim - 2))
            inv_freqs = 1.0 / (
                base
                ** (
                    torch.arange(0, self.rope_head_dim, 2).float().to(device)
                    / self.rope_head_dim
                )
            )
            self.register_buffer("inv_freqs", inv_freqs, persistent=False)
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def find_yarn_dim(ratio, training_context_length, rope_head_dim, base):
    # ratio = training_context_length / wave_length
    # wave_length = 2 * math.pi * base ** (2 * d / rope_head_dim)
    # this is to solve the above equations to find d
    return (
        rope_head_dim
        * math.log(training_context_length / (2 * math.pi * ratio))
        / (2 * math.log(base))
    )


def find_yarn_cut_dims(alpha, beta, training_context_length, rope_head_dim, base):
    low = find_yarn_dim(beta, training_context_length, rope_head_dim, base)
    low = math.floor(low)
    high = find_yarn_dim(alpha, training_context_length, rope_head_dim, base)
    high = math.ceil(high)
    return max(low, 0), min(high, rope_head_dim - 1)


def yarn_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def get_yarn_mscale(scaling_factor, attn_factor):
    if scaling_factor <= 1:
        return 1.0
    return (0.1 * math.log(scaling_factor) + 1.0) * attn_factor


class YarnRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        rope_head_dim: int,
        max_position_embeddings: int,
        base: int = 10000,
        device: str = "cuda",
        scaling_factor: float = 1.0,
        training_context_length: int = 1024,
        alpha: int = 1,
        beta: int = 32,
        attn_factor: float = 1.0,
    ):
        """
        From the paper: https://arxiv.org/pdf/2309.00071
        scaling_factor: the ratio between inference context length and training context length
        alpha: eq(18) of YaRN paper
        beta: eq(18) of YaRN paper
        attn_factor: the scaling factor for the temperature
        """
        self.scaling_factor = scaling_factor
        self.training_context_length = training_context_length
        self.alpha = alpha
        self.beta = beta
        self.attn_factor = attn_factor
        super().__init__(rope_head_dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        inv_freqs_interpolation = 1.0 / (
            self.scaling_factor
            * self.base ** torch.arange(0, self.rope_head_dim, 2).float().to(device)
            / self.rope_head_dim
        )
        inv_freqs_extrapolation = 1.0 / (
            self.base ** torch.arange(0, self.rope_head_dim, 2).float().to(device)
            / self.rope_head_dim
        )
        low, high = find_yarn_cut_dims(
            self.alpha,
            self.beta,
            self.training_context_length,
            self.rope_head_dim,
            self.base,
        )
        # for dimension lower than low, use extrapolation
        # for dimension higher than high, use interpolation
        # for dimension between low and high, use both extrapolation and interpolation
        # for mask is 1, use extrapolation, for mask is 0, use interpolation
        # in between use both
        inv_freq_mask = 1.0 - yarn_ramp_mask(
            low, high, self.rope_head_dim // 2
        ).float().to(device)
        inv_freqs = (
            1.0 - inv_freq_mask
        ) * inv_freqs_interpolation + inv_freq_mask * inv_freqs_extrapolation
        self.register_buffer("inv_freqs", inv_freqs, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freqs)
        emb = torch.cat((freqs, freqs), dim=-1)

        _mscale = get_yarn_mscale(self.scaling_factor, self.attn_factor)

        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )


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
        self.config = config
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
        self.attention_dropout_rate = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)

        self.nheads = config.nheads
        self.rope_head_dim = config.rope_head_dim
        self.nope_head_dim = config.nope_head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_base = config.rope_base
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.v_head_dim = config.v_head_dim
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rope = RotaryEmbedding(
                self.rope_head_dim,
                self.max_position_embeddings,
                base=self.rope_base,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["scaling_factor"]
            if scaling_type == "pi":
                self.rope = PositinalInterpolationRotaryEmbedding(
                    self.rope_head_dim,
                    self.max_position_embeddings,
                    base=self.rope_base,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rope = DynamicNTKAwareScalingRotaryEmbedding(
                    self.rope_head_dim,
                    self.max_position_embeddings,
                    base=self.rope_base,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "yarn":
                alpha = self.config.rope_scaling["alpha"]
                beta = self.config.rope_scaling["beta"]
                attn_factor = self.config.rope_scaling["attn_factor"]
                self.rope = YarnRotaryEmbedding(
                    self.rope_head_dim,
                    self.max_position_embeddings,
                    base=self.rope_base,
                    scaling_factor=scaling_factor,
                    alpha=alpha,
                    beta=beta,
                    attn_factor=attn_factor,
                    training_context_length=self.config.rope_scaling[
                        "training_context_length"
                    ],
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
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout_rate,
        )
        # B, q_len, nheads * v_head_dim
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(B, -1, self.nheads * self.v_head_dim)
        )
        output = self.residual_dropout(self.o_proj(output))
        return output, past_key_value


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    config = DeepSeekConfig(**config)
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
