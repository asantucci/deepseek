import torch
import torch.nn as nn
from torch.nn import RMSNorm
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass


"""
From deepseek v2 paper,

d_model: 5120, hidden dimension
nheads: 128, n_h
block_size: 4096 then extend to 128k
q_lora_rank: 1536, d_c_prime
kv_lora_rank: 512, d_c
rope_head_dim: 64, d_h_R
nope_head_dim: 128, d_h
"""

@dataclass
class DeepSeekConfig:
    d_model: int
    nheads: int
    block_size: int
    dropout: float
    device: str
    use_kv_cache: bool
    # query low rank
    q_lora_rank: int
    kv_lora_rank: int
    rope_head_dim: int
    nope_head_dim: int
    

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        
        # transformation for Q
        self.q_down_proj = nn.Linear(config.d_model, config.q_lora_rank, bias=False)
        self.q_rms_norm = RMSNorm(config.q_lora_rank)
        self.q_up_proj = nn.Linear(config.q_lora_rank, config.nheads * (config.rope_head_dim + config.nope_head_dim), bias=False)
        
        # tranformation for K and V
        self.kv_down_proj = nn.Linear(config.d_model, config.kv_lora_rank, bias=False)
        self.kv_rms_norm = RMSNorm(config.kv_lora_rank)
        self.k_up_proj = nn.Linear(config.kv_lora_rank, config.nheads * config.nope_head_dim, bias=False)
        self.k_rope_proj = nn.Linear(config.d_model, config.rope_head_dim, bias=False)
        self.v_up_proj = nn.Linear(config.kv_lora_rank, config.nheads * config.nope_head_dim, bias=False)
        self.out = nn.Linear(config.nheads * config.nope_head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        self.nheads = config.nheads
        self.rope_head_dim = config.rope_head_dim
        self.nope_head_dim = config.nope_head_dim
        
        self.use_kv_cache = config.use_kv_cache
        self.cache_kv_lora = None
        self.cache_k_rope = None
        self.register_buffer(
            'rope_embeddings',
            self.rope_embedding(
                config.block_size,
                config.rope_head_dim,
                base=10000,
                device=torch.device(config.device)
            )
        )

        
    def rope_embedding(self, block_size: int, rope_head_dim: int, base: int = 10000, device: torch.device = torch.device('cpu')) -> torch.Tensor:
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
        dim_indices = torch.arange(0, rope_head_dim, 2).to(device) # lenght of rope_head_dim / 2
        freqs = 1 / (base ** (dim_indices / rope_head_dim))
        positions = torch.arange(block_size).to(device)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0) # (block_size, rope_head_dim/2)
        embeddings = torch.zeros((block_size, rope_head_dim)).to(device) # (block_size, rope_head_dim)
        embeddings[:, 0::2] = torch.cos(angles)
        embeddings[:, 1::2] = torch.sin(angles)
        return embeddings
    
    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor, rope_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply rope to the input tensor
        
        Args:
            x: the input tensor of shape (batch_size, seq_len, rope_head_dim) or (batch_size, seq_len, nheads, rope_head_dim)
            positions: the positions of the input tensor (seq_len)
            rope_embeddings: the rope embeddings of shape (max_seq_len, rope_head_dim)
        Returns:
            the rotated tensor of shape (batch_size, seq_len, rope_head_dim)
        """
        assert x.shape[1] <= rope_embeddings.shape[0], "The sequence length of x must be less than or equal to the sequence length of rope_embeddings"
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos_embed = rope_embeddings[positions, 0::2]
        sin_embed = rope_embeddings[positions, 1::2]
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even * cos_embed - x_odd * sin_embed
        x_rotated[...,1::2] = x_odd * cos_embed + x_even * sin_embed
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
        q_lora = self.q_down_proj(x) # B, T, q_lora_rank
        q_lora_norm = self.q_rms_norm(q_lora)
        q_up_proj = self.q_up_proj(q_lora_norm) # B, T, nheads * (rope_head_dim + nope_head_dim)
        q_up_proj = q_up_proj.view(B, T, self.nheads, -1) # B, T, nheads, rope_head_dim + nope_head_dim
        q_up_proj.transpose_(1, 2) # B, nheads, T, rope_head_dim + nope_head_dim
        # q_nope: B, nheads, T, nope_head_dim
        # q_rope: B, nheads, T, rope_head_dim
        q_nope, q_rope = torch.split(q_up_proj, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_rope = self.apply_rope(q_rope, positions, self.rope_embeddings) # B, T, nheads, rope_head_dim
        q = torch.cat((q_nope, q_rope), dim=-1) # B, T, nheads, rope_head_dim + nope_head_dim
        
        # transform K and V
        kv_lora = self.kv_rms_norm(self.kv_down_proj(x)) # B, T, kv_lora_rank
        k_shared_rope = self.k_rope_proj(x) # B, T, rope_head_dim
        k_shared_rope = self.apply_rope(k_shared_rope, positions, self.rope_embeddings) # B, T, rope_head_dim
        if self.cache_kv_lora is None:
            self.cache_kv_lora = kv_lora
            self.cache_k_rope = k_shared_rope
        else:
            self.cache_kv_lora = torch.cat((self.cache_kv_lora, kv_lora), dim=1)
            self.cache_k_rope = torch.cat((self.cache_k_rope, k_shared_rope), dim=1)
        k_nope = self.k_up_proj(self.cache_kv_lora) # B, T, nheads * nope_head_dim
        k_nope = k_nope.view(B, -1, self.nheads, self.nope_head_dim) # B, T, nheads, nope_head_dim
        k_shared_rope = self.cache_k_rope.view(B, -1, 1, self.rope_head_dim) \
                            .expand(-1, -1, self.nheads, -1) # B, T, nheads, rope_head_dim
        print(f"k_shared_rope shape: {k_shared_rope.shape}")
        print(f"k_nope shape: {k_nope.shape}")
        k = torch.cat((k_nope, k_shared_rope), dim=-1) # B, T, nheads, rope_head_dim + nope_head_dim
        
        v = self.v_up_proj(self.cache_kv_lora) # B, T, nheads * nope_head_dim
        v = v.view(B, -1, self.nheads, self.nope_head_dim) # B, T, nheads, nope_head_dim
        
        k.transpose_(1, 2) # B, nheads, T, rope_head_dim + nope_head_dim
        v.transpose_(1, 2) # B, nheads, T, nope_head_dim
        
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(B, -1, self.nheads * self.nope_head_dim)
        output = self.out(output)
        return self.dropout(output)

if __name__ == "__main__":
    config = DeepSeekConfig(
        d_model=1024, 
        nheads=128, 
        block_size=1024,
        dropout=0.0, 
        device='cuda', 
        use_kv_cache=True, 
        q_lora_rank=1536, 
        kv_lora_rank=512, 
        rope_head_dim=64, 
        nope_head_dim=128
    )
    mla = MultiHeadLatentAttention(config)
    mla = mla.to(config.device)
    input = torch.randn(2, 2, 1024).to(config.device)
    output = mla(input)
    print(f"output shape: {output.shape}")
    print("Add another token")
    new_token_ermbedding = torch.randn(2, 1, 1024).to(config.device)
    input = torch.cat((input, new_token_ermbedding), dim=1)
    output = mla(input)
    print(f"output shape: {output.shape}")
    