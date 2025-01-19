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

    # MoE parameters
    num_shared_experts: int  # Ns
    total_num_experts: int  # N
    hidden_dimension: int
    num_smaller_experts_per_expert: int  # m
    num_activated_experts: int  # K
    epsilon: float
    expert_load_balance_factor: float # alpha1
    
    # DeepSeek model
    num_layers: int
    vocab_size: int
    init_weight_std: float
