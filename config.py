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
    max_position_embeddings: int
    dropout: float
    device: str
    use_kv_cache: bool
    # MLA parameters
    q_lora_rank: int
    kv_lora_rank: int
    nope_head_dim: int
    v_head_dim: int
    rope: dict

    # MoE parameters
    num_shared_experts: int
    num_routed_experts: int
    topk: int
    moe_hidden_dimension: int
    mlp_hidden_dimension: int
    topk_norm_epsilon: float
    normalized_moe_gates: bool
    expert_load_balance_factor: float # alpha1
    rms_norm_eps: float
    first_k_dense_replace: int
    # DeepSeek model
    num_layers: int
    vocab_size: int
    init_weight_std: float
