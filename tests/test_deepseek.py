from dataclasses import dataclass
import pytest
import torch
import torch.nn.functional as F
from deepseek import DeepSeekModelForCausalLM
from config import DeepSeekConfig
from mla import KVCache  # Ensure your test environment has this importable

@pytest.fixture
def config():
    return DeepSeekConfig(
        **{
            "d_model": 128,
            "nheads": 2,
            "max_position_embeddings": 4096,
            "dropout": 0.4,
            "device": "cuda",
            "use_kv_cache": True,
            "q_lora_rank": None,
            "kv_lora_rank": 128,
            "rope": {
                "head_dim": 8,
                "base": 10_000,
                "scaling": {
                    "type": "yarn",
                    "scaling_factor": 4.0,
                    "alpha": 1,
                    "beta": 32,
                    "attn_factor": 1.0,
                    "training_context_length": 128
                },
            },
            "nope_head_dim": 8,
            "v_head_dim": 4,
            "num_shared_experts": 2,
            "num_routed_experts": 2,
            "moe_hidden_dimension": 32,
            "mlp_hidden_dimension": 32,
            "topk": 1,
            "topk_norm_epsilon": 1e-9,
            "rms_norm_eps": 1e-6,
            "normalized_moe_gates": True,
            "expert_load_balance_factor": 0.01,
            "num_layers": 4,
            "vocab_size": 100,
            "init_weight_std": 0.006,
            "first_k_dense_replace": 1
        }
    )

@pytest.fixture
def model(config):
    return DeepSeekModelForCausalLM(config).to(config.device)

@pytest.mark.parametrize("include_targets", [True, False])
def test_forward_pass_shapes(model, config, include_targets):
    batch_size, seq_len = 2, 4
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(config.device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(config.device) if include_targets else None

    logits, loss, _ = model(input_ids.to(config.device), targets=targets)

    if include_targets:
        assert loss is not None
        assert loss.dtype == torch.float
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
    else:
        assert loss is None
        assert logits.shape == (batch_size, 1, config.vocab_size)

def test_kv_cache_integration(model, config):
    batch_size, seq_len = 2, 4
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(model.config.device)
    kv_cache = KVCache(config.num_layers)

    logits, loss, updated_cache = model(input_ids, past_key_value=kv_cache)
    assert isinstance(updated_cache, KVCache)
    assert updated_cache.key_cache[0].shape[0] == batch_size

def test_generate_function_output_shape(model, config):
    input_ids = torch.randint(0, config.vocab_size, (1, 3)).to(config.device)
    output = model.generate(input=input_ids, max_length=5)
    # output should be input shape + max_length
    assert output.shape == (1, 3 + 5)
    assert torch.all(output >= 0)  # ensure valid token ids

def test_get_total_parameters_counts_all(model):
    total, activated = model.get_total_parameters()
    assert total > 0
    assert activated > 0
    assert activated <= total
