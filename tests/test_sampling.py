import pytest
import torch
from sampling import sample_top_p, sample_top_k

@pytest.fixture
def logits():
    return torch.tensor([
        [1.0, 2.0, 3.0, 2.5],
        [1.0, 2.0, 3.0, 2.5]
    ])

def test_sample_top_k_shape(logits):
    output = sample_top_k(logits, temperature=1.0, k=2)
    assert output.shape == (2, 1)
    assert output.dtype == torch.long

@pytest.mark.parametrize("batch_idx", [0, 1])
def test_sample_top_k_respects_k(logits, batch_idx):
    k = 2
    output = sample_top_k(logits, temperature=1.0, k=k)
    # Get top-k indices for the specific row
    topk_indices = torch.topk(logits[batch_idx], k).indices.tolist()
    assert output[batch_idx].item() in topk_indices

def test_sample_top_p_shape(logits):
    output = sample_top_p(logits, temperature=1.0, top_p=0.9)
    assert output.shape == (2, 1)
    assert output.dtype == torch.long

@pytest.mark.parametrize("batch_idx", [0, 1])
def test_sample_top_p_is_subset(logits, batch_idx):
    output = sample_top_p(logits, temperature=1.0, top_p=0.9)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = cum_probs[batch_idx] <= 0.9
    top_p_mask[0] = True  # Always keep the highest prob token
    valid_indices = sorted_indices[batch_idx][top_p_mask].tolist()
    assert output[batch_idx].item() in valid_indices