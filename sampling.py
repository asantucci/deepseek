import torch

def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float):
    '''
    logits: [batch_size, vocab_size]
    temperature: float > 0
    top_p: float between 0 and 1
    '''
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_prob_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_removed_indices = cum_probs > top_p
    sorted_removed_indices[:, 1:] = sorted_removed_indices[:, :-1].clone()
    sorted_removed_indices[:, 0] = False # always keep the top first token
    # map the removed indices to the original logits
    removed_indices = sorted_removed_indices.scatter(dim=1, index=sorted_prob_indices, src=sorted_removed_indices)
    logits[removed_indices] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    # sample from the distribution
    return torch.multinomial(probs, num_samples=1)


def sample_top_k(logits: torch.Tensor, temperature: float, k: int):
    '''
    logits: [batch_size, vocab_size]
    temperature: float > 0
    k: int > 0
    '''
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    masked_logits = torch.full_like(logits, float('-inf'))
    masked_logits.scatter_(dim=1, index=top_k_indices, src=top_k_logits)
    probs = torch.softmax(masked_logits, dim=-1)
    # sample from the distribution
    return torch.multinomial(probs, num_samples=1)
