import torch
import math

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes predictive entropy from logits.
    Uses log_softmax for numerical stability.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

def compute_normalized_entropy(entropy: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Computes normalized entropy: H / log(|V|)
    """
    max_entropy = math.log(vocab_size)
    return entropy / max_entropy
