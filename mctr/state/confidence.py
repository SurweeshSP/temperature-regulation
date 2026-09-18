import torch
from typing import Tuple

def compute_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    """
    Computes primary confidence: max P_t(i)
    """
    return probabilities.max(dim=-1).values

def compute_margin_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    """
    Computes optional margin confidence: P_t[1] - P_t[2]
    """
    top2_probs = torch.topk(probabilities, k=2, dim=-1).values
    return top2_probs[..., 0] - top2_probs[..., 1]
