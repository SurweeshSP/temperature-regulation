import torch
import torch.nn.functional as F
from typing import Optional

def compute_js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Computes Jensen-Shannon divergence between two predictive distributions p and q.
    """
    m = 0.5 * (p + q)
    
    p = torch.clamp(p, min=1e-10)
    q = torch.clamp(q, min=1e-10)
    m = torch.clamp(m, min=1e-10)
    
    # kl_div expects input in log-space, target in prob-space
    kl_p_m = F.kl_div(m.log(), p, reduction='none').sum(dim=-1)
    kl_q_m = F.kl_div(m.log(), q, reduction='none').sum(dim=-1)
    
    return 0.5 * kl_p_m + 0.5 * kl_q_m

def compute_conflict(p_a: torch.Tensor, p_b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes conflict between two generated distributions.
    If p_b is not provided, returns zero conflict.
    """
    if p_b is None:
        return torch.zeros(p_a.shape[:-1], dtype=p_a.dtype, device=p_a.device)
    
    return compute_js_divergence(p_a, p_b)
