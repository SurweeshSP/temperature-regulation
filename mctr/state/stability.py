import torch
import torch.nn.functional as F
from typing import Optional
from .conflict import compute_js_divergence

def compute_stability(h_t: torch.Tensor, h_prev: Optional[torch.Tensor], 
                      p_t: torch.Tensor, p_prev: Optional[torch.Tensor], 
                      lambda_h: float = 1.0, lambda_p: float = 1.0) -> torch.Tensor:
    """
    Computes candidate stability score:
    S_t = exp[ -lambda_h * ||h_t - h_{t-1}||_2 - lambda_p * D_JS(P_t, P_{t-1}) ]
    
    If previous states are not provided (e.g., at t=0), returns a baseline stability of 1.0.
    """
    if h_prev is None or p_prev is None:
        return torch.ones(h_t.shape[:-1], dtype=h_t.dtype, device=h_t.device)
        
    delta_h_norm = torch.norm(h_t - h_prev, p=2, dim=-1)
    
    # We use JS divergence for stable probability distance
    d_p = compute_js_divergence(p_t, p_prev)
    
    s_t = torch.exp(-lambda_h * delta_h_norm - lambda_p * d_p)
    return s_t
