import torch
import torch.nn.functional as F

class NoveltyReference:
    def __init__(self, method="cosine"):
        self.method = method
        self.mu_ref = None
        self.L = None # Cholesky factor for Mahalanobis
        
    def fit(self, hidden_states: torch.Tensor):
        if hidden_states.dim() > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            
        self.mu_ref = hidden_states.mean(dim=0)
        
        if self.method == "mahalanobis":
            centered = hidden_states - self.mu_ref
            cov = (centered.T @ centered) / (hidden_states.size(0) - 1)
            # Add small identity for numerical stability
            cov += torch.eye(cov.size(0), device=cov.device) * 1e-5
            self.L = torch.linalg.cholesky(cov)

    def score(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.mu_ref is None:
            raise RuntimeError("Reference distribution must be fitted before scoring.")
            
        if self.method == "cosine":
            cos_sim = F.cosine_similarity(hidden_state, self.mu_ref.unsqueeze(0) if hidden_state.dim() > 1 else self.mu_ref, dim=-1)
            return 1.0 - cos_sim
            
        elif self.method == "mahalanobis":
            centered = hidden_state - self.mu_ref
            orig_shape = centered.shape
            centered_flat = centered.view(-1, orig_shape[-1])
            
            B = centered_flat.T
            X = torch.cholesky_solve(B, self.L)
            
            dist_sq = (B * X).sum(dim=0)
            dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))
            return dist.view(orig_shape[:-1])
        else:
            raise ValueError(f"Unknown novelty method: {self.method}")
