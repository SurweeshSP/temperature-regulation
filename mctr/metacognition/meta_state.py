from dataclasses import dataclass
import torch

@dataclass
class MetaState:
    confidence: torch.Tensor
    uncertainty: torch.Tensor
    novelty: torch.Tensor
    conflict: torch.Tensor
    stability: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    
    def to_tensor(self) -> torch.Tensor:
        """
        Combines the state variables into a single tensor Psi_t.
        Order: [C_t, U_t, N_t, K_t, S_t, H_t_norm]
        Shape: [..., 6]
        """
        def _ensure_last_dim(t):
            if t.dim() < self.entropy.dim(): # if it's missing the sequence dimension? No, just unsqueeze
                pass
            return t.unsqueeze(-1) if t.dim() == self.entropy.dim() or t.shape[-1] != 1 else t
            
        c = _ensure_last_dim(self.confidence)
        u = _ensure_last_dim(self.uncertainty)
        n = _ensure_last_dim(self.novelty)
        k = _ensure_last_dim(self.conflict)
        s = _ensure_last_dim(self.stability)
        h_norm = _ensure_last_dim(self.normalized_entropy)
        
        return torch.cat([c, u, n, k, s, h_norm], dim=-1)
