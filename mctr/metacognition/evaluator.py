import torch
import torch.nn as nn
import torch.nn.functional as F
from .meta_state import MetaState

class MetaEvaluator(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, meta_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, meta_dim)
        )
        
        self.prediction_head = nn.Linear(meta_dim, 1)
        
    def forward(self, meta_state: MetaState) -> torch.Tensor:
        psi_t = meta_state.to_tensor()
        return self.mlp(psi_t)
        
    def predict_correctness(self, m_t: torch.Tensor) -> torch.Tensor:
        logits = self.prediction_head(m_t).squeeze(-1)
        return torch.sigmoid(logits)
        
    def compute_loss(self, m_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """
        Computes BCE loss for correctness prediction.
        y_t: 1 for correct, 0 for incorrect.
        """
        logits = self.prediction_head(m_t).squeeze(-1)
        # BCEWithLogitsLoss is numerically stable
        return F.binary_cross_entropy_with_logits(logits, y_t.float())
