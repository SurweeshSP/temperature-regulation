import torch
import torch.nn as nn

class LearnedUncertaintyEstimator(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        # Example learned estimator taking hidden_state and entropy as input
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_state: torch.Tensor, probabilities: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        # Example concatenation. In a real scenario, probability projection could be used.
        # Here we just use hidden_state and entropy for simplicity.
        x = torch.cat([hidden_state, entropy.unsqueeze(-1)], dim=-1)
        return self.fc(x).squeeze(-1)

def compute_uncertainty(estimator_type: str, normalized_entropy: torch.Tensor, 
                        learned_estimator: nn.Module = None, hidden_state: torch.Tensor = None, 
                        probabilities: torch.Tensor = None, entropy: torch.Tensor = None) -> torch.Tensor:
    """
    Computes uncertainty, either as normalized entropy or using a learned estimator.
    """
    if estimator_type == "normalized_entropy":
        return normalized_entropy
    elif estimator_type == "learned":
        if learned_estimator is None:
            raise ValueError("Learned estimator must be provided when type is 'learned'")
        return learned_estimator(hidden_state, probabilities, entropy)
    else:
        raise ValueError(f"Unknown uncertainty estimator type: {estimator_type}")
