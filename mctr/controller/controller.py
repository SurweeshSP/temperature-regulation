import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ControlState:
    temperature: torch.Tensor
    # Extensible for Phase 2 (e.g., alpha, meta-attention parameters)

class MCTRController(nn.Module):
    def __init__(self, meta_dim: int, config: Dict[str, Any]):
        super().__init__()
        self.mode = config.get('mode', 'learned') # 'learned' or 'entropy_feedback'
        self.t_min = config.get('temperature_min', 0.2)
        self.t_max = config.get('temperature_max', 1.2)
        
        # We define a lightweight MLP mapping from meta-state embedding (m_t) to a raw control signal
        hidden_dim = config.get('hidden_dim', 32)
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Raw temperature logit
        )
        
        # For entropy feedback mode (MODE B)
        if self.mode == 'entropy_feedback':
            self.learning_rate_T = config.get('eta_T', 0.1)
            
    def _bound_temperature(self, raw_t: torch.Tensor) -> torch.Tensor:
        """
        Uses sigmoid to bound the raw temperature to [t_min, t_max].
        T_t = T_min + (T_max - T_min) * sigmoid(raw_T)
        """
        scaled = torch.sigmoid(raw_t)
        return self.t_min + (self.t_max - self.t_min) * scaled

    def forward(self, m_t: torch.Tensor, current_T: Optional[torch.Tensor] = None, 
                current_H: Optional[torch.Tensor] = None, target_H: Optional[torch.Tensor] = None) -> ControlState:
        """
        Calculates the temperature control state for the next step.
        
        Args:
            m_t: Meta-evaluator embedding of shape [..., meta_dim]
            current_T: Current temperature (required for entropy_feedback mode)
            current_H: Current entropy (required for entropy_feedback mode)
            target_H: Target entropy (required for entropy_feedback mode)
            
        Returns:
            ControlState containing the regulated temperature.
        """
        if self.mode == 'learned':
            # MODE A - Direct Learned Temperature
            raw_t = self.mlp(m_t).squeeze(-1)
            T_next = self._bound_temperature(raw_t)
            
        elif self.mode == 'entropy_feedback':
            # MODE B - Entropy-Feedback Temperature
            if current_T is None or current_H is None or target_H is None:
                raise ValueError("current_T, current_H, and target_H must be provided in entropy_feedback mode.")
                
            e_t = target_H - current_H
            
            # Predict a delta modifier from the meta state, or just use constant eta
            # For phase 1, we can optionally use the MLP to predict the learning rate delta, 
            # but standard form is T_{t+1} = clip(T_t + eta_T * e_t, T_min, T_max)
            # We'll use the MLP to dynamically scale the feedback if needed, but for strict adherence:
            T_next_unbounded = current_T + self.learning_rate_T * e_t
            
            # Clip instead of sigmoid for recursive feedback to maintain numerical meaning
            T_next = torch.clamp(T_next_unbounded, self.t_min, self.t_max)
            
        else:
            raise ValueError(f"Unknown controller mode: {self.mode}")
            
        return ControlState(temperature=T_next)

class EntropyTarget(nn.Module):
    def __init__(self, config: Dict[str, Any], meta_dim: int = 6):
        super().__init__()
        self.mode = config.get('target_mode', 'fixed')
        
        if self.mode == 'fixed':
            self.fixed_target = config.get('target_value', 0.5) # Assuming normalized entropy [0, 1]
        elif self.mode == 'adaptive':
            # Maps the raw state Psi_t to a target entropy
            hidden_dim = config.get('target_hidden_dim', 16)
            self.target_mlp = nn.Sequential(
                nn.Linear(meta_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid() # Bounds target between 0 and 1 for normalized entropy
            )
            
    def forward(self, psi_t: torch.Tensor) -> torch.Tensor:
        if self.mode == 'fixed':
            # Return a tensor of the same batch shape filled with the fixed target
            return torch.full(psi_t.shape[:-1], self.fixed_target, device=psi_t.device, dtype=psi_t.dtype)
        elif self.mode == 'adaptive':
            return self.target_mlp(psi_t).squeeze(-1)
        else:
            raise ValueError(f"Unknown entropy target mode: {self.mode}")
