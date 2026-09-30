"""
MCTR Phase 2: Attention Controller
===================================
Maps meta-evaluator embedding m_(t-1) → attention control signal A_t.

Implements three complexity levels:
  LEVEL 1 — Global scalar α_t  (one scalar for all layers & heads)
  LEVEL 2 — Layer-wise  α_t^(l)
  LEVEL 3 — Head-wise   α_t^(l,h)

In all cases the controller produces a *bounded* modulation signal:
  ΔA ∈ [-δ, +δ]

applied as an additive bias to the pre-softmax attention logits:
  A_logits'^(l,h) = A_logits^(l,h) + β * (2·g^(l,h) - 1)

where g^(l,h) = sigmoid(r_ω(m_t)) ∈ (0,1) and β ≤ β_max.

Causal contract: only Ψ_(t-1) is consumed; the controller never sees
the attention map produced at step t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class AttentionControlMode(Enum):
    """Complexity level for attention modulation."""
    LEVEL1_GLOBAL = "global"       # one scalar per step
    LEVEL2_LAYER  = "layer"        # one scalar per layer
    LEVEL3_HEAD   = "head"         # one scalar per (layer, head)
    NONE          = "none"         # pass-through (ablation)


@dataclass
class AttentionControlState:
    """
    Output of AttentionController.forward().

    Fields
    ------
    modulation : Tensor, shape depends on mode
        LEVEL1 : [batch]
        LEVEL2 : [batch, n_layers]
        LEVEL3 : [batch, n_layers, n_heads]
    mode       : AttentionControlMode
    beta       : float — modulation amplitude bound
    raw_gate   : Tensor, same shape as modulation — sigmoid pre-bound raw gate
    """
    modulation: torch.Tensor         # bounded ΔA signal
    mode: AttentionControlMode
    beta: float
    raw_gate: torch.Tensor           # g ∈ (0,1) before (2g-1) centering


class AttentionController(nn.Module):
    """
    Meta-cognitive attention controller Gω.

    Architecture
    ------------
    m_(t-1)  [meta_dim]
        → LayerNorm
        → Linear(meta_dim, hidden_dim)
        → GELU
        → Linear(hidden_dim, output_dim)
        → sigmoid  →  g ∈ (0,1)
        → 2g - 1   →  ΔA ∈ (-1,+1)
        → β * ΔA   →  bounded modulation ΔA ∈ [-β, +β]

    The backbone is never altered; modulation is injected externally
    into attention logits via hooks (see attention_hooks.py).
    """

    def __init__(
        self,
        meta_dim: int,
        config: Dict[str, Any],
        n_layers: int = 24,
        n_heads: int = 16,
    ):
        super().__init__()
        self.mode = AttentionControlMode(config.get("attention_control_mode", "global"))
        self.beta = float(config.get("attention_modulation_bound", 2.0))
        self.n_layers = n_layers
        self.n_heads  = n_heads
        hidden_dim = int(config.get("attention_hidden_dim", 64))

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, hidden_dim),
            nn.GELU(),
        )

        # Output head — size depends on mode
        if self.mode == AttentionControlMode.LEVEL1_GLOBAL:
            self.head = nn.Linear(hidden_dim, 1)
        elif self.mode == AttentionControlMode.LEVEL2_LAYER:
            self.head = nn.Linear(hidden_dim, n_layers)
        elif self.mode == AttentionControlMode.LEVEL3_HEAD:
            self.head = nn.Linear(hidden_dim, n_layers * n_heads)
        else:  # NONE
            self.head = nn.Identity()

    def forward(self, m_t: torch.Tensor) -> AttentionControlState:
        """
        Parameters
        ----------
        m_t : Tensor [batch, meta_dim]
            Output of the Meta-Evaluator from step t-1.

        Returns
        -------
        AttentionControlState
        """
        if self.mode == AttentionControlMode.NONE:
            batch = m_t.shape[0]
            zero = torch.zeros(batch, device=m_t.device, dtype=m_t.dtype)
            return AttentionControlState(
                modulation=zero,
                mode=self.mode,
                beta=0.0,
                raw_gate=zero + 0.5,
            )

        shared = self.trunk(m_t)              # [batch, hidden_dim]
        logits = self.head(shared)             # [batch, output_size]

        # Bounded gate: g ∈ (0,1)
        g = torch.sigmoid(logits)

        # Center: ΔA ∈ (-1, +1), then scale by β
        delta = self.beta * (2.0 * g - 1.0)  # ∈ [-β, +β]

        # Reshape for level 2 / 3
        if self.mode == AttentionControlMode.LEVEL1_GLOBAL:
            # [batch, 1] → squeeze to [batch]
            delta = delta.squeeze(-1)
            g     = g.squeeze(-1)
        elif self.mode == AttentionControlMode.LEVEL2_LAYER:
            pass  # [batch, n_layers]
        elif self.mode == AttentionControlMode.LEVEL3_HEAD:
            batch = m_t.shape[0]
            delta = delta.view(batch, self.n_layers, self.n_heads)
            g     = g.view(batch, self.n_layers, self.n_heads)

        # Runtime assertions
        assert torch.all(torch.isfinite(delta)), "AttentionController: non-finite modulation detected!"
        assert torch.all(delta.abs() <= self.beta + 1e-5), (
            f"AttentionController: modulation exceeded bound β={self.beta}!"
        )

        return AttentionControlState(
            modulation=delta,
            mode=self.mode,
            beta=self.beta,
            raw_gate=g,
        )
