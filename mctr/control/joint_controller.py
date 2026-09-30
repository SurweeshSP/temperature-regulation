"""
MCTR Phase 2: Joint Controller
================================
A single nn.Module that maps m_(t-1) to both temperature and attention control.

Architecture:
    Ψ_(t-1)  →  MetaEvaluator M_φ  →  m_(t-1)
                                         │
                          ┌──────────────┤
                          │              │
                   Temperature         Attention
                      Head              Head
                          │              │
                         T_t           ΔA_t

Temporal contract: always uses Ψ_(t-1), never Ψ_t.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional

from .attention_controller import AttentionController, AttentionControlState, AttentionControlMode


@dataclass
class JointControlOutput:
    """Combined output of the joint Phase 2 controller."""
    temperature: torch.Tensor          # [batch], bounded [T_min, T_max]
    attention: AttentionControlState   # Attention modulation
    meta_score: torch.Tensor           # [batch], correctness prediction probability


class JointController(nn.Module):
    """
    Joint Phase 2 controller that simultaneously produces:
      - Adaptive temperature T_t (using Phase 1 temperature head)
      - Adaptive attention modulation ΔA_t

    Shared representation from m_(t-1), separate linear heads.

    Mode flags
    ----------
    enable_temperature : bool  — if False, uses fixed baseline_temp
    enable_attention   : bool  — if False, returns zero modulation
    """

    def __init__(
        self,
        meta_dim: int,
        config: Dict[str, Any],
        n_layers: int = 24,
        n_heads: int  = 16,
    ):
        super().__init__()
        self.enable_temperature = config.get("enable_temperature", True)
        self.enable_attention   = config.get("enable_attention",   True)
        self.t_min = float(config.get("temperature_min", 0.3))
        self.t_max = float(config.get("temperature_max", 1.2))

        hidden_dim = int(config.get("controller_hidden_dim", 64))

        # ── Shared trunk ──────────────────────────────────────────────────────
        self.shared_trunk = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # ── Temperature head ──────────────────────────────────────────────────
        self.temperature_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # ── Attention controller ──────────────────────────────────────────────
        # Passes hidden_dim as meta_dim so it receives the shared representation
        attn_cfg = dict(config)
        attn_cfg["attention_hidden_dim"] = max(16, hidden_dim // 2)
        self.attention_controller = AttentionController(
            meta_dim=hidden_dim,
            config=attn_cfg,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        # ── Correctness prediction head (from meta-evaluator) ─────────────────
        self.correctness_head = nn.Linear(hidden_dim, 1)

    def _bound_temperature(self, raw_t: torch.Tensor) -> torch.Tensor:
        """T_t = T_min + (T_max - T_min) * sigmoid(raw_t)"""
        return self.t_min + (self.t_max - self.t_min) * torch.sigmoid(raw_t)

    def forward(
        self,
        m_t: torch.Tensor,
        baseline_temp: Optional[float] = None,
    ) -> JointControlOutput:
        """
        Parameters
        ----------
        m_t          : Tensor [batch, meta_dim]  — meta-evaluator embedding of Ψ_(t-1)
        baseline_temp: float or None             — if set, override temperature head

        Returns
        -------
        JointControlOutput
        """
        shared = self.shared_trunk(m_t)  # [batch, hidden_dim]

        # ── Temperature ───────────────────────────────────────────────────────
        if baseline_temp is not None or not self.enable_temperature:
            bt = baseline_temp if baseline_temp is not None else 1.0
            T_t = torch.full(
                (m_t.shape[0],), bt,
                device=m_t.device, dtype=m_t.dtype
            )
        else:
            raw_t = self.temperature_head(shared).squeeze(-1)  # [batch]
            T_t   = self._bound_temperature(raw_t)

        # ── Attention ─────────────────────────────────────────────────────────
        if not self.enable_attention:
            zero = torch.zeros(m_t.shape[0], device=m_t.device, dtype=m_t.dtype)
            attn_state = AttentionControlState(
                modulation=zero,
                mode=AttentionControlMode.NONE,
                beta=0.0,
                raw_gate=zero + 0.5,
            )
        else:
            attn_state = self.attention_controller(shared)

        # ── Correctness prediction ─────────────────────────────────────────────
        meta_score = torch.sigmoid(self.correctness_head(shared).squeeze(-1))

        # Runtime guards
        assert torch.all(torch.isfinite(T_t)), "JointController: non-finite temperature!"
        assert torch.all(T_t >= self.t_min - 1e-4), f"Temperature below T_min={self.t_min}"
        assert torch.all(T_t <= self.t_max + 1e-4), f"Temperature above T_max={self.t_max}"

        return JointControlOutput(
            temperature=T_t,
            attention=attn_state,
            meta_score=meta_score,
        )
