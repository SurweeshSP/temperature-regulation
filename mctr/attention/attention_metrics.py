"""
MCTR Phase 2: Attention Metrics
==================================
Quantitative metrics for measuring and comparing attention distributions.

Implements all Phase 2 required metrics:
  1.  Attention entropy         H_attn = -Σ A_i log A_i
  2.  Attention concentration   max_i A_i
  3.  Attention effective rank  exp(H_attn)
  4.  Layer-wise entropy
  5.  Head-wise entropy
  6.  Attention JS divergence   JS(A_native, A_mctr)
  7.  Attention KL divergence   KL(A_native || A_mctr)
  8.  L1 distance               ||A_native - A_mctr||_1
  9.  Cosine similarity
  10. Modulation magnitude       ||ΔA||
  11. Sparsity (Hoyer measure)
  12. Temporal attention stability
  13. Controller responsiveness  ΔA / ΔΨ
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


EPS = 1e-10  # numerical floor for log computations


def compute_attention_entropy(
    attn_weights: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Compute per-head attention entropy.

    Parameters
    ----------
    attn_weights : Tensor [batch, n_heads, seq_q, seq_k]

    Returns
    -------
    Tensor [batch, n_heads, seq_q]  — entropy at each query position
    """
    p = attn_weights.clamp(min=eps)
    H = -(p * p.log()).sum(dim=-1)  # [batch, n_heads, seq_q]
    return H


def compute_attention_concentration(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    max_i A_i — measure of attention focus.

    Returns Tensor [batch, n_heads, seq_q]
    """
    return attn_weights.max(dim=-1).values


def compute_attention_effective_rank(attn_weights: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Effective rank = exp(H_attn), the exponent of entropy.
    Ranges from 1 (fully concentrated) to seq_k (uniform).

    Returns Tensor [batch, n_heads, seq_q]
    """
    return torch.exp(compute_attention_entropy(attn_weights, eps))


def compute_attention_js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Jensen-Shannon divergence between two attention distributions.

    Parameters
    ----------
    p, q : Tensor [batch, n_heads, seq_q, seq_k]

    Returns
    -------
    Tensor [batch, n_heads, seq_q]
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    m = m.clamp(min=eps)

    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def compute_attention_kl(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    KL(p || q) — asymmetric.

    Returns Tensor [batch, n_heads, seq_q]
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p / q).log()).sum(dim=-1)


def compute_attention_l1(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    L1 distance ||p - q||_1.

    Returns Tensor [batch, n_heads, seq_q]
    """
    return (p - q).abs().sum(dim=-1)


def compute_attention_cosine(p: torch.Tensor, q: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Cosine similarity between two attention distributions.

    Returns Tensor [batch, n_heads, seq_q]
    """
    return F.cosine_similarity(p, q, dim=-1)


def compute_attention_sparsity(attn_weights: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Hoyer sparsity measure: 1 - (||A||_1 / (√seq_k * ||A||_2))
    Range [0, 1], 1 = maximally sparse.

    Returns Tensor [batch, n_heads, seq_q]
    """
    l1 = attn_weights.sum(dim=-1)               # always 1 for valid dist
    l2 = torch.sqrt((attn_weights ** 2).sum(dim=-1).clamp(min=eps))
    seq_k = attn_weights.shape[-1]
    return 1.0 - l1 / (math.sqrt(seq_k) * l2 + eps)


def compute_temporal_attention_stability(
    attn_prev: Optional[torch.Tensor],
    attn_curr: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Temporal stability: JS divergence between consecutive step attention maps.

    Returns Tensor [batch, n_heads, seq_q] or None if attn_prev is None.
    """
    if attn_prev is None:
        return None
    return compute_attention_js_divergence(attn_prev, attn_curr)


def compute_modulation_magnitude(
    modulation: torch.Tensor,
) -> float:
    """L2 norm of the modulation signal (scalar)."""
    return modulation.norm(p=2).item()


def compute_controller_responsiveness(
    delta_modulation: torch.Tensor,
    delta_psi: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    ΔA / ΔΨ — approximate responsiveness.

    Parameters
    ----------
    delta_modulation : Tensor — change in modulation ΔA_(t) - ΔA_(t-1)
    delta_psi        : Tensor — change in meta-state Ψ_t - Ψ_(t-1)

    Returns
    -------
    float ratio
    """
    num = delta_modulation.norm(p=2).item()
    den = delta_psi.norm(p=2).item() + eps
    return num / den


import math  # needed for compute_attention_sparsity — must be at module level


@dataclass
class AttentionMetrics:
    """
    Per-step attention metrics for one generation step.

    All tensors are CPU numpy arrays (for easy logging).
    """
    # Per-head entropies averaged over query positions: [n_layers, n_heads]
    entropy_native:    Optional[np.ndarray] = None
    entropy_modulated: Optional[np.ndarray] = None

    # Averaged over query positions: [n_layers, n_heads]
    concentration_native:    Optional[np.ndarray] = None
    concentration_modulated: Optional[np.ndarray] = None

    # JS divergence between native and modulated: [n_layers, n_heads]
    js_divergence: Optional[np.ndarray] = None

    # L1 distance: [n_layers, n_heads]
    l1_distance: Optional[np.ndarray] = None

    # Modulation magnitude (scalar)
    modulation_magnitude: float = 0.0

    # Controller responsiveness (scalar)
    responsiveness: float = 0.0

    # Temporal stability JS (scalar, averaged across layers/heads)
    temporal_stability: float = 0.0

    @classmethod
    def compute(
        cls,
        native_maps:    List[Optional[torch.Tensor]],
        modulated_maps: List[Optional[torch.Tensor]],
        modulation:     Optional[torch.Tensor] = None,
        prev_native_maps: Optional[List[Optional[torch.Tensor]]] = None,
        delta_psi:      Optional[torch.Tensor] = None,
        prev_modulation: Optional[torch.Tensor] = None,
    ) -> "AttentionMetrics":
        """
        Compute all metrics from lists of per-layer attention maps.

        Parameters
        ----------
        native_maps    : [n_layers] of Tensor [batch, n_heads, seq_q, seq_k] or None
        modulated_maps : same shape
        modulation     : Tensor — the controller's raw modulation signal
        prev_native_maps : previous step's native maps (for stability)
        delta_psi      : change in meta-state (for responsiveness)
        prev_modulation : previous step modulation (for responsiveness)
        """
        ent_n_list, ent_m_list = [], []
        conc_n_list, conc_m_list = [], []
        js_list, l1_list = [], []

        for nat, mod in zip(native_maps, modulated_maps):
            if nat is None:
                continue

            # Collapse batch dim (take item 0) and average over seq_q
            nat_ = nat[0].float()   # [n_heads, seq_q, seq_k]
            mod_ = mod[0].float() if mod is not None else nat_

            ent_n = compute_attention_entropy(nat_.unsqueeze(0)).squeeze(0).mean(dim=-1).numpy()  # [n_heads]
            ent_m = compute_attention_entropy(mod_.unsqueeze(0)).squeeze(0).mean(dim=-1).numpy()

            conc_n = compute_attention_concentration(nat_.unsqueeze(0)).squeeze(0).mean(dim=-1).numpy()
            conc_m = compute_attention_concentration(mod_.unsqueeze(0)).squeeze(0).mean(dim=-1).numpy()

            js = compute_attention_js_divergence(nat_.unsqueeze(0), mod_.unsqueeze(0)).squeeze(0).mean(dim=-1).numpy()
            l1 = compute_attention_l1(nat_.unsqueeze(0), mod_.unsqueeze(0)).squeeze(0).mean(dim=-1).numpy()

            ent_n_list.append(ent_n)
            ent_m_list.append(ent_m)
            conc_n_list.append(conc_n)
            conc_m_list.append(conc_m)
            js_list.append(js)
            l1_list.append(l1)

        def _stack(lst):
            return np.stack(lst, axis=0) if lst else None  # [n_layers, n_heads]

        # Modulation magnitude
        mag = compute_modulation_magnitude(modulation) if modulation is not None else 0.0

        # Responsiveness
        resp = 0.0
        if modulation is not None and prev_modulation is not None and delta_psi is not None:
            dm = modulation - prev_modulation
            resp = compute_controller_responsiveness(dm, delta_psi)

        # Temporal stability
        stab = 0.0
        if prev_native_maps is not None:
            stab_vals = []
            for nat, prev_nat in zip(native_maps, prev_native_maps):
                if nat is not None and prev_nat is not None:
                    nat_ = nat[0].float().unsqueeze(0)
                    prev_ = prev_nat[0].float().unsqueeze(0)
                    # pad/trim seq_k to match
                    sq = min(nat_.shape[-1], prev_.shape[-1])
                    s = compute_attention_js_divergence(nat_[..., :sq], prev_[..., :sq]).mean().item()
                    stab_vals.append(s)
            stab = float(np.mean(stab_vals)) if stab_vals else 0.0

        return cls(
            entropy_native=_stack(ent_n_list),
            entropy_modulated=_stack(ent_m_list),
            concentration_native=_stack(conc_n_list),
            concentration_modulated=_stack(conc_m_list),
            js_divergence=_stack(js_list),
            l1_distance=_stack(l1_list),
            modulation_magnitude=mag,
            responsiveness=resp,
            temporal_stability=stab,
        )

    def mean_entropy_native(self) -> float:
        return float(np.mean(self.entropy_native)) if self.entropy_native is not None else 0.0

    def mean_entropy_modulated(self) -> float:
        return float(np.mean(self.entropy_modulated)) if self.entropy_modulated is not None else 0.0

    def mean_js(self) -> float:
        return float(np.mean(self.js_divergence)) if self.js_divergence is not None else 0.0

    def mean_concentration_native(self) -> float:
        return float(np.mean(self.concentration_native)) if self.concentration_native is not None else 0.0

    def mean_concentration_modulated(self) -> float:
        return float(np.mean(self.concentration_modulated)) if self.concentration_modulated is not None else 0.0

    def to_dict(self) -> dict:
        return {
            "attention_entropy_native":       self.mean_entropy_native(),
            "attention_entropy_modulated":    self.mean_entropy_modulated(),
            "attention_concentration_native": self.mean_concentration_native(),
            "attention_concentration_mod":    self.mean_concentration_modulated(),
            "attention_js_divergence":        self.mean_js(),
            "modulation_magnitude":           self.modulation_magnitude,
            "responsiveness":                 self.responsiveness,
            "temporal_stability":             self.temporal_stability,
        }
