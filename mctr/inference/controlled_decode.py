"""
MCTR Phase 2: Controlled Decode
==================================
Implements the controlled intervention experiment (Section 17 of spec):

    Same input + same Ψ_(t-1) → compare:
      Condition A: Native attention
      Condition B: MCTR attention (full modulation)
      Condition C: Weakened attention modulation (β * 0.5)
      Condition D: Strengthened attention modulation (β * 2.0)

This allows causal separation of Ψ → attention → output.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from mctr.control.attention_controller import AttentionControlState, AttentionControlMode
from mctr.attention.attention_metrics import (
    AttentionMetrics,
    compute_attention_entropy,
    compute_attention_js_divergence,
)


@dataclass
class InterventionResult:
    """Result from one intervention condition at a single step."""
    condition: str
    logits:       torch.Tensor   # [vocab]
    probs:        torch.Tensor   # [vocab]
    entropy:      float
    confidence:   float
    attn_entropy: float
    attn_js_vs_native: float


def run_attention_intervention(
    model,
    input_ids: torch.Tensor,
    attn_state_native: AttentionControlState,
    attn_state_mctr:   AttentionControlState,
    hook_manager,
    temperature: float = 1.0,
    device: str = "cpu",
) -> Dict[str, InterventionResult]:
    """
    Run four intervention conditions for a single decoding step:
      A: Native (no modulation)
      B: MCTR (full modulation as from controller)
      C: Weakened (β * 0.5)
      D: Strengthened (β * 2.0)

    Parameters
    ----------
    model           : frozen Qwen model
    input_ids       : Tensor [1, seq_len]
    attn_state_mctr : full MCTR attention state
    hook_manager    : AttentionHookManager
    temperature     : float — applied to logits
    device          : str

    Returns
    -------
    Dict mapping condition name → InterventionResult
    """

    def _single_forward(mod_state: Optional[AttentionControlState]) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """Run one forward pass, return (logits, probs, attn_maps)."""
        with torch.no_grad():
            if mod_state is None:
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=False,
                    output_attentions=True,
                )
                maps = [
                    a.detach().cpu() if a is not None else None
                    for a in (outputs.attentions or [])
                ]
            else:
                with hook_manager.apply_modulation(mod_state):
                    outputs = model(
                        input_ids=input_ids,
                        output_hidden_states=False,
                        output_attentions=True,
                    )
                maps = list(hook_manager.modulated_attn_maps)

        logits = outputs.logits[:, -1, :].squeeze(0)  # [vocab]
        scaled = logits / max(temperature, 1e-6)
        probs  = F.softmax(scaled, dim=-1)
        return logits, probs, maps

    def _get_attn_entropy(maps: List) -> float:
        vals = []
        for m in maps:
            if m is not None:
                # m: [batch, n_heads, seq_q, seq_k]
                e = compute_attention_entropy(m.float()).mean().item()
                vals.append(e)
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _get_attn_js(maps_a: List, maps_b: List) -> float:
        vals = []
        for a, b in zip(maps_a, maps_b):
            if a is not None and b is not None:
                sq = min(a.shape[-1], b.shape[-1])
                js = compute_attention_js_divergence(
                    a.float()[..., :sq], b.float()[..., :sq]
                ).mean().item()
                vals.append(js)
        return float(sum(vals) / len(vals)) if vals else 0.0

    # Build weakened and strengthened states
    def _scale_modulation(state: AttentionControlState, scale: float) -> AttentionControlState:
        return AttentionControlState(
            modulation=state.modulation * scale,
            mode=state.mode,
            beta=state.beta * scale,
            raw_gate=state.raw_gate,
        )

    cond_a_logits, cond_a_probs, cond_a_maps = _single_forward(None)
    cond_b_logits, cond_b_probs, cond_b_maps = _single_forward(attn_state_mctr)
    cond_c_logits, cond_c_probs, cond_c_maps = _single_forward(_scale_modulation(attn_state_mctr, 0.5))
    cond_d_logits, cond_d_probs, cond_d_maps = _single_forward(_scale_modulation(attn_state_mctr, 2.0))

    def _result(name, logits, probs, maps, ref_maps=None):
        ent  = float(-(probs.clamp(min=1e-10) * probs.clamp(min=1e-10).log()).sum().item())
        conf = float(probs.max().item())
        ae   = _get_attn_entropy(maps)
        ajs  = _get_attn_js(ref_maps, maps) if ref_maps is not None else 0.0
        return InterventionResult(
            condition=name,
            logits=logits, probs=probs,
            entropy=ent, confidence=conf,
            attn_entropy=ae, attn_js_vs_native=ajs,
        )

    return {
        "A_native":       _result("A_native",      cond_a_logits, cond_a_probs, cond_a_maps),
        "B_mctr":         _result("B_mctr",         cond_b_logits, cond_b_probs, cond_b_maps, cond_a_maps),
        "C_weakened":     _result("C_weakened",     cond_c_logits, cond_c_probs, cond_c_maps, cond_a_maps),
        "D_strengthened": _result("D_strengthened", cond_d_logits, cond_d_probs, cond_d_maps, cond_a_maps),
    }
