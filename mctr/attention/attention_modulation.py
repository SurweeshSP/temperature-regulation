"""
MCTR Phase 2: Attention Modulation
=====================================
Functional API for applying bounded attention modulation to a set of
attention logit tensors without touching the Transformer internals directly.

Used when the caller has direct access to QK^T logits (e.g., in controlled
decode experiments) or when testing modulation in isolation.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from mctr.control.attention_controller import AttentionControlState, AttentionControlMode


def apply_attention_modulation(
    attn_logits: torch.Tensor,
    attn_state: AttentionControlState,
    layer_idx: int = 0,
) -> torch.Tensor:
    """
    Apply bounded additive modulation to raw attention logits.

    Parameters
    ----------
    attn_logits : Tensor [batch, n_heads, seq_q, seq_k]
        Raw (pre-softmax) attention scores: QK^T / sqrt(d_k)
    attn_state  : AttentionControlState
        Output of AttentionController.
    layer_idx   : int
        Which layer index (needed for LEVEL2 / LEVEL3).

    Returns
    -------
    Tensor [batch, n_heads, seq_q, seq_k]
        Modulated attention logits, still pre-softmax.
    """
    if attn_state.mode == AttentionControlMode.NONE:
        return attn_logits

    mod = attn_state.modulation  # shape depends on mode

    if attn_state.mode == AttentionControlMode.LEVEL1_GLOBAL:
        # mod: [batch] → [batch, 1, 1, 1]
        delta = mod[:, None, None, None]

    elif attn_state.mode == AttentionControlMode.LEVEL2_LAYER:
        # mod: [batch, n_layers] → pick layer → [batch, 1, 1, 1]
        delta = mod[:, layer_idx, None, None, None] if mod.dim() >= 2 else mod[:, None, None, None]

    elif attn_state.mode == AttentionControlMode.LEVEL3_HEAD:
        # mod: [batch, n_layers, n_heads] → pick layer → [batch, n_heads, 1, 1]
        delta = mod[:, layer_idx, :, None, None]  # [batch, n_heads, 1, 1]

    else:
        return attn_logits

    modulated = attn_logits + delta.to(attn_logits.device, dtype=attn_logits.dtype)

    # Safety assertion
    assert torch.all(torch.isfinite(modulated)), \
        f"apply_attention_modulation: non-finite logits at layer {layer_idx}!"

    return modulated


def compute_attention_from_logits(
    attn_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute attention weights from logits.

    Parameters
    ----------
    attn_logits     : Tensor [batch, n_heads, seq_q, seq_k]
    attention_mask  : Optional Tensor [batch, 1, seq_q, seq_k], additive mask.

    Returns
    -------
    Tensor [batch, n_heads, seq_q, seq_k], summing to 1 along last dim.
    """
    if attention_mask is not None:
        attn_logits = attn_logits + attention_mask

    attn_weights = F.softmax(attn_logits, dim=-1)

    assert torch.all(torch.isfinite(attn_weights)), \
        "compute_attention_from_logits: non-finite attention weights!"
    attn_sum = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-4), \
        "compute_attention_from_logits: attention weights do not sum to 1!"

    return attn_weights
