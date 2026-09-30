"""MCTR Phase 2 Attention Components."""
from .attention_hooks import AttentionHookManager
from .attention_modulation import apply_attention_modulation
from .attention_metrics import AttentionMetrics, compute_attention_entropy, compute_attention_js_divergence

__all__ = [
    "AttentionHookManager",
    "apply_attention_modulation",
    "AttentionMetrics",
    "compute_attention_entropy",
    "compute_attention_js_divergence",
]
