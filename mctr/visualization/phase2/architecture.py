"""
MCTR Phase 2: Architecture Diagram Generator
Uses matplotlib to create a publication-quality Phase 2 architecture diagram.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


def generate_phase2_architecture_diagram(save_path: str):
    """
    Generate Figure 1 / 2: Phase 2 MCTR architecture.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 12), facecolor="#0f172a")
    fig.suptitle(
        "MCTR Phase 2 Architecture\nMeta-Cognitive Attention Regulation",
        fontsize=16, color="#e2e8f0", y=0.98, fontweight="bold",
    )

    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")

    # ── Left panel: Full Phase 2 architecture ─────────────────────────────
    ax = axes[0]
    ax.set_title("Phase 2: Full MCTR-TA Pipeline", color="#e2e8f0", fontsize=13, pad=10)

    def _box(ax, x, y, w, h, label, color, textcolor="#0f172a", fontsize=9):
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            linewidth=1.5,
            edgecolor=color,
            facecolor=color + "33",
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                color="#e2e8f0", fontsize=fontsize, fontweight="bold",
                wrap=True, multialignment="center")

    def _arrow(ax, x1, y1, x2, y2, color="#64748b"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # Nodes (Phase 2 causal pipeline)
    nodes = [
        (5, 11.0, 3.5, 0.7, "Psi_(t-1)\n[C, U, N, K, S, H_norm]", "#6366f1"),
        (5,  9.5, 3.0, 0.7, "Meta-Evaluator  M_phi\nMLP: 6 -> 64 -> 32", "#8b5cf6"),
        (5,  8.0, 3.0, 0.7, "Shared Controller Trunk\n32 -> 64 -> 64", "#a855f7"),
        (2.5, 6.5, 2.2, 0.65, "Temperature Head\n-> T_t in [0.3, 1.2]", "#3b82f6"),
        (7.5, 6.5, 2.2, 0.65, "Attention Head\n-> DeltaA in [-beta, +beta]", "#10b981"),
        (5,   5.0, 3.5, 0.7, "Regulated Qwen2.5-0.5B\n(FROZEN backbone)", "#ef4444"),
        (5,   3.5, 3.0, 0.7, "Softmax(z/T_t)\n-> P_t", "#f59e0b"),
        (5,   2.0, 3.0, 0.7, "Stochastic Sample\ny_t ~ Categorical(P_t)", "#06b6d4"),
        (5,   0.7, 3.5, 0.65, "Psi_t\n[new meta-state]", "#6366f1"),
    ]

    for x, y, w, h, label, color in nodes:
        _box(ax, x, y, w, h, label, color)

    # Arrows
    arrows = [
        (5, 10.65, 5, 9.85),
        (5, 9.15,  5, 8.35),
        (5, 7.65,  2.5, 6.83),  # to temp head
        (5, 7.65,  7.5, 6.83),  # to attn head
        (2.5, 6.17, 5, 5.35),   # temp head to Qwen
        (7.5, 6.17, 5, 5.35),   # attn head to Qwen
        (5, 4.65,  5, 3.85),
        (5, 3.15,  5, 2.35),
        (5, 1.65,  5, 1.02),
        (5, 0.38,  8.5, 0.38),   # recurrence horizontal
    ]
    for x1, y1, x2, y2 in arrows:
        _arrow(ax, x1, y1, x2, y2, "#64748b")

    # Recurrence arrow
    ax.annotate("", xy=(8.5, 10.98), xytext=(8.5, 0.38),
                arrowprops=dict(arrowstyle="->", color="#6366f1", lw=1.5))
    ax.text(9.0, 5.5, "recurrence\nPsi_(t) ->\nnext step", ha="center",
            color="#6366f1", fontsize=7.5, style="italic")

    # ── Right panel: Attention Modulation Levels ─────────────────────────
    ax2 = axes[1]
    ax2.set_title("Attention Control Levels", color="#e2e8f0", fontsize=13, pad=10)

    levels = [
        (5, 10.0, 4.0, 0.9, "LEVEL 1 - Global\nalpha_t : 1 scalar -> all layers & heads", "#3b82f6"),
        (5, 8.0,  4.0, 0.9, "LEVEL 2 - Layer-wise\nalpha_t^(l) : 1 scalar per layer", "#10b981"),
        (5, 6.0,  4.0, 0.9, "LEVEL 3 - Head-wise\nalpha_t^(l,h) : 1 per (layer,head)", "#f59e0b"),
        (5, 4.0,  4.5, 1.2,
         "Modulation Formula\n"
         "g_t = sigmoid(r_w(m_t))  in (0,1)\n"
         "DeltaA = beta*(2g-1)  in [-beta,+beta]\n"
         "A_logits' = A_logits + DeltaA", "#8b5cf6"),
        (5, 2.0, 4.5, 1.0,
         "Safety Guarantees\n"
         "[x] DeltaA bounded  [x] sum(A)=1\n"
         "[x] no NaN/Inf      [x] causal ordering", "#ef4444"),
    ]

    for x, y, w, h, label, color in levels:
        _box(ax2, x, y, w, h, label, color, fontsize=8.5)

    ax2.text(5, 0.6,
             "Temporal Causality: Psi_(t-1) -> controller -> A_t -> P_t -> Psi_t",
             ha="center", color="#94a3b8", fontsize=9, style="italic")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Architecture diagram saved: {save_path}")
