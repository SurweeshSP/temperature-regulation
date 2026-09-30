"""
MCTR Phase 2: Attention Visualization
========================================
Publication-quality attention plots for Phase 2.

Implements:
  Fig 7:  Attention entropy trajectory
  Fig 8:  Native vs MCTR attention maps (heatmap)
  Fig 9:  Attention redistribution / JS divergence
  Fig 10: Meta-state vs attention-control response
  + Correlation matrix for state redundancy analysis
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional, Dict, Any
import pandas as pd


_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#e2e8f0",
    "axes.labelcolor":  "#e2e8f0",
    "xtick.color":      "#e2e8f0",
    "ytick.color":      "#e2e8f0",
    "text.color":       "#e2e8f0",
    "grid.color":       "#2d3748",
    "grid.alpha":       0.4,
}

_PALETTE = {
    "native":  "#64748b",
    "mctr_t":  "#3b82f6",
    "mctr_a":  "#10b981",
    "mctr_ta": "#f59e0b",
    "fixed":   "#94a3b8",
}


def _apply_dark_style():
    plt.rcParams.update(_STYLE)
    plt.rcParams["font.family"] = "DejaVu Sans"


def save_fig(fig, path: str, dpi: int = 200):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"      Saved: {path}")


# ── Figure 7: Attention Entropy Trajectory ───────────────────────────────────

def plot_attention_entropy_trajectory(
    trajectories_by_condition: Dict[str, List[float]],
    save_path: str,
):
    """
    Attention entropy over generation steps for each condition.
    One line per condition, shaded band where available.
    """
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=_STYLE["figure.facecolor"])
    ax.set_facecolor(_STYLE["axes.facecolor"])

    for cond, traj in trajectories_by_condition.items():
        if not traj:
            continue
        color = _PALETTE.get(cond.lower().replace("-", "_"), "#e2e8f0")
        steps = np.arange(len(traj))
        ax.plot(steps, traj, color=color, lw=2, label=cond)

    ax.set_xlabel("Generation Step", fontsize=12)
    ax.set_ylabel("Mean Attention Entropy", fontsize=12)
    ax.set_title("Figure 7: Attention Entropy Trajectory — Native vs MCTR", fontsize=14, color="#e2e8f0")
    ax.legend(framealpha=0.3, fontsize=10)
    ax.grid(True, alpha=0.3)
    save_fig(fig, save_path)


# ── Figure 8: Native vs MCTR Attention Maps ──────────────────────────────────

def plot_attention_heatmaps(
    native_map: np.ndarray,
    mctr_map:   np.ndarray,
    layer_idx:  int,
    head_idx:   int,
    step:       int,
    save_path:  str,
    tokens:     Optional[List[str]] = None,
):
    """
    Side-by-side attention heatmap: native vs MCTR.

    native_map, mctr_map : [seq_q, seq_k] numpy arrays
    """
    _apply_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=_STYLE["figure.facecolor"])

    def _heatmap(ax, data, title):
        im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0, vmax=data.max())
        ax.set_title(title, fontsize=11, color="#e2e8f0")
        ax.set_xlabel("Key Position", fontsize=9, color="#e2e8f0")
        ax.set_ylabel("Query Position", fontsize=9, color="#e2e8f0")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _heatmap(axes[0], native_map, f"Native Attention\n(Layer {layer_idx}, Head {head_idx}, Step {step})")
    _heatmap(axes[1], mctr_map,   f"MCTR Attention\n(Layer {layer_idx}, Head {head_idx}, Step {step})")
    diff = mctr_map - native_map
    im3  = axes[2].imshow(diff, cmap="RdBu_r", aspect="auto",
                          vmin=-abs(diff).max(), vmax=abs(diff).max())
    axes[2].set_title(f"Difference (MCTR - Native)", fontsize=11, color="#e2e8f0")
    axes[2].set_xlabel("Key Position", fontsize=9, color="#e2e8f0")
    axes[2].set_ylabel("Query Position", fontsize=9, color="#e2e8f0")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Figure 8: Native vs MCTR Attention Maps\n"
        f"[Selection: Layer {layer_idx}, Head {head_idx}, Step {step}]",
        fontsize=13, color="#e2e8f0", y=1.02,
    )
    save_fig(fig, save_path)


# ── Figure 9: Attention Redistribution / JS Divergence ──────────────────────

def plot_attention_js_distribution(
    df: pd.DataFrame,
    save_path: str,
):
    """
    Distribution of per-example JS divergence (native vs MCTR-A / MCTR-TA).
    """
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_STYLE["figure.facecolor"])
    ax.set_facecolor(_STYLE["axes.facecolor"])

    conditions = [c for c in df["condition"].unique() if "mctr_a" in c.lower() or "mctr-a" in c.lower()
                  or "mctr_ta" in c.lower() or "mctr-ta" in c.lower()]

    if not conditions:
        # Show all
        conditions = df["condition"].unique()

    for cond in conditions:
        sub = df[df["condition"] == cond]["attention_js"].dropna()
        if len(sub) > 0:
            color = _PALETTE.get(cond.lower().replace("-", "_"), "#e2e8f0")
            ax.hist(sub, bins=30, alpha=0.5, color=color, label=cond, density=True)

    ax.set_xlabel("Attention JS Divergence (Native vs MCTR)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Figure 9: Attention Redistribution — JS Divergence Distribution", fontsize=14, color="#e2e8f0")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.3)
    save_fig(fig, save_path)


# ── Figure 10: Meta-State vs Attention Control Response ──────────────────────

def plot_meta_vs_attention_response(
    df: pd.DataFrame,
    save_path: str,
):
    """
    Scatter: Ψ-derived meta_score vs attention_js to visualise controller response.
    """
    _apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_STYLE["figure.facecolor"])

    for ax, (x_col, x_label) in zip(axes, [
        ("entropy",    "Predictive Entropy"),
        ("meta_score", "Meta-Score"),
    ]):
        ax.set_facecolor(_STYLE["axes.facecolor"])
        conditions = [c for c in df["condition"].unique() if "mctr_a" in c.lower()
                      or "mctr-a" in c.lower() or "mctr_ta" in c.lower() or "mctr-ta" in c.lower()]
        if not conditions:
            conditions = df["condition"].unique()[:2]

        for cond in conditions:
            sub = df[df["condition"] == cond].dropna(subset=[x_col, "attention_js"])
            if len(sub) > 0:
                color = _PALETTE.get(cond.lower().replace("-", "_"), "#e2e8f0")
                ax.scatter(sub[x_col], sub["attention_js"], alpha=0.5, s=20, c=color, label=cond)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Attention JS Divergence", fontsize=11)
        ax.set_title(f"{x_label} → Attention Control", fontsize=12, color="#e2e8f0")
        ax.legend(framealpha=0.3, fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 10: Meta-State → Attention Control Response", fontsize=14, color="#e2e8f0")
    save_fig(fig, save_path)


# ── Figure 11: Accuracy Comparison ───────────────────────────────────────────

def plot_accuracy_comparison(
    metrics_by_condition: Dict[str, Dict[str, float]],
    save_path: str,
):
    """Bar chart: accuracy for Native / Fixed-T / MCTR-T / MCTR-A / MCTR-TA."""
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=_STYLE["figure.facecolor"])
    ax.set_facecolor(_STYLE["axes.facecolor"])

    conds  = list(metrics_by_condition.keys())
    accs   = [metrics_by_condition[c].get("accuracy", 0) * 100 for c in conds]
    colors = [_PALETTE.get(c.lower().replace("-", "_"), "#94a3b8") for c in conds]

    bars = ax.bar(conds, accs, color=colors, edgecolor="#e2e8f0", linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#e2e8f0",
        )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Figure 11: Accuracy — Native vs MCTR-T vs MCTR-A vs MCTR-TA", fontsize=14, color="#e2e8f0")
    ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(accs) * 1.15 if accs else 100)
    save_fig(fig, save_path)


# ── Figure 12: ECE / Brier Comparison ────────────────────────────────────────

def plot_calibration_comparison(
    metrics_by_condition: Dict[str, Dict[str, float]],
    save_path: str,
):
    _apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_STYLE["figure.facecolor"])

    conds = list(metrics_by_condition.keys())
    eces  = [metrics_by_condition[c].get("ece", 0)   for c in conds]
    briers= [metrics_by_condition[c].get("brier", 0) for c in conds]
    colors= [_PALETTE.get(c.lower().replace("-", "_"), "#94a3b8") for c in conds]

    for ax, vals, ylabel, title in zip(
        axes,
        [eces, briers],
        ["ECE", "Brier Score"],
        ["ECE (↓ better)", "Brier Score (↓ better)"],
    ):
        ax.set_facecolor(_STYLE["axes.facecolor"])
        ax.bar(conds, vals, color=colors, edgecolor="#e2e8f0", linewidth=0.5, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, color="#e2e8f0")
        ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure 12: Calibration Metrics — ECE & Brier Score", fontsize=14, color="#e2e8f0")
    save_fig(fig, save_path)


# ── Figure 13: Ablation Study ────────────────────────────────────────────────

def plot_ablation_study(
    metrics_by_condition: Dict[str, Dict[str, float]],
    save_path: str,
):
    """Grouped bar chart for ablation results."""
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=_STYLE["figure.facecolor"])
    ax.set_facecolor(_STYLE["axes.facecolor"])

    conds  = list(metrics_by_condition.keys())
    accs   = [metrics_by_condition[c].get("accuracy", 0) * 100 for c in conds]
    colors = [_PALETTE.get(c.lower().replace("-", "_"), "#94a3b8") for c in conds]

    ax.bar(conds, accs, color=colors, edgecolor="#e2e8f0", linewidth=0.5, alpha=0.85)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Figure 13: Ablation Study", fontsize=14, color="#e2e8f0")
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, save_path)


# ── Figure 14: Efficiency ────────────────────────────────────────────────────

def plot_efficiency_comparison(
    metrics_by_condition: Dict[str, Dict[str, float]],
    save_path: str,
):
    _apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_STYLE["figure.facecolor"])

    conds    = list(metrics_by_condition.keys())
    latencies= [metrics_by_condition[c].get("mean_latency", 0) for c in conds]
    vrams    = [metrics_by_condition[c].get("peak_vram", 0)    for c in conds]
    colors   = [_PALETTE.get(c.lower().replace("-", "_"), "#94a3b8") for c in conds]

    for ax, vals, ylabel, title in zip(
        axes,
        [latencies, vrams],
        ["Latency (s/example)", "Peak VRAM (MB)"],
        ["Per-Example Latency", "Peak VRAM Usage"],
    ):
        ax.set_facecolor(_STYLE["axes.facecolor"])
        ax.bar(conds, vals, color=colors, edgecolor="#e2e8f0", linewidth=0.5, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, color="#e2e8f0")
        ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure 14: Efficiency — Latency & VRAM", fontsize=14, color="#e2e8f0")
    save_fig(fig, save_path)


# ── State Redundancy Plot ─────────────────────────────────────────────────────

def plot_state_correlation_matrix(
    corr_df: pd.DataFrame,
    save_path: str,
    title: str = "Figure 4: Meta-State Correlation Matrix",
):
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_STYLE["figure.facecolor"])
    ax.set_facecolor(_STYLE["axes.facecolor"])

    data = corr_df.values
    labels = list(corr_df.columns)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(data[i, j]) > 0.5 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=13, color="#e2e8f0")

    # Annotate known redundancies
    ax.text(0.02, -0.15,
            "Note: U_t = Ĥ_t by construction (uncertainty = normalized entropy). "
            "High |Corr| pairs indicate near-redundant state dimensions.",
            transform=ax.transAxes, fontsize=7, color="#94a3b8",
            wrap=True)

    save_fig(fig, save_path)
