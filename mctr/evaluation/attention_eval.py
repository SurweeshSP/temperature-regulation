"""
MCTR Phase 2: Attention Evaluation
=====================================
Higher-level evaluation utilities for Phase 2 attention metrics:
  - State redundancy analysis (Pearson/Spearman correlations)
  - Per-feature AUROC
  - H1-H8 hypothesis tests
  - Attention redistribution quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats as scipy_stats

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── Meta-State Redundancy Analysis ──────────────────────────────────────────

def compute_state_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix of all Ψ components.
    Reports correlations with explicit redundancy note.

    NOTE (Phase 2): The Phase 1 analysis found:
        Corr(C, H) ≈ -0.99  (confidence vs entropy — near-redundant)
        Corr(H, N) ≈  0.98  (entropy vs novelty)
        Corr(H, K) ≈  0.93  (entropy vs conflict)
        Corr(C, S) ≈  1.00  (confidence vs stability — effectively redundant)
        Corr(U, H) =  1.00  (uncertainty = normalized entropy by construction)

    This redundancy is preserved for backward compatibility.
    """
    state_cols = ["confidence", "entropy", "uncertainty", "novelty", "conflict", "stability"]
    available  = [c for c in state_cols if c in df.columns]
    if not available:
        return pd.DataFrame()

    corr_df = df[available].corr(method="pearson")
    return corr_df


def compute_spearman_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlations for meta-state components."""
    state_cols = ["confidence", "entropy", "uncertainty", "novelty", "conflict", "stability"]
    available  = [c for c in state_cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    return df[available].corr(method="spearman")


def compute_per_feature_auroc(df: pd.DataFrame) -> Dict[str, float]:
    """
    Per-feature AUROC for predicting correctness.
    Tests H1: Meta-state Ψ predicts correctness.
    """
    if not _HAS_SKLEARN:
        return {}
    if "correct" not in df.columns:
        return {}

    y = df["correct"].values
    if len(np.unique(y)) < 2:
        return {"note": "single class, AUROC undefined"}

    results = {}
    feature_cols = ["confidence", "entropy", "uncertainty", "novelty", "conflict", "stability", "meta_score"]
    for col in feature_cols:
        if col not in df.columns:
            continue
        x = df[col].values
        try:
            auroc = roc_auc_score(y, x)
        except Exception:
            auroc = float("nan")
        results[f"AUROC_{col}"] = float(auroc)

    # Full Ψ (just using meta_score as proxy, since it's the learned combination)
    if "meta_score" in df.columns:
        try:
            results["AUROC_full_psi"] = float(roc_auc_score(y, df["meta_score"].values))
        except Exception:
            results["AUROC_full_psi"] = float("nan")

    return results


# ── Hypothesis Testing ────────────────────────────────────────────────────────

def test_h1_psi_predicts_correctness(df: pd.DataFrame) -> Dict[str, Any]:
    """H1: Psi predicts correctness -- logistic correlation test."""
    aurocs = compute_per_feature_auroc(df)
    result = {
        "hypothesis": "H1: Meta-state Psi predicts correctness",
        "auroc_by_feature": aurocs,
        "conclusion": "NOT TESTED (insufficient data)"
    }
    # Filter to only numeric AUROC entries (skip 'note' key etc.)
    numeric_aurocs = {k: v for k, v in aurocs.items() if isinstance(v, float) and not (v != v)}  # exclude NaN
    if numeric_aurocs:
        best_feat = max(numeric_aurocs, key=lambda k: numeric_aurocs[k])
        best_val  = numeric_aurocs[best_feat]
        result["best_feature"]  = best_feat
        result["best_auroc"]    = best_val
        result["conclusion"]    = (
            f"Supported (AUROC={best_val:.3f})" if best_val > 0.6
            else f"Null result (best AUROC={best_val:.3f})"
        )
    elif "note" in aurocs:
        result["conclusion"] = f"Cannot compute AUROC: {aurocs['note']}"
    return result



def test_h4_attention_redistribution(
    df_native: pd.DataFrame,
    df_mctr_a: pd.DataFrame,
) -> Dict[str, Any]:
    """H4: MCTR-A produces measurable attention redistribution."""
    result = {"hypothesis": "H4: MCTR-A produces measurable attention redistribution"}

    if "attention_js" not in df_mctr_a.columns:
        result["conclusion"] = "No attention_js column; skipped"
        return result

    js_vals = df_mctr_a["attention_js"].dropna()
    result["mean_attn_js"]   = float(js_vals.mean()) if len(js_vals) > 0 else 0.0
    result["std_attn_js"]    = float(js_vals.std())  if len(js_vals) > 0 else 0.0
    result["n"]              = len(js_vals)
    result["conclusion"] = (
        "Supported (mean JS > 0.001)"    if result["mean_attn_js"] > 0.001
        else "Null result (JS ≈ 0)"
    )
    return result


def test_h5_ta_differs_from_t(
    df_mctr_t: pd.DataFrame,
    df_mctr_ta: pd.DataFrame,
) -> Dict[str, Any]:
    """H5: MCTR-TA produces different inference behavior from MCTR-T."""
    result = {"hypothesis": "H5: MCTR-TA ≠ MCTR-T"}

    for col in ["entropy", "attention_js", "accuracy"]:
        if col in df_mctr_t.columns and col in df_mctr_ta.columns:
            t_vals  = df_mctr_t[col].dropna().values
            ta_vals = df_mctr_ta[col].dropna().values
            if len(t_vals) > 1 and len(ta_vals) > 1:
                stat, pval = scipy_stats.mannwhitneyu(t_vals, ta_vals, alternative="two-sided")
                result[f"mannwhitney_p_{col}"] = float(pval)
                result[f"delta_{col}"] = float(np.mean(ta_vals) - np.mean(t_vals))

    result["conclusion"] = "Requires experiment data"
    return result


def compute_attention_redistribution_table(
    df_combined: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a table comparing attention metrics across all conditions.
    """
    conditions = df_combined["condition"].unique() if "condition" in df_combined.columns else []
    rows = []
    for cond in conditions:
        sub = df_combined[df_combined["condition"] == cond]
        row = {"Condition": cond}
        for col in ["attention_entropy_native", "attention_entropy_mod", "attention_js", "attention_modulation"]:
            if col in sub.columns:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_std"]  = float(sub[col].std())
        rows.append(row)
    return pd.DataFrame(rows)
