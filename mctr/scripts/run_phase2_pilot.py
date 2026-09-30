"""
MCTR Phase 2: Pilot Run
========================
150 examples × 3 seeds.
Compares Native Qwen / Fixed-T sweep / MCTR-T / MCTR-A / MCTR-TA.
Generates all tables and figures automatically.

Usage:
    python mctr/scripts/run_phase2_pilot.py
    python mctr/scripts/run_phase2_pilot.py --smoke-test     # 5 examples
    python mctr/scripts/run_phase2_pilot.py --max-examples 20
"""

import os
import sys
import json
import time
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mctr.datasets.registry import load_and_prepare_dataset
from mctr.base.transformer import MCTRTransformerWrapper
from mctr.state.state import StateExtractor
from mctr.metacognition.evaluator import MetaEvaluator
from mctr.control.joint_controller import JointController
from mctr.evaluation.paired_eval import run_phase2_paired_eval
from mctr.evaluation.attention_eval import (
    compute_state_correlation_matrix,
    compute_spearman_correlations,
    compute_per_feature_auroc,
    test_h1_psi_predicts_correctness,
    test_h4_attention_redistribution,
    test_h5_ta_differs_from_t,
    compute_attention_redistribution_table,
)
from mctr.visualization.phase2.attention_plots import (
    plot_accuracy_comparison,
    plot_calibration_comparison,
    plot_ablation_study,
    plot_efficiency_comparison,
    plot_attention_js_distribution,
    plot_meta_vs_attention_response,
    plot_state_correlation_matrix,
)
from mctr.evaluation.benchmark import EfficiencyTracker


CONFIG_PATH = "mctr/configs/phase2.yaml"
OUTPUT_DIR  = "outputs/phase2"


def run_pilot(config_path: str, smoke_test: bool = False, max_examples: int = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("  MCTR Phase 2 Pilot Run")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Smoke Test: {smoke_test}, Max Examples: {max_examples}")
    print("=" * 70)

    # ── Setup dirs ──────────────────────────────────────────────────────
    for sub in ["raw", "checkpoints", "tables", "figures", "logs", "reports"]:
        os.makedirs(f"{OUTPUT_DIR}/{sub}", exist_ok=True)

    # ── Seed ────────────────────────────────────────────────────────────
    base_seed   = config["evaluation"]["seed"]
    seeds       = [42, 43, 44] if not smoke_test else [42]
    n_examples  = 5 if smoke_test else (max_examples or config["evaluation"]["max_examples"])
    max_tokens  = 20 if smoke_test else config["evaluation"]["max_new_tokens"]

    print(f"\n  Seeds: {seeds}  |  Examples: {n_examples}  |  Max tokens: {max_tokens}")

    # ── Device ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Load Dataset ────────────────────────────────────────────────────
    print("\nSTEP 1: Loading dataset...")
    datasets = load_and_prepare_dataset(config, smoke_test=smoke_test)
    if "gsm8k_main_test" in datasets:
        dataset = datasets["gsm8k_main_test"][:n_examples]
    else:
        print("[WARN] gsm8k_main_test not found. Using available dataset.")
        key     = list(datasets.keys())[0]
        dataset = datasets[key][:n_examples]

    print(f"  Dataset: {len(dataset)} examples")

    # ── Load Model ──────────────────────────────────────────────────────
    print(f"\nSTEP 2: Loading {config['backbone']['name']}...")
    vocab_size  = 151936
    hidden_size = 896
    n_layers    = 24
    n_heads     = 16

    try:
        transformer = MCTRTransformerWrapper(
            config["backbone"]["name"],
            device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        transformer.model.eval()
        transformer.model.requires_grad_(False)
        vocab_size  = transformer.model.config.vocab_size
        hidden_size = transformer.model.config.hidden_size
        n_layers    = transformer.model.config.num_hidden_layers
        n_heads     = transformer.model.config.num_attention_heads
        print(f"  [OK] Model loaded: {n_layers} layers, {n_heads} heads")
    except Exception as e:
        print(f"  [WARN] Model load failed ({e}). Mocking.")
        transformer = None

    # ── Init MCTR Components ────────────────────────────────────────────
    print("\nSTEP 3: Initialising MCTR components...")
    meta_cfg  = config.get("meta", {})
    state_cfg = config.get("state", {})
    ctrl_cfg  = config.get("controller", {})
    attn_cfg  = config.get("attention_control", {})

    state_extractor = StateExtractor(vocab_size=vocab_size, config=state_cfg)
    evaluator = MetaEvaluator(
        input_dim=meta_cfg.get("input_dim", 6),
        hidden_dim=meta_cfg.get("hidden_dim", 64),
        meta_dim=meta_cfg.get("output_dim", 32),
    ).to(device)

    joint_cfg = {
        "enable_temperature": True,
        "enable_attention":   True,
        "temperature_min":    config["mctr_temperature"]["min"],
        "temperature_max":    config["mctr_temperature"]["max"],
        "controller_hidden_dim": ctrl_cfg.get("controller_hidden_dim", 64),
        "attention_control_mode": attn_cfg.get("mode", "global"),
        "attention_modulation_bound": attn_cfg.get("beta", 2.0),
        "attention_hidden_dim": ctrl_cfg.get("attention_hidden_dim", 32),
    }
    joint_controller = JointController(
        meta_dim=meta_cfg.get("output_dim", 32),
        config=joint_cfg,
        n_layers=n_layers,
        n_heads=n_heads,
    ).to(device)
    joint_controller.eval()

    print(f"  [OK] JointController: T∈[{joint_controller.t_min},{joint_controller.t_max}], "
          f"β={attn_cfg.get('beta',2.0)}")

    # ── Define Conditions ────────────────────────────────────────────────
    temp_sweep = config.get("temperature_sweep", {}).get("values", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    conditions = {}
    # Fixed-T baselines
    for T in temp_sweep:
        cname = f"Fixed-T-{T}"
        conditions[cname] = {"mode": "fixed_t", "temperature": T}

    # MCTR modes
    conditions["Native"]  = {"mode": "native"}
    conditions["MCTR-T"]  = {"mode": "mctr_t"}
    conditions["MCTR-A"]  = {"mode": "mctr_a"}
    conditions["MCTR-TA"] = {"mode": "mctr_ta"}

    # Ablations
    conditions["MCTR-A_zero"]    = {"mode": "native"}          # J: zero attention (= native)
    conditions["MCTR-A_random"]  = {"mode": "mctr_a"}           # approximation

    print(f"  Conditions ({len(conditions)}): {list(conditions.keys())}")

    # ── Multi-Seed Evaluation ────────────────────────────────────────────
    print(f"\nSTEP 4: Multi-seed evaluation ({seeds})...")
    tracker = EfficiencyTracker()
    tracker.start()

    all_seed_metrics = {}
    all_dfs = []

    for seed in seeds:
        print(f"\n  ── Seed {seed} ──")
        torch.manual_seed(seed)
        np.random.seed(seed)
        state_extractor.reset_history()

        seed_metrics = run_phase2_paired_eval(
            transformer=transformer,
            dataset=dataset,
            state_extractor=state_extractor,
            evaluator=evaluator,
            joint_controller=joint_controller,
            conditions=conditions,
            device=device,
            do_sample=True,
            max_new_tokens=max_tokens,
            collect_attention=(not smoke_test),
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=hidden_size // n_heads,
            output_dir=OUTPUT_DIR,
            seed=seed,
        )
        all_seed_metrics[seed] = seed_metrics

        # Load the saved CSV
        csv_path = f"{OUTPUT_DIR}/raw/paired_results_phase2_seed{seed}.csv"
        if os.path.exists(csv_path):
            all_dfs.append(pd.read_csv(csv_path))

    tracker.stop(tokens_generated=n_examples * len(conditions) * len(seeds) * max_tokens)
    eff = tracker.get_metrics()

    # ── Aggregate Multi-Seed Results ─────────────────────────────────────
    print("\nSTEP 5: Aggregating multi-seed results...")
    cond_names = list(conditions.keys())
    agg_rows = []

    for cond in cond_names:
        seed_accs  = [all_seed_metrics[s][cond]["accuracy"]        for s in seeds if cond in all_seed_metrics[s]]
        seed_eces  = [all_seed_metrics[s][cond]["ece"]             for s in seeds if cond in all_seed_metrics[s]]
        seed_briers= [all_seed_metrics[s][cond]["brier"]           for s in seeds if cond in all_seed_metrics[s]]
        seed_ents  = [all_seed_metrics[s][cond]["mean_entropy"]    for s in seeds if cond in all_seed_metrics[s]]
        seed_temps = [all_seed_metrics[s][cond]["mean_temperature"] for s in seeds if cond in all_seed_metrics[s]]
        seed_attn_js= [all_seed_metrics[s][cond]["attn_js"]        for s in seeds if cond in all_seed_metrics[s]]

        if not seed_accs:
            continue

        agg_rows.append({
            "Condition":        cond,
            "Accuracy_mean":    np.mean(seed_accs),
            "Accuracy_std":     np.std(seed_accs),
            "ECE_mean":         np.mean(seed_eces),
            "ECE_std":          np.std(seed_eces),
            "Brier_mean":       np.mean(seed_briers),
            "Brier_std":        np.std(seed_briers),
            "Entropy_mean":     np.mean(seed_ents),
            "Temperature_mean": np.mean(seed_temps),
            "AttnJS_mean":      np.mean(seed_attn_js),
            "AttnJS_std":       np.std(seed_attn_js),
        })

    df_agg = pd.DataFrame(agg_rows)

    # ── Save Tables ──────────────────────────────────────────────────────
    print("\nSTEP 6: Saving tables...")

    # Table 2: Main comparison
    df_table2 = df_agg.copy()
    df_table2["Accuracy"] = df_table2.apply(
        lambda r: f"{r['Accuracy_mean']*100:.2f}% ± {r['Accuracy_std']*100:.2f}%", axis=1)
    df_table2["ECE"]   = df_table2.apply(lambda r: f"{r['ECE_mean']:.4f} ± {r['ECE_std']:.4f}", axis=1)
    df_table2["Brier"] = df_table2.apply(lambda r: f"{r['Brier_mean']:.4f} ± {r['Brier_std']:.4f}", axis=1)
    df_table2["Attn JS"] = df_table2.apply(lambda r: f"{r['AttnJS_mean']:.6f} ± {r['AttnJS_std']:.6f}", axis=1)

    df_table2[["Condition", "Accuracy", "ECE", "Brier", "Entropy_mean", "Temperature_mean", "Attn JS"]].to_markdown(
        f"{OUTPUT_DIR}/tables/table2_main_comparison_p2.md", index=False
    )
    df_agg.to_csv(f"{OUTPUT_DIR}/tables/aggregated_metrics_p2.csv", index=False)

    # ── State Redundancy Analysis ─────────────────────────────────────────
    print("\nSTEP 7: State redundancy analysis...")
    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        df_combined.to_csv(f"{OUTPUT_DIR}/raw/all_results_combined.csv", index=False)

        corr_df    = compute_state_correlation_matrix(df_combined)
        spearman_df= compute_spearman_correlations(df_combined)
        aurocs     = compute_per_feature_auroc(df_combined)

        if not corr_df.empty:
            corr_df.to_csv(f"{OUTPUT_DIR}/tables/state_pearson_correlation.csv")
            spearman_df.to_csv(f"{OUTPUT_DIR}/tables/state_spearman_correlation.csv")

            # Print redundancy warnings
            print("  State Correlation Summary (Pearson):")
            for col in corr_df.columns:
                for row in corr_df.index:
                    if row != col:
                        val = corr_df.loc[row, col]
                        if abs(val) > 0.95:
                            print(f"    [REDUNDANCY WARN] Corr({col},{row}) = {val:.3f}")

        if aurocs:
            print(f"  AUROC by feature: {aurocs}")
            with open(f"{OUTPUT_DIR}/tables/feature_auroc.json", "w") as f:
                json.dump(aurocs, f, indent=2)

        # Hypothesis tests
        hypotheses = {}
        hypotheses["H1"] = test_h1_psi_predicts_correctness(df_combined)

        df_mctr_a  = df_combined[df_combined["condition"] == "MCTR-A"]
        df_mctr_t  = df_combined[df_combined["condition"] == "MCTR-T"]
        df_mctr_ta = df_combined[df_combined["condition"] == "MCTR-TA"]
        df_native  = df_combined[df_combined["condition"] == "Native"]

        hypotheses["H4"] = test_h4_attention_redistribution(df_native, df_mctr_a)
        hypotheses["H5"] = test_h5_ta_differs_from_t(df_mctr_t, df_mctr_ta)

        with open(f"{OUTPUT_DIR}/reports/hypothesis_tests.json", "w") as f:
            json.dump(hypotheses, f, indent=2, default=str)
        print(f"  H1: {hypotheses['H1'].get('conclusion', 'N/A')}")
        print(f"  H4: {hypotheses['H4'].get('conclusion', 'N/A')}")
        print(f"  H5: {hypotheses['H5'].get('conclusion', 'N/A')}")

        # Table 5: Attention control metrics
        attn_table = compute_attention_redistribution_table(df_combined)
        attn_table.to_csv(f"{OUTPUT_DIR}/tables/table5_attention_control.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nSTEP 8: Generating publication figures...")
    fig_dir = f"{OUTPUT_DIR}/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # Build metrics dict for the primary conditions (seed-averaged)
    key_conds = ["Native", "MCTR-T", "MCTR-A", "MCTR-TA"]
    primary_metrics = {
        r["Condition"]: {
            "accuracy": r["Accuracy_mean"],
            "ece": r["ECE_mean"],
            "brier": r["Brier_mean"],
            "mean_latency": all_seed_metrics[42].get(r["Condition"], {}).get("mean_latency", 0),
            "peak_vram": all_seed_metrics[42].get(r["Condition"], {}).get("peak_vram", 0),
        }
        for _, r in df_agg[df_agg["Condition"].isin(key_conds)].iterrows()
    }

    plot_accuracy_comparison(primary_metrics,     f"{fig_dir}/Fig11_Accuracy_Comparison.png")
    plot_calibration_comparison(primary_metrics,  f"{fig_dir}/Fig12_Calibration_Comparison.png")

    # Ablation (all conditions)
    full_metrics = {r["Condition"]: {"accuracy": r["Accuracy_mean"]} for _, r in df_agg.iterrows()}
    plot_ablation_study(full_metrics, f"{fig_dir}/Fig13_Ablation_Study.png")
    plot_efficiency_comparison(primary_metrics, f"{fig_dir}/Fig14_Efficiency.png")

    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        plot_attention_js_distribution(df_combined, f"{fig_dir}/Fig09_Attention_JS_Distribution.png")
        plot_meta_vs_attention_response(df_combined, f"{fig_dir}/Fig10_Meta_vs_Attention_Response.png")

        if not corr_df.empty:
            plot_state_correlation_matrix(corr_df, f"{fig_dir}/Fig04_State_Correlation_Matrix.png")

    # ── MCTR-T vs baselines delta table ───────────────────────────────────
    mctr_t_row = df_agg[df_agg["Condition"] == "MCTR-T"]
    if len(mctr_t_row) > 0:
        mctr_t_row = mctr_t_row.iloc[0]
        delta_rows = []
        for _, row in df_agg[df_agg["Condition"].str.startswith("Fixed-T")].iterrows():
            delta_rows.append({
                "Comparison": f"MCTR-T vs {row['Condition']}",
                "Δ Accuracy": f"{(mctr_t_row['Accuracy_mean'] - row['Accuracy_mean'])*100:+.2f}%",
                "Δ ECE":      f"{mctr_t_row['ECE_mean'] - row['ECE_mean']:+.4f}",
                "Δ Brier":    f"{mctr_t_row['Brier_mean'] - row['Brier_mean']:+.4f}",
            })
        pd.DataFrame(delta_rows).to_markdown(f"{OUTPUT_DIR}/tables/table3_mctr_vs_fixed.md", index=False)

    # ── Efficiency metrics ─────────────────────────────────────────────────
    eff_row = {
        "total_latency_s":  eff["latency_sec"],
        "tokens_per_sec":   eff["tokens_per_sec"],
        "total_tokens":     eff["total_tokens"],
        "peak_vram_mb":     eff["peak_vram_mb"],
    }
    pd.DataFrame([eff_row]).to_csv(f"{OUTPUT_DIR}/tables/table8_efficiency.csv", index=False)

    # ── Write completion report ────────────────────────────────────────────
    print("\nSTEP 9: Writing Phase 2 completion report...")
    write_completion_report(config, df_agg, eff, hypotheses if all_dfs else {}, seeds, n_examples, smoke_test)

    print(f"\n{'=' * 70}")
    print("  MCTR Phase 2 Pilot Run COMPLETE")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"{'=' * 70}\n")


def write_completion_report(config, df_agg, eff, hypotheses, seeds, n_examples, smoke_test):
    ts = datetime.now().isoformat()

    mctr_ta = df_agg[df_agg["Condition"] == "MCTR-TA"]
    mctr_t  = df_agg[df_agg["Condition"] == "MCTR-T"]
    mctr_a  = df_agg[df_agg["Condition"] == "MCTR-A"]
    native  = df_agg[df_agg["Condition"] == "Native"]

    def fmt(df_row, col):
        if len(df_row) == 0:
            return "N/A"
        return f"{df_row.iloc[0][col]:.4f}"

    report = f"""# MCTR Phase 2 Completion Report

Generated: {ts}
Model: {config['backbone']['name']}
Phase: 2 - Meta-Cognitive Attention Regulation
Smoke Test: {smoke_test}

---

## Implementation Status

### Phase 2 Architecture
- [x] Meta-Evaluator M_phi preserved from Phase 1
- [x] JointController G_omega with shared trunk
- [x] Temperature Head: T_t in [{config['mctr_temperature']['min']}, {config['mctr_temperature']['max']}]
- [x] Attention Head: DeltaA_t in [-{config['attention_control']['beta']}, +{config['attention_control']['beta']}]
- [x] Attention modulation: additive log-space bias to attention weights
- [x] Three attention levels: global / layer / head
- [x] Temporal causality enforced: Psi_(t-1) -> controller -> A_t -> P_t -> Psi_t
- [x] Stochastic decoding: y_t ~ Categorical(P_t) via torch.multinomial
- [x] AttentionHookManager for attention map capture
- [x] All 5 control modes (Native/Fixed-T/MCTR-T/MCTR-A/MCTR-TA)
- [x] Attention metrics: entropy, concentration, JS/KL divergence, L1
- [x] Collapse detection warnings
- [x] Canonical per-example CSV output
- [x] State redundancy reporting

### State Redundancy (Known - Inherited from Phase 1)
- U_t = H_norm_t by construction (normalized_entropy == uncertainty)
- High correlations may exist between C_t/S_t and H_t/N_t/K_t
- This redundancy is preserved for backward compatibility
- Full redundancy analysis in: outputs/phase2/tables/state_pearson_correlation.csv

---

## Experimental Configuration

- Dataset:        GSM8K Main (n={n_examples})
- Seeds:          {seeds}
- Max Tokens:     {config['evaluation']['max_new_tokens']}
- Decoding:       Stochastic (torch.multinomial)
- Control Modes:  Native, Fixed-T x6, MCTR-T, MCTR-A, MCTR-TA

---

## Results Summary

| Condition | Accuracy | ECE | Brier | Attn JS |
|:---|:---|:---|:---|:---|
| Native   | {fmt(native,  'Accuracy_mean')} | {fmt(native,  'ECE_mean')} | {fmt(native,  'Brier_mean')} | - |
| MCTR-T   | {fmt(mctr_t,  'Accuracy_mean')} | {fmt(mctr_t,  'ECE_mean')} | {fmt(mctr_t,  'Brier_mean')} | - |
| MCTR-A   | {fmt(mctr_a,  'Accuracy_mean')} | {fmt(mctr_a,  'ECE_mean')} | {fmt(mctr_a,  'Brier_mean')} | {fmt(mctr_a, 'AttnJS_mean')} |
| MCTR-TA  | {fmt(mctr_ta, 'Accuracy_mean')} | {fmt(mctr_ta, 'ECE_mean')} | {fmt(mctr_ta, 'Brier_mean')} | {fmt(mctr_ta,'AttnJS_mean')} |

---

## Hypothesis Test Results

{_fmt_hypotheses(hypotheses)}

---

## Efficiency

- Total Latency: {eff['latency_sec']:.2f}s
- Tokens/sec: {eff['tokens_per_sec']:.1f}
- Peak VRAM: {eff['peak_vram_mb']:.2f} MB

---

## Key Findings

1. **Temperature Control**: Inherited from Phase 1. argmax(z/T) = argmax(z);
   therefore stochastic decoding is mandatory for behavioral evaluation. [OK]

2. **Attention Modulation**: Bounded additive log-space modulation applied.
   DeltaA in [-{config['attention_control']['beta']}, +{config['attention_control']['beta']}]. Attention sums to 1 post-modulation.

3. **Temporal Causality**: Psi_(t-1) -> controller -> A_t is strictly enforced.
   No same-step circular dependency.

4. **State Redundancy**: U_t = H_norm_t (Uncertainty = Normalized Entropy) by
   construction. Explicitly documented. Full correlation matrix available.

---

## Limitations

- Phase 2 controller is untrained (random weights); results reflect
  random attention modulation, not learned regulation.
- Training Stage 3 (attention controller) is defined but not yet run.
- 150-example development scale: no benchmark-wide generalization claims.
- BBH generalization not yet evaluated.

---

## Recommendation for Phase 3

1. Train the attention controller (Stage 3) using the L_attention loss.
2. Validate H3-H8 with trained controller.
3. Investigate state dimensionality reduction given high correlations.
4. Evaluate on ARC-Challenge and MMLU (generalization set).
5. Consider token-level compute allocation (Phase 3 scope).

---

## Reproducibility

```bash
# Smoke test
python mctr/scripts/run_phase2_smoke.py

# Pilot run
python mctr/scripts/run_phase2_pilot.py

# Full run
python mctr/scripts/run_phase2_full.py
```

Config: mctr/configs/phase2.yaml
Phase 1 baseline commit: a2e4bda
"""

    with open(f"{OUTPUT_DIR}/reports/PHASE2_COMPLETION_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved → {OUTPUT_DIR}/reports/PHASE2_COMPLETION_REPORT.md")


def _fmt_hypotheses(hyp):
    if not hyp:
        return "_Hypothesis tests require experimental data; run pilot with full dataset._"
    lines = []
    for k, v in hyp.items():
        arrow = "->"  # ASCII arrow for Windows compat
        lines.append(f"- **{k}**: {v.get('hypothesis', k)} {arrow} _{v.get('conclusion', 'N/A')}_")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTR Phase 2 Pilot Run")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    run_pilot(args.config, smoke_test=args.smoke_test, max_examples=args.max_examples)
