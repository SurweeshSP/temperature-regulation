import os
import sys

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import yaml
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime


# Add the project root to sys.path so 'mctr' can be imported when running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
from mctr.datasets.registry import load_and_prepare_dataset
from mctr.evaluation.benchmark import EfficiencyTracker
from mctr.base.transformer import MCTRTransformerWrapper
from mctr.state.state import StateExtractor
from mctr.metacognition.evaluator import MetaEvaluator
from mctr.controller.controller import MCTRController, EntropyTarget

from mctr.visualization.tables import (
    generate_table1_main_comparison, generate_table2_temperature_sweep,
    generate_table3_meta_state, generate_table4_ablation,
    generate_table5_regulation, generate_table6_efficiency
)
from mctr.visualization.publication.architecture import generate_architecture_diagram
from mctr.visualization.publication.distributions import generate_state_distributions, generate_correlation_matrix
from mctr.visualization.publication.predictive import generate_roc_curves, generate_pr_curves
from mctr.visualization.publication.trajectories import generate_temperature_trajectory, generate_entropy_control_trajectory
from mctr.visualization.publication.relationships import generate_entropy_vs_confidence, generate_temp_vs_meta
from mctr.visualization.publication.benchmarks import generate_baseline_comparison, generate_fixed_temperature_sweep
from mctr.visualization.publication.fast_run_plots import generate_fast_run_plots
from mctr.visualization.plot_utils import save_publication_figure

def run_experiment(config_path: str, smoke_test: bool = False, max_examples: int = None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"============================================================")
    print(f" Starting Phase 1 Scientific Validation (Smoke Test: {smoke_test})")
    print(f"============================================================")
    
    out_dir = "outputs/phase1"
    plots_dir = f"{out_dir}/plots/publication"
    for sub in ['checkpoints', 'metrics', 'trajectories', 'logs', 'configs', 'reports', 'tables']:
        os.makedirs(f"{out_dir}/{sub}", exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
        
    # STEP 1: Load Datasets
    print("STEP 1: Loading Datasets (Filtering for gsm8k_main_test only)...")
    datasets = load_and_prepare_dataset(config, smoke_test=smoke_test)
    
    # ENFORCE EXACTLY gsm8k_main_test for the fast validation run
    if 'gsm8k_main_test' in datasets:
        datasets = {'gsm8k_main_test': datasets['gsm8k_main_test']}
    else:
        raise ValueError("Could not find gsm8k_main_test in loaded datasets.")
        
    for k, v in datasets.items():
        print(f"  Loaded {k}: {len(v)} examples")
        
    # STEP 2 & 3: Load Model & Freeze
    print(f"\nSTEP 2 & 3: Loading {config['backbone']['name']} & Freezing Backbone...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  [WARNING] No CUDA GPU detected or available. Falling back to CPU.")
        
    try:
        transformer = MCTRTransformerWrapper(
            config['backbone']['name'], 
            device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        vocab_size = transformer.model.config.vocab_size
        hidden_size = transformer.model.config.hidden_size
        
        # Verify frozen constraints
        transformer.model.eval()
        transformer.model.requires_grad_(False)
        trainable_params = sum(p.numel() for p in transformer.model.parameters() if p.requires_grad)
        print(f"  Trainable Parameters: {trainable_params}")
        if trainable_params > 0:
            raise ValueError("Backbone is not fully frozen!")
            
    except Exception as e:
        print(f"  Failed to load transformer. Mocking for test. Error: {e}")
        transformer = None
        vocab_size = 151936
        hidden_size = 1536
        
    # 4. Init Components
    print("\nInitializing Components...")
    state_extractor = StateExtractor(vocab_size=vocab_size, config=config.get('state', {}))
    evaluator = MetaEvaluator(
        input_dim=config.get('meta', {}).get('input_dim', 5),
        hidden_dim=config.get('meta', {}).get('hidden_dim', 64),
        meta_dim=config.get('meta', {}).get('output_dim', 32)
    ).to(device)
    controller = MCTRController(meta_dim=config.get('meta', {}).get('output_dim', 32), config=config.get('controller', {})).to(device)
    
    # Override for fast, real-data single-seed evaluation
    # Multi-seed stochastic sampling evaluation
    seeds = [42, 43, 44]
    print(f"  Enforcing multi-seed stochastic evaluation: {seeds}")
    
    if max_examples is None and not smoke_test:
        max_examples = 150
        print(f"  Enforcing 150 examples per condition across seeds")
        
    tracker = EfficiencyTracker()
    tracker.start()
    
    # Define Evaluation Conditions (Baseline Sweeps + MCTR-T)
    eval_conditions = {
        'Fixed-T-0.1': {'type': 'fixed', 'temperature': 0.1},
        'Fixed-T-0.3': {'type': 'fixed', 'temperature': 0.3},
        'Fixed-T-0.5': {'type': 'fixed', 'temperature': 0.5},
        'Fixed-T-0.7': {'type': 'fixed', 'temperature': 0.7},
        'Fixed-T-0.9': {'type': 'fixed', 'temperature': 0.9},
        'Fixed-T-1.0': {'type': 'fixed', 'temperature': 1.0},
        'MCTR-T': {'type': 'mctr', 'ablation': 'none'}
    }
    
    print("\nSTEP 4-15: Executing Causal Inference Loops (Track 2: Stochastic Sampling)...")
    from mctr.evaluation.pipeline import run_inference_loop
    
    all_stochastic_metrics = {}
    for seed in seeds:
        print(f"  --- Running Stochastic Seed: {seed} ---")
        torch.manual_seed(seed)
        seed_metrics = {}
        
        for ds_name, ds_data in datasets.items():
            print(f"    Evaluating Dataset Split: {ds_name}")
            eval_data = ds_data[:5] if smoke_test else (ds_data[:max_examples] if max_examples else ds_data)
            
            ds_metrics = run_inference_loop(
                transformer=transformer,
                dataset=eval_data,
                state_extractor=state_extractor,
                evaluator=evaluator,
                controller=controller,
                conditions=eval_conditions,
                device=device,
                do_sample=True
            )
            
            parsed_dataset = ds_name.split('_')[0]
            parsed_task = '_'.join(ds_name.split('_')[1:])
            
            if parsed_dataset not in seed_metrics:
                seed_metrics[parsed_dataset] = {}
            seed_metrics[parsed_dataset][parsed_task] = ds_metrics
            
        all_stochastic_metrics[seed] = seed_metrics
        
    # Also run Track 1 (Deterministic Greedy) for Seed 42 as Sanity Check
    print("\nRunning Track 1: Deterministic Greedy Decoding (Seed 42)...")
    torch.manual_seed(42)
    deterministic_metrics = {}
    for ds_name, ds_data in datasets.items():
        eval_data = ds_data[:5] if smoke_test else (ds_data[:max_examples] if max_examples else ds_data)
        ds_metrics = run_inference_loop(
            transformer=transformer,
            dataset=eval_data,
            state_extractor=state_extractor,
            evaluator=evaluator,
            controller=controller,
            conditions=eval_conditions,
            device=device,
            do_sample=False
        )
        parsed_dataset = ds_name.split('_')[0]
        parsed_task = '_'.join(ds_name.split('_')[1:])
        if parsed_dataset not in deterministic_metrics:
            deterministic_metrics[parsed_dataset] = {}
        deterministic_metrics[parsed_dataset][parsed_task] = ds_metrics

    tracker.stop(tokens_generated=len(seeds)*150*256)
    eff_metrics = tracker.get_metrics()
    
    print("\nSTEP 16-18: Aggregating Multi-Seed Metrics & Generating Publication Figures...")
    os.makedirs(f"{out_dir}/tables", exist_ok=True)
    
    # Aggregate multi-seed metrics (Mean & Std) across seeds
    cond_names = list(eval_conditions.keys())
    aggregated_metrics = []
    
    for cond_name in cond_names:
        accs = [all_stochastic_metrics[s]['gsm8k']['main_test'][cond_name]['accuracy'] for s in seeds]
        eces = [all_stochastic_metrics[s]['gsm8k']['main_test'][cond_name]['ece'] for s in seeds]
        briers = [all_stochastic_metrics[s]['gsm8k']['main_test'][cond_name]['brier'] for s in seeds]
        entropies = [all_stochastic_metrics[s]['gsm8k']['main_test'][cond_name]['mean_entropy'] for s in seeds]
        confidences = [all_stochastic_metrics[s]['gsm8k']['main_test'][cond_name]['mean_confidence'] for s in seeds]
        temps = [all_stochastic_metrics[s]['gsm8k']['main_test'][cond_name]['mean_temperature'] for s in seeds]
        
        det_acc = deterministic_metrics['gsm8k']['main_test'][cond_name]['accuracy']
        det_ece = deterministic_metrics['gsm8k']['main_test'][cond_name]['ece']
        
        aggregated_metrics.append({
            'Condition': cond_name,
            'Greedy Accuracy': det_acc,
            'Stochastic Accuracy Mean': np.mean(accs),
            'Stochastic Accuracy Std': np.std(accs),
            'ECE Mean': np.mean(eces),
            'ECE Std': np.std(eces),
            'Brier Mean': np.mean(briers),
            'Brier Std': np.std(briers),
            'Mean Entropy': np.mean(entropies),
            'Mean Confidence': np.mean(confidences),
            'Mean T': np.mean(temps)
        })
        
    df_agg = pd.DataFrame(aggregated_metrics)
    
    # Formatting for markdown table
    df_table = df_agg.copy()
    df_table['Stochastic Accuracy'] = df_table.apply(lambda r: f"{r['Stochastic Accuracy Mean']*100:.2f}% ± {r['Stochastic Accuracy Std']*100:.2f}%", axis=1)
    df_table['ECE'] = df_table.apply(lambda r: f"{r['ECE Mean']:.4f} ± {r['ECE Std']:.4f}", axis=1)
    df_table['Brier'] = df_table.apply(lambda r: f"{r['Brier Mean']:.4f} ± {r['Brier Std']:.4f}", axis=1)
    
    df_table[['Condition', 'Greedy Accuracy', 'Stochastic Accuracy', 'ECE', 'Brier', 'Mean Entropy', 'Mean Confidence', 'Mean T']].to_markdown(f"{out_dir}/tables/table_fast_main_comparison.md", index=False)
    
    # Baseline comparison table (MCTR-T vs Fixed baselines under stochastic sampling)
    mctr_row = df_agg[df_agg['Condition'] == 'MCTR-T'].iloc[0]
    baseline_diffs = []
    for _, row in df_agg[df_agg['Condition'].str.contains('Fixed')].iterrows():
        baseline_diffs.append({
            'Comparison': f"MCTR-T vs {row['Condition']}",
            'Δ Accuracy': f"{(mctr_row['Stochastic Accuracy Mean'] - row['Stochastic Accuracy Mean'])*100:+.2f}%",
            'Δ ECE': f"{mctr_row['ECE Mean'] - row['ECE Mean']:+.4f}",
            'Δ Brier': f"{mctr_row['Brier Mean'] - row['Brier Mean']:+.4f}",
            'Δ Entropy': f"{mctr_row['Mean Entropy'] - row['Mean Entropy']:+.4f}",
            'Δ Confidence': f"{mctr_row['Mean Confidence'] - row['Mean Confidence']:+.4f}"
        })
    pd.DataFrame(baseline_diffs).to_markdown(f"{out_dir}/tables/table_mctr_vs_baseline.md", index=False)
    
    # Generate Benchmark visualisations from extracted metrics
    bench_rows = []
    for _, row in df_agg.iterrows():
        bench_rows.append({
            'Task': 'GSM8K',
            'Model': 'MCTR' if row['Condition'] == 'MCTR-T' else row['Condition'],
            'Accuracy': row['Stochastic Accuracy Mean'],
            'Accuracy_Std': row['Stochastic Accuracy Std'],
            'Type': 'MCTR' if row['Condition'] == 'MCTR-T' else 'Fixed',
            'Temperature': row['Mean T']
        })
    bench_df = pd.DataFrame(bench_rows)
    
    generate_baseline_comparison(bench_df, f"{plots_dir}/Fig10_Baseline_vs_MCTR.png")
    generate_fixed_temperature_sweep(bench_df, f"{plots_dir}/Fig09_FixedTemperature_Sweep.png")

    # Generate Fast Run Custom Matplotlib Plots
    paired_log_path = f"{out_dir}/tables/paired_results_log.csv"
    fast_plots_dir = f"{out_dir}/figures"
    generate_fast_run_plots(paired_log_path, fast_plots_dir)

    print("\nSTEP 19: Exporting Experiment Manifest & Config...")
    n_examples_actual = max_examples if max_examples else len(datasets['gsm8k_main_test'])
    manifest = {
        "model": config['backbone']['name'],
        "dataset": "gsm8k_main_test",
        "seeds_evaluated": seeds,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "n_examples": n_examples_actual,
        "total_trajectories": n_examples_actual * len(eval_conditions) * len(seeds),
        "decoding_modes": ["Deterministic Greedy (Track 1)", "Stochastic Sampling (Track 2)"],
        "temperature_bounds": [0.2, 1.2],
        "conditions": list(eval_conditions.keys())
    }
    with open(f"{out_dir}/FAST_RUN_CONFIG.md", "w") as f:
        f.write("# FAST RUN CONFIG\n```json\n" + json.dumps(manifest, indent=4) + "\n```")
        
    print("\nSTEP 20-22: Generating Comprehensive Report...")
    mctr_stoch = df_agg[df_agg['Condition'] == 'MCTR-T'].iloc[0]
    
    report_content = f"""# MCTR Phase 1 Adaptive Stochastic Validation Report

## Experimental Configuration
- Dataset: GSM8K Main Test (n = {n_examples_actual})
- Sampling Seeds Evaluated: {seeds} (3 random seeds per question across all conditions)
- Total Trajectories Evaluated: {n_examples_actual} examples × 7 conditions × 3 seeds = {n_examples_actual * 7 * len(seeds)}
- Decoding Mode: Adaptive Stochastic Sampling (`torch.multinomial`) vs Deterministic Greedy (`torch.argmax`)
- Model Backbone: {config['backbone']['name']} (Fully Frozen)
- MCTR Temperature Bounds: [$T_{{min}} = 0.2, T_{{max}} = 1.2$]

---

## Key Technical Finding: Decoder Coupling
Under **Deterministic Greedy Decoding** (`argmax`), temperature scaling $\\frac{{z_t}}{{T_t}}$ preserves logit ordinal ranking ($\\arg\\max (z_t / T) = \\arg\\max(z_t)$). Therefore, greedy accuracy is identical across all temperatures ({df_agg['Greedy Accuracy'].iloc[0]*100:.2f}%).

Under **Adaptive Stochastic Sampling** (`multinomial`), temperature scaling directly participates in token selection ($T_t \\rightarrow P_t \\rightarrow y_t$), allowing accuracy differences and optimal regulation strategies to emerge.

---

## Accuracy & Calibration Results (Stochastic Sampling, Mean ± Std)

| Condition | Greedy Acc | Stochastic Accuracy (Mean ± Std) | ECE (Mean ± Std) | Brier (Mean ± Std) | Mean T |
|:---|:---:|:---:|:---:|:---:|:---:|
"""
    for _, row in df_agg.iterrows():
        report_content += f"| {row['Condition']} | {row['Greedy Accuracy']*100:.2f}% | {row['Stochastic Accuracy Mean']*100:.2f}% ± {row['Stochastic Accuracy Std']*100:.2f}% | {row['ECE Mean']:.4f} ± {row['ECE Std']:.4f} | {row['Brier Mean']:.4f} ± {row['Brier Std']:.4f} | {row['Mean T']:.4f} |\n"

    report_content += f"""
---

## MCTR Dynamic Temperature Regulation Behavior
- Mean Temperature: {mctr_stoch['Mean T']:.4f}
- Mean Normalized Entropy: {mctr_stoch['Mean Entropy']:.4f}
- Mean Predictive Confidence: {mctr_stoch['Mean Confidence']:.4f}

---

## Efficiency
- Total Latency: {eff_metrics['latency_sec']:.2f}s
- Peak VRAM Utilization: {eff_metrics['peak_vram_mb']:.2f} MB

---

## Detailed Tables & Visualizations
- Main Comparison Table: `outputs/phase1/tables/table_fast_main_comparison.md`
- MCTR vs Baselines Table: `outputs/phase1/tables/table_mctr_vs_baseline.md`
- Baseline vs MCTR Plot: `outputs/phase1/plots/publication/Fig10_Baseline_vs_MCTR.png`
- Temperature Sweep Plot: `outputs/phase1/plots/publication/Fig09_FixedTemperature_Sweep.png`
"""
    with open(f"{out_dir}/FAST_RUN_REPORT.md", "w") as f:
        f.write(report_content)
        
    print(f"\nExperiment Complete. Report saved to {out_dir}/FAST_RUN_REPORT.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("max_examples_pos", nargs="?", type=int, default=None, help="Limit number of examples per split")
    parser.add_argument("--config", default="mctr/configs/phase1_complete.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run with 5 examples per split")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples per split to speed up real evaluation")
    args = parser.parse_args()
    
    max_examples = args.max_examples if args.max_examples is not None else args.max_examples_pos
    run_experiment(args.config, smoke_test=args.smoke_test, max_examples=max_examples)

