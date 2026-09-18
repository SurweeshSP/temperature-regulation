import os
import pandas as pd

def export_table(df: pd.DataFrame, base_path: str, table_name: str):
    """
    Exports a DataFrame into CSV, Markdown, and LaTeX formats.
    """
    os.makedirs(base_path, exist_ok=True)
    
    csv_path = os.path.join(base_path, f"{table_name}.csv")
    md_path = os.path.join(base_path, f"{table_name}.md")
    tex_path = os.path.join(base_path, f"{table_name}.tex")
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    
    # Save Markdown
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(df.to_markdown(index=False))
        
    # Save LaTeX
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))

def generate_table1_main_comparison(metrics_dict, output_dir: str):
    """
    Table 1: WITHOUT vs WITH TEMPERATURE REGULATION
    """
    rows = []
    for dataset, tasks in metrics_dict.items():
        for task, conditions in tasks.items():
            for condition, metrics in conditions.items():
                if condition in ['Fixed-T', 'MCTR-T']:
                    rows.append({
                        'Dataset': dataset,
                        'Task': task,
                        'Condition': condition,
                        'Accuracy': metrics.get('accuracy', 0.0),
                        'NLL': metrics.get('nll', 0.0),
                        'ECE': metrics.get('ece', 0.0),
                        'Brier': metrics.get('brier', 0.0),
                        'Mean Entropy': metrics.get('mean_entropy', 0.0),
                        'Mean Confidence': metrics.get('mean_confidence', 0.0),
                        'Mean Temperature': metrics.get('mean_temperature', 0.0),
                        'Temperature Std': metrics.get('temperature_std', 0.0),
                        'Entropy Target MAE': metrics.get('entropy_target_mae', 0.0),
                        'Tokens': metrics.get('tokens', 0),
                        'Latency': metrics.get('latency', 0.0),
                        'Peak VRAM': metrics.get('peak_vram', 0.0)
                    })
    
    if rows:
        df = pd.DataFrame(rows)
        export_table(df, output_dir, "table1_main_comparison")

def generate_table2_temperature_sweep(metrics_dict, output_dir: str):
    """
    Table 2: FIXED TEMPERATURE SWEEP
    """
    rows = []
    for dataset, tasks in metrics_dict.items():
        for task, conditions in tasks.items():
            for condition, metrics in conditions.items():
                if condition.startswith('Fixed-T-') or condition == 'MCTR-T':
                    temp_val = condition.replace('Fixed-T-', '') if condition != 'MCTR-T' else 'MCTR'
                    rows.append({
                        'Dataset': f"{dataset}-{task}",
                        'Temperature': temp_val,
                        'Accuracy': metrics.get('accuracy', 0.0),
                        'NLL': metrics.get('nll', 0.0),
                        'ECE': metrics.get('ece', 0.0),
                        'Brier': metrics.get('brier', 0.0),
                        'Mean Entropy': metrics.get('mean_entropy', 0.0),
                        'Mean Confidence': metrics.get('mean_confidence', 0.0),
                        'Mean Tokens': metrics.get('mean_tokens', 0.0),
                        'Latency': metrics.get('latency', 0.0)
                    })
    if rows:
        df = pd.DataFrame(rows)
        export_table(df, output_dir, "table2_temperature_sweep")

def generate_table3_meta_state(predictive_metrics, output_dir: str):
    """
    Table 3: META-STATE PREDICTIVE POWER
    """
    rows = []
    for feature, metrics in predictive_metrics.items():
        rows.append({
            'Feature': feature,
            'AUROC': metrics.get('auroc', 0.0),
            'AUPRC': metrics.get('auprc', 0.0),
            'ECE': metrics.get('ece', 0.0),
            'Brier': metrics.get('brier', 0.0),
            'Pearson r': metrics.get('pearson', 0.0),
            'Spearman ρ': metrics.get('spearman', 0.0)
        })
    if rows:
        df = pd.DataFrame(rows)
        export_table(df, output_dir, "table3_meta_state")

def generate_table4_ablation(ablation_metrics, output_dir: str):
    """
    Table 4: MCTR COMPONENT ABLATION
    """
    rows = []
    for condition, metrics in ablation_metrics.items():
        rows.append({
            'Condition': condition,
            'Accuracy': metrics.get('accuracy', 0.0),
            'NLL': metrics.get('nll', 0.0),
            'ECE': metrics.get('ece', 0.0),
            'Brier': metrics.get('brier', 0.0),
            'Mean Entropy': metrics.get('mean_entropy', 0.0),
            'Entropy Target MAE': metrics.get('entropy_target_mae', 0.0),
            'Temperature Std': metrics.get('temperature_std', 0.0),
            'Latency': metrics.get('latency', 0.0)
        })
    if rows:
        df = pd.DataFrame(rows)
        export_table(df, output_dir, "table4_ablation")

def generate_table5_regulation(regulation_metrics, output_dir: str):
    """
    Table 5: TEMPERATURE REGULATION QUALITY
    """
    rows = []
    for condition, metrics in regulation_metrics.items():
        rows.append({
            'Condition': condition,
            'Mean T': metrics.get('mean_t', 0.0),
            'Std T': metrics.get('std_t', 0.0),
            'Min T': metrics.get('min_t', 0.0),
            'Max T': metrics.get('max_t', 0.0),
            'Mean |H-H*|': metrics.get('mean_h_diff', 0.0),
            'RMSE(H,H*)': metrics.get('rmse_h', 0.0),
            'Temp-State Corr': metrics.get('temp_state_corr', 0.0),
            'Temp-Entropy Corr': metrics.get('temp_entropy_corr', 0.0),
            '% T at Tmin': metrics.get('pct_tmin', 0.0),
            '% T at Tmax': metrics.get('pct_tmax', 0.0)
        })
    if rows:
        df = pd.DataFrame(rows)
        export_table(df, output_dir, "table5_regulation")

def generate_table6_efficiency(efficiency_metrics, output_dir: str):
    """
    Table 6: COMPUTATIONAL COST
    """
    rows = []
    for condition, metrics in efficiency_metrics.items():
        rows.append({
            'Condition': condition,
            'Dataset': metrics.get('dataset', 'All'),
            'Samples': metrics.get('samples', 0),
            'Total Tokens': metrics.get('total_tokens', 0),
            'Mean Tokens/Sample': metrics.get('mean_tokens_sample', 0.0),
            'Total Runtime': metrics.get('total_runtime', 0.0),
            'Mean Latency/Sample': metrics.get('mean_latency_sample', 0.0),
            'Tokens/sec': metrics.get('tokens_per_sec', 0.0),
            'Peak VRAM': metrics.get('peak_vram', 0.0)
        })
    if rows:
        df = pd.DataFrame(rows)
        export_table(df, output_dir, "table6_efficiency")
