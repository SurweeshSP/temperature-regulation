import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_fast_run_plots(paired_results_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(paired_results_path):
        print(f"Missing paired results data file. Cannot generate fast run plots.")
        return
        
    df_paired = pd.read_csv(paired_results_path)
    
    def simple_ece(conf, corr, bins=10):
        conf = np.array(conf)
        corr = np.array(corr)
        bin_boundaries = np.linspace(0, 1, bins + 1)
        ece = 0.0
        for i in range(bins):
            bin_mask = (conf >= bin_boundaries[i]) & (conf < (bin_boundaries[i+1] if i < bins-1 else 1.01))
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(corr[bin_mask])
                bin_conf = np.mean(conf[bin_mask])
                ece += (np.sum(bin_mask) / len(conf)) * np.abs(bin_acc - bin_conf)
        return ece
        
    def simple_brier(conf, corr):
        return np.mean((np.array(conf) - np.array(corr))**2)
        
    agg_rows = []
    for cond in df_paired['condition'].unique():
        cond_df = df_paired[df_paired['condition'] == cond]
        ece = simple_ece(cond_df['confidence'], cond_df['correct'])
        brier = simple_brier(cond_df['confidence'], cond_df['correct'])
        acc = cond_df['correct'].mean()
        agg_rows.append({
            'Condition': cond,
            'Accuracy': acc,
            'ECE': ece,
            'Brier': brier,
            'Mean Entropy': cond_df['entropy'].mean(),
            'Mean Confidence': cond_df['confidence'].mean(),
            'Mean T': cond_df['temperature'].mean()
        })
        
    df_agg = pd.DataFrame(agg_rows)
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df_agg['Condition'], df_agg['Accuracy'], color='skyblue')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.png")
    plt.close()
    
    # 2. Calibration Comparison (ECE & Brier)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_agg['Condition']))
    width = 0.35
    ax1.bar(x - width/2, df_agg['ECE'], width, label='ECE', color='orange')
    ax1.bar(x + width/2, df_agg['Brier'], width, label='Brier', color='brown')
    ax1.set_ylabel('Score')
    ax1.set_title('Calibration Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_agg['Condition'], rotation=45)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_comparison.png")
    plt.close()
    
    # 3. Accuracy vs ECE
    plt.figure(figsize=(8, 6))
    for i, row in df_agg.iterrows():
        plt.scatter(row['ECE'], row['Accuracy'], label=row['Condition'])
        plt.annotate(row['Condition'], (row['ECE'], row['Accuracy']))
    plt.xlabel('ECE (Lower is better)')
    plt.ylabel('Accuracy (Higher is better)')
    plt.title('Accuracy vs ECE')
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_vs_ece.png")
    plt.close()
    
    # 4. Entropy Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df_agg['Condition'], df_agg['Mean Entropy'], color='purple')
    plt.ylabel('Mean Entropy')
    plt.title('Entropy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/entropy_comparison.png")
    plt.close()
    
    # 5. Confidence vs Correctness
    plt.figure(figsize=(8, 6))
    correct_conf = df_paired[df_paired['correct'] == 1]['confidence']
    incorrect_conf = df_paired[df_paired['correct'] == 0]['confidence']
    plt.hist(correct_conf, bins=20, alpha=0.5, label='Correct', color='green')
    plt.hist(incorrect_conf, bins=20, alpha=0.5, label='Incorrect', color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence vs Correctness')
    plt.legend()
    plt.savefig(f"{output_dir}/confidence_correctness.png")
    plt.close()
    
    # 6. MCTR Temperature Distribution
    plt.figure(figsize=(8, 6))
    mctr_temps = df_paired[df_paired['condition'] == 'MCTR-T']['temperature']
    plt.hist(mctr_temps, bins=20, color='teal')
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')
    plt.title('MCTR Temperature Distribution')
    plt.savefig(f"{output_dir}/mctr_temperature_distribution.png")
    plt.close()
    
    # 7. Entropy vs Temperature (Scatter for MCTR)
    plt.figure(figsize=(8, 6))
    mctr_df = df_paired[df_paired['condition'] == 'MCTR-T']
    plt.scatter(mctr_df['entropy'], mctr_df['temperature'], alpha=0.5)
    plt.xlabel('Entropy')
    plt.ylabel('Temperature')
    plt.title('Entropy vs Temperature (MCTR-T)')
    plt.savefig(f"{output_dir}/temperature_vs_entropy.png")
    plt.close()
