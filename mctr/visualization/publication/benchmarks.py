"""
Publication-grade visualizations for Benchmarks (Figures 12, 13, 14, 15).
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", palette="muted")

def generate_baseline_comparison(df: pd.DataFrame, output_path: str):
    """Figure 12: Compact benchmark comparison (Accuracy/NLL/ECE)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Task', y='Accuracy', hue='Model')
    plt.title("Baseline vs MCTR Phase 1")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_fixed_temperature_sweep(df: pd.DataFrame, output_path: str):
    """Figure 13: Fixed Temperature vs MCTR Accuracy."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if df.empty: return
    
    plt.figure(figsize=(8, 5))
    
    # Assuming df has 'Temperature', 'Accuracy', 'Type' (Fixed/MCTR)
    fixed_df = df[df['Type'] == 'Fixed']
    sns.lineplot(data=fixed_df, x='Temperature', y='Accuracy', marker='o', label='Fixed Sweep')
    
    mctr_df = df[df['Type'] == 'MCTR']
    if not mctr_df.empty:
        # Plot horizontal line for MCTR
        mctr_acc = mctr_df['Accuracy'].mean()
        plt.axhline(mctr_acc, color='r', linestyle='--', label=f'MCTR Adaptive ({mctr_acc:.2f})')
        
    plt.title("Temperature Sensitivity vs MCTR")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
