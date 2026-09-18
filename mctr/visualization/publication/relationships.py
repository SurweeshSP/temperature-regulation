"""
Relationships and Scatter plots (Figures 7, 8, 11).
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", palette="muted")

def generate_entropy_vs_confidence(df: pd.DataFrame, output_path: str):
    """Figure 7: Entropy vs Confidence grouped by correctness."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if 'Normalized Entropy' not in df.columns or 'Confidence' not in df.columns:
        return
        
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Normalized Entropy', y='Confidence', hue='Correct', alpha=0.6)
    plt.title("Predictive Uncertainty vs Confidence")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_temp_vs_meta(df: pd.DataFrame, output_path: str):
    """Figure 11: Temperature vs Meta-Score."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if 'Temperature' not in df.columns or 'Meta-Score' not in df.columns:
        return
        
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x='Meta-Score', y='Temperature', scatter_kws={'alpha':0.5})
    plt.title("Temperature vs Meta-Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
