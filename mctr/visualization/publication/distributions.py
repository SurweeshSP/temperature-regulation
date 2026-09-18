"""
Publication-grade visualizations for State Distributions (Figures 2 & 3).
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", palette="muted")

def generate_state_distributions(df: pd.DataFrame, output_path: str):
    """Figure 2: Violin plots for state vars comparing correct vs incorrect."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    features = ['Confidence', 'Normalized Entropy', 'Novelty', 'Conflict', 'Stability']
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        return
        
    fig, axes = plt.subplots(1, len(available_features), figsize=(15, 5))
    if len(available_features) == 1:
        axes = [axes]
        
    for i, feature in enumerate(available_features):
        sns.violinplot(data=df, x='Correct', y=feature, ax=axes[i], split=True, inner="quart")
        axes[i].set_title(f"{feature} Distribution")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def generate_correlation_matrix(df: pd.DataFrame, output_path: str):
    """Figure 3: Meta-State correlation matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    features = ['Confidence', 'Normalized Entropy', 'Novelty', 'Conflict', 'Stability']
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        return
        
    corr = df[available_features].corr(method='pearson')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Meta-State Correlation Matrix")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
