import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_confidence_vs_entropy(confidence: np.ndarray, entropy: np.ndarray, save_path: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(entropy.flatten(), confidence.flatten(), alpha=0.1, s=2)
    plt.title('Confidence vs Normalized Entropy')
    plt.xlabel('Normalized Entropy $\\hat{H}_t$')
    plt.ylabel('Confidence $C_t$')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_entropy_vs_correctness(entropy: np.ndarray, correctness: np.ndarray, save_path: str):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=correctness.flatten(), y=entropy.flatten())
    plt.title('Entropy vs Correctness')
    plt.xlabel('Correctness')
    plt.ylabel('Normalized Entropy $\\hat{H}_t$')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_state_correlation(state_matrix: np.ndarray, labels: list, save_path: str):
    # state_matrix shape: [N, D]
    corr = np.corrcoef(state_matrix, rowvar=False)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, xticklabels=labels, yticklabels=labels, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('State Correlation Matrix')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_novelty_ood(novelty: np.ndarray, is_ood: np.ndarray, save_path: str):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=is_ood.flatten(), y=novelty.flatten())
    plt.title('Novelty vs OOD')
    plt.xlabel('Is OOD')
    plt.ylabel('Novelty Score $N_t$')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_conflict(conflict: np.ndarray, is_conflict: np.ndarray, save_path: str):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=is_conflict.flatten(), y=conflict.flatten())
    plt.title('Conflict Score under Disagreement')
    plt.xlabel('Is Conflict')
    plt.ylabel('Conflict $K_t$')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_stability_consistency(stability: np.ndarray, consistency: np.ndarray, save_path: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(consistency.flatten(), stability.flatten(), alpha=0.1, s=2)
    plt.title('Stability vs Consistency')
    plt.xlabel('Cross-run Consistency')
    plt.ylabel('Stability Score $S_t$')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
