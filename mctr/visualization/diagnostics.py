import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import seaborn as sns

def plot_meta_trajectory(meta_states: dict, save_path: str):
    plt.figure(figsize=(12, 6))
    for key, values in meta_states.items():
        plt.plot(values, label=key, alpha=0.8)
        
    plt.title('Meta-State Trajectory')
    plt.xlabel('Time step $t$')
    plt.ylabel('Score')
    plt.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_meta_embedding(m_t: np.ndarray, labels: np.ndarray, label_name: str, save_path: str):
    if m_t.ndim > 2:
        m_t = m_t.reshape(-1, m_t.shape[-1])
        
    pca = PCA(n_components=2)
    m_t_pca = pca.fit_transform(m_t)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(m_t_pca[:, 0], m_t_pca[:, 1], c=labels.flatten(), cmap='coolwarm', alpha=0.5, s=10)
    plt.colorbar(scatter, label=label_name)
    plt.title('Meta-State PCA Embedding')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_meta_vs_correctness(probabilities: np.ndarray, correctness: np.ndarray, save_path: str):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=correctness.flatten(), y=probabilities.flatten())
    plt.title('Evaluator Score vs Correctness')
    plt.xlabel('Correctness')
    plt.ylabel('P(Correct | $m_t$)')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
