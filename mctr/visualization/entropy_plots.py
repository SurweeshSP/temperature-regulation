import matplotlib.pyplot as plt
import numpy as np
import os

def plot_entropy_distribution(entropy: np.ndarray, normalized_entropy: np.ndarray, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(entropy.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0].set_title('Entropy Distribution')
    axes[0].set_xlabel('Entropy $H_t$')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(normalized_entropy.flatten(), bins=50, alpha=0.7, color='green')
    axes[1].set_title('Normalized Entropy Distribution')
    axes[1].set_xlabel('Normalized Entropy $\\hat{H}_t$')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_probability_concentration(probs: np.ndarray, k: int = 10, save_path: str = None):
    if probs.ndim > 2:
        probs = probs.reshape(-1, probs.shape[-1])
        
    mean_probs = np.mean(probs, axis=0)
    top_k_indices = np.argsort(mean_probs)[-k:][::-1]
    top_k_probs = mean_probs[top_k_indices]
    
    plt.figure(figsize=(8, 5))
    plt.bar(range(k), top_k_probs, color='purple', alpha=0.7)
    plt.title(f'Top-{k} Probability Concentration')
    plt.xlabel('Token Rank')
    plt.ylabel('Average Probability')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
