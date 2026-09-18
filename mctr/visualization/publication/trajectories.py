"""
Publication-grade visualizations for Trajectories (Figures 9, 10).
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="muted")

def generate_temperature_trajectory(trajectories_dict, output_path: str):
    """Figure 9: Temperature over time."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    temp = trajectories_dict.get('temperature', [])
    steps = np.arange(len(temp))
    
    sns.lineplot(x=steps, y=temp, marker="o", color="crimson", label="Temperature")
    plt.title("Inference Temperature Trajectory")
    plt.xlabel("Inference Step (t)")
    plt.ylabel("Temperature (T)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_entropy_control_trajectory(trajectories_dict, output_path: str):
    """Figure 10: Target Entropy vs Observed Entropy."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    obs_h = trajectories_dict.get('entropy', [])
    tgt_h = trajectories_dict.get('target_entropy', [])
    
    if not obs_h or not tgt_h:
        return
        
    steps = np.arange(len(obs_h))
    
    sns.lineplot(x=steps, y=obs_h, label="Observed Entropy", color="blue")
    sns.lineplot(x=steps, y=tgt_h, label="Target Entropy", color="orange", linestyle="--")
    
    plt.title("Entropy Control Trajectory")
    plt.xlabel("Inference Step (t)")
    plt.ylabel("Normalized Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
