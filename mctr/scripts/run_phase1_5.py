import os
import yaml
import json
import torch
import numpy as np
from datetime import datetime

from mctr.base.transformer import MCTRTransformerWrapper
from mctr.base.interface import TransformerState
from mctr.state.state import StateExtractor
from mctr.metacognition.evaluator import MetaEvaluator
from mctr.visualization.entropy_plots import plot_entropy_distribution, plot_probability_concentration
from mctr.visualization.state_plots import (
    plot_confidence_vs_entropy, plot_entropy_vs_correctness, plot_state_correlation,
    plot_novelty_ood, plot_conflict, plot_stability_consistency
)
from mctr.visualization.diagnostics import plot_meta_trajectory, plot_meta_embedding, plot_meta_vs_correctness

def run_experiment(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Starting Phase 1-5 Experiment: {config['experiment']['name']}")
    
    # 1. Setup Directories
    os.makedirs("artifacts/metrics", exist_ok=True)
    os.makedirs("artifacts/plots", exist_ok=True)
    os.makedirs("artifacts/checkpoints", exist_ok=True)
    
    # 2. Setup Device & Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 151936 # Qwen2.5 approx vocab size, will be updated if model loads
    hidden_size = 1536 # For 1.5B
    
    print("Loading Transformer...")
    try:
        transformer = MCTRTransformerWrapper(
            config['backbone']['name'], 
            device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        vocab_size = transformer.model.config.vocab_size
        hidden_size = transformer.model.config.hidden_size
        print("Transformer loaded successfully.")
    except Exception as e:
        print(f"Failed to load transformer, falling back to mock mode for testing. Error: {e}")
        transformer = None
        
    # 3. Setup Components
    state_extractor = StateExtractor(vocab_size=vocab_size, config=config['state'])
    meta_evaluator = MetaEvaluator(
        input_dim=config['meta']['input_dim'],
        hidden_dim=config['meta']['hidden_dim'],
        meta_dim=config['meta']['output_dim']
    ).to(device)
    
    # 4. Generate/Simulate Data
    batch_size = 8
    seq_len = 16
    
    if transformer:
        # Generate some dummy inputs for real model
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        print("Running forward pass...")
        with torch.no_grad():
            state = transformer.forward_with_state(input_ids)
            # Create a mock alt distribution for conflict
            alt_logits = state.logits + torch.randn_like(state.logits) * 0.1
            alt_p = torch.softmax(alt_logits, dim=-1)
    else:
        print("Simulating mock forward pass...")
        h = torch.randn(batch_size, seq_len, hidden_size, device=device)
        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        probs = torch.softmax(logits, dim=-1)
        state = TransformerState(hidden_state=h, logits=logits, probabilities=probs)
        
        alt_logits = logits + torch.randn_like(logits) * 0.1
        alt_p = torch.softmax(alt_logits, dim=-1)
        
    # 5. Extract State
    print("Extracting Meta-State...")
    # Fit novelty reference using the current batch as reference (for demo)
    state_extractor.novelty_ref.fit(state.hidden_state.view(-1, hidden_size))
    meta_state = state_extractor(state, alt_p=alt_p)
    
    psi_t = meta_state.to_tensor()
    print(f"Meta-State shape: {psi_t.shape}")
    
    # 6. Evaluate
    print("Running Meta-Evaluator...")
    m_t = meta_evaluator(meta_state)
    y_hat = meta_evaluator.predict_correctness(m_t)
    
    # 7. Generate Visualizations
    print("Generating Visualizations...")
    h_numpy = meta_state.entropy.cpu().numpy()
    h_norm_numpy = meta_state.normalized_entropy.cpu().numpy()
    plot_entropy_distribution(h_numpy, h_norm_numpy, "artifacts/plots/entropy_distribution.png")
    
    plot_probability_concentration(state.probabilities.cpu().numpy(), k=10, save_path="artifacts/plots/probability_concentration.png")
    
    c_numpy = meta_state.confidence.cpu().numpy()
    plot_confidence_vs_entropy(c_numpy, h_norm_numpy, "artifacts/plots/confidence_entropy.png")
    
    # Mock labels for correctness (random for demo)
    correctness = np.random.randint(0, 2, size=h_numpy.shape)
    plot_entropy_vs_correctness(h_norm_numpy, correctness, "artifacts/plots/entropy_correctness.png")
    
    psi_numpy = psi_t.cpu().numpy().reshape(-1, 6)
    labels = ['Conf', 'Uncert', 'Novelty', 'Conflict', 'Stab', 'Entropy']
    plot_state_correlation(psi_numpy, labels, "artifacts/plots/state_correlation.png")
    
    is_ood = np.random.randint(0, 2, size=h_numpy.shape)
    plot_novelty_ood(meta_state.novelty.cpu().numpy(), is_ood, "artifacts/plots/novelty_ood.png")
    
    is_conflict = np.random.randint(0, 2, size=h_numpy.shape)
    plot_conflict(meta_state.conflict.cpu().numpy(), is_conflict, "artifacts/plots/conflict.png")
    
    consistency = np.random.rand(*h_numpy.shape)
    plot_stability_consistency(meta_state.stability.cpu().numpy(), consistency, "artifacts/plots/stability_consistency.png")
    
    traj_dict = {
        'Confidence': c_numpy[0],
        'Uncertainty': meta_state.uncertainty.cpu().numpy()[0],
        'Novelty': meta_state.novelty.cpu().numpy()[0],
        'Conflict': meta_state.conflict.cpu().numpy()[0],
        'Stability': meta_state.stability.cpu().numpy()[0],
        'Entropy': h_norm_numpy[0]
    }
    plot_meta_trajectory(traj_dict, "artifacts/plots/meta_trajectory.png")
    
    m_t_numpy = m_t.detach().cpu().numpy()
    plot_meta_embedding(m_t_numpy, correctness, 'Correctness', "artifacts/plots/meta_embedding.png")
    
    y_hat_numpy = y_hat.detach().cpu().numpy()
    plot_meta_vs_correctness(y_hat_numpy, correctness, "artifacts/plots/meta_vs_correctness.png")
    
    # 8. Save Metrics & Checkpoints
    print("Saving Checkpoint and Metrics...")
    torch.save(meta_evaluator.state_dict(), "artifacts/checkpoints/meta_evaluator.pt")
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": config['backbone']['name'],
        "pytorch_version": torch.__version__,
        "device": device,
        "batch_size": batch_size,
        "seq_len": seq_len
    }
    with open("artifacts/metrics/meta_metrics.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Phase 1-5 completed successfully.")

if __name__ == "__main__":
    run_experiment("mctr/configs/phase5_meta_evaluator.yaml")
