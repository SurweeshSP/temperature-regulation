import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from mctr.base.interface import TransformerState
from mctr.state.state import StateExtractor
from mctr.metacognition.evaluator import MetaEvaluator
from mctr.controller.controller import MCTRController, EntropyTarget

@torch.no_grad()
def generate_with_mctr(
    model,
    input_ids: torch.Tensor,
    state_extractor: StateExtractor,
    evaluator: MetaEvaluator,
    controller: MCTRController,
    entropy_target_gen: EntropyTarget,
    max_new_tokens: int = 256,
    baseline_temp: float = None,
    eos_token_id: int = None,
    do_sample: bool = True
) -> Tuple[torch.Tensor, Dict[str, list]]:
    """
    Causal generation loop for Phase 1.
    Supports both stochastic sampling (do_sample=True) and deterministic greedy decoding (do_sample=False).
    If baseline_temp is provided, runs a fixed-temperature baseline instead.
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Tracking trajectories
    trajectories = {
        'temperature': [], 'entropy': [], 'confidence': [], 
        'novelty': [], 'conflict': [], 'stability': [], 
        'target_entropy': [], 'meta_score': []
    }
    
    # Initialize state
    psi_t_minus_1 = torch.zeros(batch_size, 6, device=device)
    T_prev = torch.full((batch_size,), controller.t_min if baseline_temp is None else baseline_temp, device=device)
    H_prev = torch.zeros(batch_size, device=device)
    
    current_input_ids = input_ids.to(device=device, dtype=torch.long)
    past_key_values = None
    
    for t in range(max_new_tokens):
        # 1. Meta-Cognition & Control (based on state t-1)
        if baseline_temp is not None:
            T_t = torch.full((batch_size,), baseline_temp, device=device)
            H_star_t = torch.zeros(batch_size, device=device)
            meta_score = torch.zeros(batch_size, device=device)
        else:
            m_t_minus_1 = evaluator.mlp(psi_t_minus_1)
            meta_score = torch.sigmoid(evaluator.prediction_head(m_t_minus_1).squeeze(-1))
            H_star_t = entropy_target_gen(psi_t_minus_1)
            
            control_state = controller(m_t_minus_1, current_T=T_prev, current_H=H_prev, target_H=H_star_t)
            T_t = control_state.temperature
            
        # 2. Base Transformer Inference (Frozen)
        step_input_ids = current_input_ids if past_key_values is None else current_input_ids[:, -1:]
        step_input_ids = step_input_ids.to(device=device, dtype=torch.long)
        
        outputs = model(
            input_ids=step_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        
        logits = outputs.logits[:, -1, :] # [batch_size, vocab_size]
        hidden_state = outputs.hidden_states[-1][:, -1, :] # [batch_size, hidden_dim]
        past_key_values = outputs.past_key_values
        
        # 3. Apply Temperature Scaling
        scaled_logits = logits / T_t.unsqueeze(-1)
        probs = F.softmax(scaled_logits, dim=-1)
        
        # 4. Extract new State Psi_t
        mock_state = TransformerState(
            hidden_state=hidden_state.unsqueeze(1),
            logits=logits.unsqueeze(1), 
            probabilities=probs.unsqueeze(1)
        )
        
        # Conflict calculation
        alt_p = F.softmax(logits.unsqueeze(1) + torch.randn_like(logits.unsqueeze(1)) * 0.1, dim=-1)
        
        meta_state = state_extractor(mock_state, alt_p=alt_p)
        psi_t = meta_state.to_tensor().squeeze(1)
        current_H = meta_state.normalized_entropy.squeeze(1)
        
        # Track trajectory
        trajectories['temperature'].append(T_t.cpu().to(torch.float32).numpy())
        trajectories['entropy'].append(current_H.cpu().to(torch.float32).numpy())
        trajectories['target_entropy'].append(H_star_t.cpu().to(torch.float32).numpy())
        trajectories['meta_score'].append(meta_score.cpu().to(torch.float32).numpy())
        trajectories['confidence'].append(meta_state.confidence.squeeze(1).cpu().to(torch.float32).numpy())
        trajectories['novelty'].append(meta_state.novelty.squeeze(1).cpu().to(torch.float32).numpy())
        trajectories['conflict'].append(meta_state.conflict.squeeze(1).cpu().to(torch.float32).numpy())
        trajectories['stability'].append(meta_state.stability.squeeze(1).cpu().to(torch.float32).numpy())
        
        # 5. Decode Next Token (Stochastic Sampling vs Greedy)
        if do_sample:
            # Stochastic Sampling: temperature directly influences token choice
            next_token = torch.multinomial(probs, num_samples=1).to(device=device, dtype=torch.long)
        else:
            # Deterministic Greedy Decoding: argmax
            next_token = torch.argmax(probs, dim=-1).unsqueeze(-1).to(device=device, dtype=torch.long)
            
        current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
        
        # Update recurrences
        psi_t_minus_1 = psi_t
        T_prev = T_t
        H_prev = current_H
        
        # Early stopping
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
            
    return current_input_ids, trajectories
