import torch
from tqdm import tqdm
import time
import numpy as np

from mctr.inference.phase1_generation import generate_with_mctr
from mctr.datasets.gsm8k import extract_gsm8k_answer, normalize_gsm8k_answer
from mctr.evaluation.calibration import compute_ece, compute_brier_score

def check_correctness(predicted_text, example):
    """Fallback simplistic exact-match correctness checker."""
    target_val = example.get('raw_target', example.get('target', example.get('answer', '')))
    if target_val: # GSM8K
        truth_raw = extract_gsm8k_answer(str(target_val))
        pred_raw = extract_gsm8k_answer(predicted_text)
        
        truth = normalize_gsm8k_answer(truth_raw)
        pred = normalize_gsm8k_answer(pred_raw)
        
        return 1 if truth == pred and truth is not None else 0
    return 0


def run_inference_loop(transformer, dataset, state_extractor, evaluator, controller, conditions, device, do_sample: bool = True):
    """
    Executes the inference loop over the dataset for specified conditions (e.g., Fixed-T sweeps, MCTR-T).
    Returns a dictionary of metrics for each condition.
    """
    metrics_out = {}
    
    has_tokenizer = transformer is not None and hasattr(transformer, 'tokenizer') and transformer.tokenizer is not None
    if transformer is not None and hasattr(transformer, 'model') and transformer.model is not None:
        model_obj = transformer.model
    else:
        model_obj = None

    
    import pandas as pd
    import os
    
    paired_results = []
    
    for condition_name, condition_cfg in conditions.items():
        print(f"      -> Running condition: {condition_name}")
        
        # Tracking metrics
        correct_count = 0
        total_nll = 0.0
        total_entropy = 0.0
        total_tokens = 0
        
        all_confidences = []
        all_correctness = []
        all_temperatures = []
        all_target_entropy_mae = []
        all_generated_tokens = []
        all_truncated = []
        
        start_time = time.time()
        
        for idx, example in enumerate(tqdm(dataset, desc=f"Evaluating {condition_name}")):
            # Tokenize question/prompt
            question = example.get('prompt', example.get('question', example.get('input', '')))
            if not question:
                question = "Question: "
                
            if has_tokenizer:
                inputs = transformer.tokenizer(question, return_tensors="pt")
                input_ids = inputs.input_ids.to(device=device, dtype=torch.long)
                if input_ids.shape[1] == 0:
                    pad_id = transformer.tokenizer.eos_token_id if transformer.tokenizer.eos_token_id is not None else 1
                    input_ids = torch.tensor([[pad_id]], device=device, dtype=torch.long)
                eos_token_id = transformer.tokenizer.eos_token_id
            else:
                input_ids = torch.randint(0, 151936, (1, 10), device=device, dtype=torch.long)
                eos_token_id = None

                
            baseline_temp = condition_cfg.get('temperature', None) if condition_cfg.get('type') == 'fixed' else None
            
            # Apply ablation overrides if it's MCTR
            if condition_cfg.get('type') == 'mctr':
                # Normally we'd pass this flag to the controller, but for now we execute the base loop
                # Future: implement ablation masking
                pass
                
            # RUN CAUSAL PIPELINE
            from mctr.controller.controller import EntropyTarget
            entropy_tgt = EntropyTarget(config={}).to(device)
            
            with torch.inference_mode():
                output_ids, trajectories = generate_with_mctr(
                    model=model_obj,
                    input_ids=input_ids,
                    state_extractor=state_extractor,
                    evaluator=evaluator,
                    controller=controller,
                    entropy_target_gen=entropy_tgt,
                    max_new_tokens=256,
                    baseline_temp=baseline_temp,
                    eos_token_id=eos_token_id,
                    do_sample=do_sample
                )
                
            # Decode and Check correctness
            if has_tokenizer:
                generated_tokens = output_ids[0][input_ids.shape[1]:]
                predicted_text = transformer.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                gen_len = len(generated_tokens)
                truncated = gen_len >= 256
            else:
                predicted_text = "mock_answer"
                gen_len = 0
                truncated = False
                
            correct = check_correctness(predicted_text, example)
            
            # Aggregate Trajectory Metrics for this example
            ex_temp = np.mean(trajectories['temperature']) if trajectories['temperature'] else 0.0
            ex_ent = np.mean(trajectories['entropy']) if trajectories['entropy'] else 0.0
            ex_conf = np.mean(trajectories['confidence']) if trajectories['confidence'] else 0.0
            
            correct_count += correct
            total_entropy += ex_ent
            total_tokens += gen_len
            
            all_confidences.append(ex_conf)
            all_correctness.append(correct)
            all_temperatures.extend(trajectories['temperature'])
            all_generated_tokens.append(gen_len)
            all_truncated.append(truncated)
            
            if trajectories['target_entropy'] and trajectories['entropy']:
                h_target = np.array(trajectories['target_entropy'])
                h_obs = np.array(trajectories['entropy'])
                all_target_entropy_mae.append(np.mean(np.abs(h_target - h_obs)))
                
            # Log for paired dataset
            paired_results.append({
                'example_id': example.get('id', idx),
                'condition': condition_name,
                'prediction': predicted_text,
                'target': example.get('answer', example.get('target', '')),
                'correct': correct,
                'confidence': ex_conf,
                'entropy': ex_ent,
                'temperature': ex_temp,
                'latency': time.time() - start_time, # approximate accumulated latency for now
                'generated_tokens': gen_len,
                'truncated': truncated
            })
                
                
        latency = time.time() - start_time
        num_examples = max(len(dataset), 1)
        
        # Calculate Calibration Metrics
        ece = compute_ece(np.array(all_confidences), np.array(all_correctness)) if all_confidences else 0.0
        brier = compute_brier_score(np.array(all_confidences), np.array(all_correctness)) if all_confidences else 0.0
        
        metrics_out[condition_name] = {
            'accuracy': correct_count / num_examples,
            'nll': total_nll / num_examples, # mocked until NLL extraction added
            'ece': ece,
            'brier': brier,
            'mean_entropy': total_entropy / num_examples,
            'mean_confidence': np.mean(all_confidences) if all_confidences else 0.0,
            'mean_temperature': np.mean(all_temperatures) if all_temperatures else 0.0,
            'temperature_std': np.std(all_temperatures) if all_temperatures else 0.0,
            'temperature_min': np.min(all_temperatures) if all_temperatures else 0.0,
            'temperature_max': np.max(all_temperatures) if all_temperatures else 0.0,
            'entropy_target_mae': np.mean(all_target_entropy_mae) if all_target_entropy_mae else 0.0,
            'tokens': total_tokens,
            'mean_generated_tokens': np.mean(all_generated_tokens) if all_generated_tokens else 0.0,
            'std_generated_tokens': np.std(all_generated_tokens) if all_generated_tokens else 0.0,
            'truncation_rate': np.mean(all_truncated) if all_truncated else 0.0,
            'latency': latency,
            'peak_vram': torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        }
        
    # Save the paired results to disk immediately
    os.makedirs('outputs/phase1/tables', exist_ok=True)
    pd.DataFrame(paired_results).to_csv('outputs/phase1/tables/paired_results_log.csv', index=False)
        
    return metrics_out
