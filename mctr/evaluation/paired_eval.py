"""
MCTR Phase 2: Paired Evaluation Pipeline
==========================================
Runs all five Phase 2 conditions over the same canonical example set
and saves per-example results to a canonical CSV.

Conditions:
  MODE 0: Native Qwen
  MODE 1: Fixed-T sweep {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}
  MODE 2: MCTR-T
  MODE 3: MCTR-A
  MODE 4: MCTR-TA

Same examples, same tokenizer, same parser, same seeds across all conditions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import os
from typing import Dict, List, Any, Optional

from mctr.inference.phase2_generation import generate_phase2, ControlMode
from mctr.evaluation.calibration import compute_ece, compute_brier_score
from mctr.datasets.gsm8k import extract_gsm8k_answer, normalize_gsm8k_answer
from mctr.attention.attention_hooks import AttentionHookManager


def check_correctness_p2(predicted_text: str, example: dict) -> int:
    """GSM8K-style answer extraction and correctness check."""
    target_val = example.get("raw_target", example.get("target", example.get("answer", "")))
    if target_val:
        truth_raw = extract_gsm8k_answer(str(target_val))
        pred_raw  = extract_gsm8k_answer(predicted_text)
        truth = normalize_gsm8k_answer(truth_raw)
        pred  = normalize_gsm8k_answer(pred_raw)
        return 1 if truth == pred and truth is not None else 0
    return 0


def run_phase2_paired_eval(
    transformer,
    dataset: List[Dict[str, Any]],
    state_extractor,
    evaluator,
    joint_controller,
    conditions: Dict[str, Dict[str, Any]],
    device: str,
    do_sample: bool = True,
    max_new_tokens: int = 256,
    collect_attention: bool = True,
    n_layers: int = 24,
    n_heads: int  = 16,
    head_dim: int = 64,
    output_dir: str = "outputs/phase2",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs Phase 2 paired evaluation over all conditions on the same dataset.

    Returns
    -------
    Dict {condition_name → metrics_dict}
    Also writes canonical CSV to output_dir/raw/paired_results_phase2.csv
    """
    os.makedirs(f"{output_dir}/raw", exist_ok=True)

    has_model     = transformer is not None and hasattr(transformer, "model") and transformer.model is not None
    has_tokenizer = transformer is not None and hasattr(transformer, "tokenizer") and transformer.tokenizer is not None
    model_obj     = transformer.model if has_model else None

    # Build hook manager (reused across conditions for the same model)
    hook_manager = None
    if collect_attention and has_model:
        hook_manager = AttentionHookManager(model_obj, n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)

    all_rows = []
    metrics_out = {}

    for cond_name, cond_cfg in conditions.items():
        print(f"\n    ── Condition: {cond_name} ──")
        mode_str = cond_cfg.get("mode", "native")
        mode = {
            "native":   ControlMode.NATIVE,
            "fixed_t":  ControlMode.FIXED_T,
            "mctr_t":   ControlMode.MCTR_T,
            "mctr_a":   ControlMode.MCTR_A,
            "mctr_ta":  ControlMode.MCTR_TA,
        }.get(mode_str, ControlMode.NATIVE)

        baseline_temp = cond_cfg.get("temperature", None)

        # Per-condition accumulators
        all_correct, all_conf, all_ent, all_temps = [], [], [], []
        all_attn_ent_n, all_attn_ent_m, all_attn_js = [], [], []
        all_mod_mag, all_gen_tokens, all_latencies = [], [], []

        cond_start = time.perf_counter()

        for idx, example in enumerate(dataset):
            ex_id = example.get("id", idx)
            question = example.get("prompt", example.get("question", example.get("input", "")))

            if has_tokenizer:
                inputs = transformer.tokenizer(question, return_tensors="pt")
                input_ids = inputs.input_ids.to(device=device, dtype=torch.long)
                if input_ids.shape[1] == 0:
                    pad = transformer.tokenizer.eos_token_id or 1
                    input_ids = torch.tensor([[pad]], device=device, dtype=torch.long)
                eos_id = transformer.tokenizer.eos_token_id
            else:
                input_ids = torch.randint(0, 151936, (1, 10), device=device, dtype=torch.long)
                eos_id = None

            ex_start = time.perf_counter()

            with torch.inference_mode():
                output_ids, traj = generate_phase2(
                    model=model_obj,
                    input_ids=input_ids,
                    state_extractor=state_extractor,
                    evaluator=evaluator,
                    joint_controller=joint_controller,
                    hook_manager=hook_manager,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_id,
                    mode=mode,
                    baseline_temp=baseline_temp,
                    do_sample=do_sample,
                    collect_attention=(collect_attention and hook_manager is not None),
                )

            ex_latency = time.perf_counter() - ex_start

            if has_tokenizer:
                generated = output_ids[0][input_ids.shape[1]:]
                predicted_text = transformer.tokenizer.decode(generated, skip_special_tokens=True)
                gen_len = len(generated)
            else:
                predicted_text = "mock"
                gen_len = 0

            correct = check_correctness_p2(predicted_text, example)

            ex_conf = float(np.mean(traj["confidence"])) if traj["confidence"] else 0.0
            ex_ent  = float(np.mean(traj["entropy"]))    if traj["entropy"]    else 0.0
            ex_temp = float(np.mean([np.mean(x) for x in traj["temperature"]])) if traj["temperature"] else 1.0
            ex_attn_ent_n = float(np.mean(traj["attn_entropy_native"]))   if traj["attn_entropy_native"] else 0.0
            ex_attn_ent_m = float(np.mean(traj["attn_entropy_mod"]))      if traj["attn_entropy_mod"]    else 0.0
            ex_attn_js    = float(np.mean(traj["attn_js"]))               if traj["attn_js"]             else 0.0
            ex_mod_mag    = float(np.mean(traj["modulation_magnitude"]))  if traj["modulation_magnitude"] else 0.0

            all_correct.append(correct)
            all_conf.append(ex_conf)
            all_ent.append(ex_ent)
            all_temps.append(ex_temp)
            all_attn_ent_n.append(ex_attn_ent_n)
            all_attn_ent_m.append(ex_attn_ent_m)
            all_attn_js.append(ex_attn_js)
            all_mod_mag.append(ex_mod_mag)
            all_gen_tokens.append(gen_len)
            all_latencies.append(ex_latency)

            # ── Canonical row ─────────────────────────────────────────────
            all_rows.append({
                "example_id":   ex_id,
                "dataset":      example.get("dataset", "gsm8k"),
                "task":         example.get("task",    "main"),
                "seed":         seed,
                "condition":    cond_name,

                "prediction":   predicted_text,
                "target":       example.get("answer", example.get("target", "")),
                "correct":      correct,

                "confidence":   ex_conf,
                "uncertainty":  float(np.mean(traj["stability"])) if traj["stability"] else 0.0,
                "novelty":      float(np.mean(traj["novelty"]))   if traj["novelty"]   else 0.0,
                "conflict":     float(np.mean(traj["conflict"]))  if traj["conflict"]  else 0.0,
                "stability":    float(np.mean(traj["stability"])) if traj["stability"] else 0.0,
                "entropy":      ex_ent,

                "meta_score":   float(np.mean(traj["meta_score"])) if traj["meta_score"] else 0.0,
                "entropy_target": float(np.mean(traj["target_entropy"])) if traj["target_entropy"] else 0.0,
                "temperature":  ex_temp,

                "attention_entropy_native": ex_attn_ent_n,
                "attention_entropy_mod":    ex_attn_ent_m,
                "attention_js":            ex_attn_js,
                "attention_modulation":    ex_mod_mag,

                "generated_tokens": gen_len,
                "latency":          ex_latency,
                "peak_vram":        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
            })

        # ── Aggregate condition metrics ───────────────────────────────────
        n = max(len(all_correct), 1)
        ece   = compute_ece(np.array(all_conf), np.array(all_correct))   if all_conf else 0.0
        brier = compute_brier_score(np.array(all_conf), np.array(all_correct)) if all_conf else 0.0

        metrics_out[cond_name] = {
            "accuracy":               float(np.mean(all_correct)),
            "ece":                    float(ece),
            "brier":                  float(brier),
            "mean_confidence":        float(np.mean(all_conf)),
            "mean_entropy":           float(np.mean(all_ent)),
            "mean_temperature":       float(np.mean(all_temps)),
            "temperature_std":        float(np.std(all_temps)),
            "attn_entropy_native":    float(np.mean(all_attn_ent_n)),
            "attn_entropy_mod":       float(np.mean(all_attn_ent_m)),
            "attn_js":                float(np.mean(all_attn_js)),
            "modulation_magnitude":   float(np.mean(all_mod_mag)),
            "mean_generated_tokens":  float(np.mean(all_gen_tokens)),
            "total_latency":          float(time.perf_counter() - cond_start),
            "mean_latency":           float(np.mean(all_latencies)),
            "peak_vram":              float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0,
            "n_examples":             n,
        }

        print(f"      Accuracy: {metrics_out[cond_name]['accuracy']*100:.2f}%  "
              f"ECE: {metrics_out[cond_name]['ece']:.4f}  "
              f"Attn-JS: {metrics_out[cond_name]['attn_js']:.6f}")

    # ── Save canonical CSV ────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = f"{output_dir}/raw/paired_results_phase2_seed{seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n    Saved canonical results → {csv_path}")

    return metrics_out
