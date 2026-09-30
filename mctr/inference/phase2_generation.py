"""
MCTR Phase 2: Generation Loop
================================
Implements the causal Phase 2 generation loop:

    Ψ_(t-1)
        ↓
    Meta-Evaluator M_φ  →  m_(t-1)
        ↓
    JointController G_ω
        ├──────→  T_t        (temperature)
        └──────→  A_t        (attention modulation)
            ↓
    Regulated Qwen  →  logits_t
        ↓
    softmax(logits / T_t)  →  P_t
        ↓
    y_t ~ Categorical(P_t)   [torch.multinomial]
        ↓
    StateExtractor  →  Ψ_t

Temporal causality enforcement:
  - Controller ONLY receives Ψ_(t-1), never Ψ_t.
  - Ψ_t is computed AFTER y_t is sampled.
  - Attention modulation at step t uses m_(t-1).

Control Modes (MODE 0-4)
  MODE 0: Native Qwen (no control)
  MODE 1: Fixed temperature (no attention)
  MODE 2: MCTR-T  (adaptive temperature only)
  MODE 3: MCTR-A  (adaptive attention only)
  MODE 4: MCTR-TA (adaptive temperature + adaptive attention)
"""

import torch
import torch.nn.functional as F
import math
import time
from typing import Dict, Any, Tuple, Optional, List
from enum import IntEnum

from mctr.base.interface import TransformerState
from mctr.state.state import StateExtractor
from mctr.metacognition.evaluator import MetaEvaluator
from mctr.control.joint_controller import JointController
from mctr.control.attention_controller import AttentionControlMode
from mctr.attention.attention_hooks import AttentionHookManager
from mctr.attention.attention_metrics import AttentionMetrics


class ControlMode(IntEnum):
    """Phase 2 control modes."""
    NATIVE    = 0  # No MCTR
    FIXED_T   = 1  # Fixed temperature, no attention
    MCTR_T    = 2  # Adaptive temperature only
    MCTR_A    = 3  # Adaptive attention only
    MCTR_TA   = 4  # Adaptive temperature + attention


def _build_initial_psi(batch_size: int, device: torch.device) -> torch.Tensor:
    """Return zero Ψ_(t=-1) for warm-start of the causal loop."""
    return torch.zeros(batch_size, 6, device=device)


def _check_collapse_warnings(
    temperatures: list,
    attn_metrics_list: list,
    t_min: float,
    t_max: float,
    step: int,
):
    """Emit warnings for degenerate controller behavior."""
    if len(temperatures) > 2:
        t_arr = torch.tensor(temperatures)
        if t_arr.std().item() < 1e-4:
            print(f"  [WARN step={step}] Temperature collapse: std(T) ≈ 0")
        frac_min = (t_arr < t_min + 1e-3).float().mean().item()
        frac_max = (t_arr > t_max - 1e-3).float().mean().item()
        if frac_min > 0.9:
            print(f"  [WARN step={step}] Controller saturated at T_min")
        if frac_max > 0.9:
            print(f"  [WARN step={step}] Controller saturated at T_max")

    if attn_metrics_list:
        last = attn_metrics_list[-1]
        if last.mean_js() < 1e-6:
            print(f"  [WARN step={step}] Attention modulation collapse: ΔA ≈ 0")


@torch.no_grad()
def generate_phase2(
    model,
    input_ids: torch.Tensor,
    state_extractor: StateExtractor,
    evaluator: MetaEvaluator,
    joint_controller: JointController,
    hook_manager: Optional[AttentionHookManager] = None,
    max_new_tokens: int = 256,
    eos_token_id: Optional[int] = None,
    mode: ControlMode = ControlMode.MCTR_TA,
    baseline_temp: Optional[float] = None,
    do_sample: bool = True,
    collect_attention: bool = True,
) -> Tuple[torch.Tensor, Dict[str, list]]:
    """
    Phase 2 causal generation loop with temporal causality guaranteed.

    Parameters
    ----------
    model            : frozen Qwen model
    input_ids        : Tensor [batch, seq_len]
    state_extractor  : StateExtractor
    evaluator        : MetaEvaluator (provides m_t = M_φ(Ψ_t))
    joint_controller : JointController (provides T_t, A_t)
    hook_manager     : AttentionHookManager (for attention capture/modulation)
    max_new_tokens   : int
    eos_token_id     : int or None
    mode             : ControlMode
    baseline_temp    : float — used when mode ∈ {NATIVE, FIXED_T}
    do_sample        : bool — True for stochastic, False for greedy (sanity only)
    collect_attention: bool — whether to capture attention maps

    Returns
    -------
    (generated_ids, trajectories_dict)
    """
    device     = input_ids.device
    batch_size = input_ids.shape[0]

    # ── Trajectory tracking ────────────────────────────────────────────────
    trajectories: Dict[str, list] = {
        "temperature":    [],
        "entropy":        [],
        "confidence":     [],
        "novelty":        [],
        "conflict":       [],
        "stability":      [],
        "target_entropy": [],
        "meta_score":     [],
        # Phase 2 additions
        "attn_entropy_native":    [],
        "attn_entropy_mod":       [],
        "attn_concentration":     [],
        "attn_js":                [],
        "modulation_magnitude":   [],
        "responsiveness":         [],
        "temporal_stability":     [],
        "latency_per_step":       [],
    }

    # ── Initial state ──────────────────────────────────────────────────────
    psi_prev = _build_initial_psi(batch_size, device)  # Ψ_(t-1)
    T_prev   = torch.full((batch_size,), joint_controller.t_min if baseline_temp is None else baseline_temp, device=device)
    H_prev   = torch.zeros(batch_size, device=device)

    current_input_ids = input_ids.to(device=device, dtype=torch.long)
    past_key_values   = None

    # For attention stability
    prev_native_maps:   Optional[List] = None
    prev_modulation_tensor: Optional[torch.Tensor] = None

    # ── State extractor reset ─────────────────────────────────────────────
    state_extractor.reset_history()

    for t in range(max_new_tokens):
        step_start = time.perf_counter()

        # ── STEP 1: Meta-Cognition (uses Ψ_(t-1)) ─────────────────────────
        # CAUSAL CONTRACT: psi_prev is the state from the PREVIOUS step.
        # The controller NEVER sees the current step's attention/state.

        if mode == ControlMode.NATIVE:
            # No meta-control
            T_t        = torch.full((batch_size,), 1.0, device=device)
            meta_score = torch.zeros(batch_size, device=device)
            attn_state = None
            H_star_t   = torch.zeros(batch_size, device=device)

        elif mode == ControlMode.FIXED_T:
            bt = baseline_temp if baseline_temp is not None else 1.0
            T_t        = torch.full((batch_size,), bt, device=device)
            meta_score = torch.zeros(batch_size, device=device)
            attn_state = None
            H_star_t   = torch.zeros(batch_size, device=device)

        else:
            # Compute meta-embedding from Ψ_(t-1)
            m_prev = evaluator.mlp(psi_prev)  # [batch, meta_dim]

            # Configure controller based on mode
            use_temp  = mode in (ControlMode.MCTR_T, ControlMode.MCTR_TA)
            use_attn  = mode in (ControlMode.MCTR_A, ControlMode.MCTR_TA)

            joint_controller.enable_temperature = use_temp
            joint_controller.enable_attention   = use_attn

            ctrl_out = joint_controller(
                m_prev,
                baseline_temp=baseline_temp if not use_temp else None,
            )

            T_t        = ctrl_out.temperature
            attn_state = ctrl_out.attention if use_attn else None
            meta_score = ctrl_out.meta_score
            H_star_t   = torch.zeros(batch_size, device=device)  # placeholder

        # ── STEP 2: Transformer Forward Pass ──────────────────────────────
        step_input_ids = (
            current_input_ids if past_key_values is None
            else current_input_ids[:, -1:]
        ).to(device=device, dtype=torch.long)

        fwd_kwargs = dict(
            input_ids=step_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=(collect_attention and hook_manager is not None),
        )

        if collect_attention and hook_manager is not None and attn_state is not None:
            # Apply attention modulation via hooks
            with hook_manager.apply_modulation(attn_state):
                outputs = model(**fwd_kwargs)
        else:
            outputs = model(**fwd_kwargs)

        logits       = outputs.logits[:, -1, :]           # [batch, vocab]
        hidden_state = outputs.hidden_states[-1][:, -1, :] # [batch, hidden]
        past_key_values = outputs.past_key_values

        # ── STEP 3: Temperature Scaling & Probability ──────────────────────
        # For MODE 0 (native), T_t = 1.0 → no distortion
        if mode == ControlMode.NATIVE:
            scaled_logits = logits
        else:
            scaled_logits = logits / T_t.unsqueeze(-1).clamp(min=1e-6)

        probs = F.softmax(scaled_logits, dim=-1)

        assert torch.all(torch.isfinite(probs)), "generate_phase2: non-finite probabilities!"

        # ── STEP 4: Sample Token ──────────────────────────────────────────
        if do_sample:
            # Stochastic sampling: y_t ~ Categorical(P_t)
            next_token = torch.multinomial(probs, num_samples=1).to(device=device, dtype=torch.long)
        else:
            # Greedy (sanity-check only)
            next_token = torch.argmax(probs, dim=-1, keepdim=True).to(device=device, dtype=torch.long)

        current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)

        # ── STEP 5: Extract Ψ_t from current step's output ─────────────────
        # This Ψ_t will become Ψ_(t-1) for the NEXT step.
        mock_state = TransformerState(
            hidden_state=hidden_state.unsqueeze(1),
            logits=logits.unsqueeze(1),
            probabilities=probs.unsqueeze(1),
        )
        alt_p = F.softmax(logits.unsqueeze(1) + torch.randn_like(logits.unsqueeze(1)) * 0.1, dim=-1)

        meta_state = state_extractor(mock_state, alt_p=alt_p)
        psi_t      = meta_state.to_tensor().squeeze(1)     # [batch, 6]
        current_H  = meta_state.normalized_entropy.squeeze(1)  # [batch]

        # ── STEP 6: Compute Attention Metrics ─────────────────────────────
        attn_m = AttentionMetrics()
        if collect_attention and hook_manager is not None:
            mod_tensor = (
                attn_state.modulation if attn_state is not None else None
            )
            attn_m = AttentionMetrics.compute(
                native_maps=hook_manager.native_attn_maps,
                modulated_maps=hook_manager.modulated_attn_maps,
                modulation=mod_tensor,
                prev_native_maps=prev_native_maps,
                delta_psi=psi_t - psi_prev if t > 0 else None,
                prev_modulation=prev_modulation_tensor,
            )
            prev_native_maps    = list(hook_manager.native_attn_maps)
            prev_modulation_tensor = mod_tensor

        step_latency = time.perf_counter() - step_start

        # ── Collapse Checks ────────────────────────────────────────────────
        if t > 0 and t % 20 == 0:
            _check_collapse_warnings(
                trajectories["temperature"],
                [attn_m],
                joint_controller.t_min,
                joint_controller.t_max,
                t,
            )

        # ── Record Trajectories ────────────────────────────────────────────
        trajectories["temperature"].append(T_t.cpu().to(torch.float32).numpy())
        trajectories["entropy"].append(current_H.cpu().to(torch.float32).numpy())
        trajectories["target_entropy"].append(H_star_t.cpu().to(torch.float32).numpy())
        trajectories["meta_score"].append(meta_score.cpu().to(torch.float32).numpy())
        trajectories["confidence"].append(meta_state.confidence.squeeze(1).cpu().to(torch.float32).numpy())
        trajectories["novelty"].append(meta_state.novelty.squeeze(1).cpu().to(torch.float32).numpy())
        trajectories["conflict"].append(meta_state.conflict.squeeze(1).cpu().to(torch.float32).numpy())
        trajectories["stability"].append(meta_state.stability.squeeze(1).cpu().to(torch.float32).numpy())
        trajectories["attn_entropy_native"].append(attn_m.mean_entropy_native())
        trajectories["attn_entropy_mod"].append(attn_m.mean_entropy_modulated())
        trajectories["attn_concentration"].append(attn_m.mean_concentration_native())
        trajectories["attn_js"].append(attn_m.mean_js())
        trajectories["modulation_magnitude"].append(attn_m.modulation_magnitude)
        trajectories["responsiveness"].append(attn_m.responsiveness)
        trajectories["temporal_stability"].append(attn_m.temporal_stability)
        trajectories["latency_per_step"].append(step_latency)

        # ── Update Recurrence ─────────────────────────────────────────────
        # CAUSAL: psi_prev ← Ψ_t (current step's state feeds NEXT step's controller)
        psi_prev = psi_t
        T_prev   = T_t
        H_prev   = current_H

        # ── Early Stop ───────────────────────────────────────────────────
        if eos_token_id is not None and (next_token.squeeze(-1) == eos_token_id).all():
            break

    return current_input_ids, trajectories
