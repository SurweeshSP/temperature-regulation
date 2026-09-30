# MCTR Phase 2 Completion Report

Generated: 2026-10-01T05:11:18.012928
Model: Qwen/Qwen2.5-0.5B-Instruct
Phase: 2 - Meta-Cognitive Attention Regulation
Smoke Test: True

---

## Implementation Status

### Phase 2 Architecture
- [x] Meta-Evaluator M_phi preserved from Phase 1
- [x] JointController G_omega with shared trunk
- [x] Temperature Head: T_t in [0.3, 1.2]
- [x] Attention Head: DeltaA_t in [-2.0, +2.0]
- [x] Attention modulation: additive log-space bias to attention weights
- [x] Three attention levels: global / layer / head
- [x] Temporal causality enforced: Psi_(t-1) -> controller -> A_t -> P_t -> Psi_t
- [x] Stochastic decoding: y_t ~ Categorical(P_t) via torch.multinomial
- [x] AttentionHookManager for attention map capture
- [x] All 5 control modes (Native/Fixed-T/MCTR-T/MCTR-A/MCTR-TA)
- [x] Attention metrics: entropy, concentration, JS/KL divergence, L1
- [x] Collapse detection warnings
- [x] Canonical per-example CSV output
- [x] State redundancy reporting

### State Redundancy (Known - Inherited from Phase 1)
- U_t = H_norm_t by construction (normalized_entropy == uncertainty)
- High correlations may exist between C_t/S_t and H_t/N_t/K_t
- This redundancy is preserved for backward compatibility
- Full redundancy analysis in: outputs/phase2/tables/state_pearson_correlation.csv

---

## Experimental Configuration

- Dataset:        GSM8K Main (n=5)
- Seeds:          [42]
- Max Tokens:     256
- Decoding:       Stochastic (torch.multinomial)
- Control Modes:  Native, Fixed-T x6, MCTR-T, MCTR-A, MCTR-TA

---

## Results Summary

| Condition | Accuracy | ECE | Brier | Attn JS |
|:---|:---|:---|:---|:---|
| Native   | 0.0000 | 0.8087 | 0.6588 | - |
| MCTR-T   | 0.0000 | 0.8636 | 0.7529 | - |
| MCTR-A   | 0.0000 | 0.7921 | 0.6371 | 0.0000 |
| MCTR-TA  | 0.0000 | 0.8646 | 0.7532 | 0.0000 |

---

## Hypothesis Test Results

- **H1**: H1: Meta-state Psi predicts correctness -> _Cannot compute AUROC: single class, AUROC undefined_
- **H4**: H4: MCTR-A produces measurable attention redistribution -> _Null result (JS ≈ 0)_
- **H5**: H5: MCTR-TA ≠ MCTR-T -> _Requires experiment data_

---

## Efficiency

- Total Latency: 44.23s
- Tokens/sec: 27.1
- Peak VRAM: 993.96 MB

---

## Key Findings

1. **Temperature Control**: Inherited from Phase 1. argmax(z/T) = argmax(z);
   therefore stochastic decoding is mandatory for behavioral evaluation. [OK]

2. **Attention Modulation**: Bounded additive log-space modulation applied.
   DeltaA in [-2.0, +2.0]. Attention sums to 1 post-modulation.

3. **Temporal Causality**: Psi_(t-1) -> controller -> A_t is strictly enforced.
   No same-step circular dependency.

4. **State Redundancy**: U_t = H_norm_t (Uncertainty = Normalized Entropy) by
   construction. Explicitly documented. Full correlation matrix available.

---

## Limitations

- Phase 2 controller is untrained (random weights); results reflect
  random attention modulation, not learned regulation.
- Training Stage 3 (attention controller) is defined but not yet run.
- 150-example development scale: no benchmark-wide generalization claims.
- BBH generalization not yet evaluated.

---

## Recommendation for Phase 3

1. Train the attention controller (Stage 3) using the L_attention loss.
2. Validate H3-H8 with trained controller.
3. Investigate state dimensionality reduction given high correlations.
4. Evaluate on ARC-Challenge and MMLU (generalization set).
5. Consider token-level compute allocation (Phase 3 scope).

---

## Reproducibility

```bash
# Smoke test
python mctr/scripts/run_phase2_smoke.py

# Pilot run
python mctr/scripts/run_phase2_pilot.py

# Full run
python mctr/scripts/run_phase2_full.py
```

Config: mctr/configs/phase2.yaml
Phase 1 baseline commit: a2e4bda
