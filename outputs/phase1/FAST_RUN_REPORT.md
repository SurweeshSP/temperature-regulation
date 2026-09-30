# MCTR Phase 1 Adaptive Stochastic Validation Report

## Experimental Configuration
- Dataset: GSM8K Main Test (n = 150)
- Sampling Seeds Evaluated: [42, 43, 44] (3 random seeds per question across all conditions)
- Total Trajectories Evaluated: 150 examples × 7 conditions × 3 seeds = 3150
- Decoding Mode: Adaptive Stochastic Sampling (`torch.multinomial`) vs Deterministic Greedy (`torch.argmax`)
- Model Backbone: Qwen/Qwen2.5-0.5B-Instruct (Fully Frozen)
- MCTR Temperature Bounds: [$T_{min} = 0.2, T_{max} = 1.2$]

---

## Key Technical Finding: Decoder Coupling
Under **Deterministic Greedy Decoding** (`argmax`), temperature scaling $\frac{z_t}{T_t}$ preserves logit ordinal ranking ($\arg\max (z_t / T) = \arg\max(z_t)$). Therefore, greedy accuracy is identical across all temperatures (32.67%).

Under **Adaptive Stochastic Sampling** (`multinomial`), temperature scaling directly participates in token selection ($T_t \rightarrow P_t \rightarrow y_t$), allowing accuracy differences and optimal regulation strategies to emerge.

---

## Accuracy & Calibration Results (Stochastic Sampling, Mean ± Std)

| Condition | Greedy Acc | Stochastic Accuracy (Mean ± Std) | ECE (Mean ± Std) | Brier (Mean ± Std) | Mean T |
|:---|:---:|:---:|:---:|:---:|:---:|
| Fixed-T-0.1 | 32.67% | 32.44% ± 1.26% | 0.6690 ± 0.0123 | 0.6626 ± 0.0120 | 0.1000 |
| Fixed-T-0.3 | 32.67% | 31.33% ± 3.03% | 0.6673 ± 0.0298 | 0.6488 ± 0.0292 | 0.3000 |
| Fixed-T-0.5 | 32.67% | 32.00% ± 3.03% | 0.6437 ± 0.0281 | 0.6106 ± 0.0260 | 0.5000 |
| Fixed-T-0.7 | 32.67% | 26.22% ± 1.57% | 0.6752 ± 0.0141 | 0.6214 ± 0.0128 | 0.7000 |
| Fixed-T-0.9 | 32.67% | 23.78% ± 1.26% | 0.6580 ± 0.0112 | 0.5754 ± 0.0096 | 0.9000 |
| Fixed-T-1.0 | 32.67% | 17.56% ± 2.45% | 0.6787 ± 0.0288 | 0.5702 ± 0.0287 | 1.0000 |
| MCTR-T | 32.67% | 26.00% ± 1.44% | 0.6806 ± 0.0124 | 0.6289 ± 0.0105 | 0.6676 |

---

## MCTR Dynamic Temperature Regulation Behavior
- Mean Temperature: 0.6676
- Mean Normalized Entropy: 0.0333
- Mean Predictive Confidence: 0.9222

---

## Efficiency
- Total Latency: 44450.22s
- Peak VRAM Utilization: 1013.71 MB

---

## Detailed Tables & Visualizations
- Main Comparison Table: `outputs/phase1/tables/table_fast_main_comparison.md`
- MCTR vs Baselines Table: `outputs/phase1/tables/table_mctr_vs_baseline.md`
- Baseline vs MCTR Plot: `outputs/phase1/plots/publication/Fig10_Baseline_vs_MCTR.png`
- Temperature Sweep Plot: `outputs/phase1/plots/publication/Fig09_FixedTemperature_Sweep.png`
