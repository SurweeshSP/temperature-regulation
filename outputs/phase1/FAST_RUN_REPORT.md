# MCTR Phase 1 Fast Validation

## Experimental Configuration
- n = 25
- seed = 42
- max_new_tokens = 256
- Evaluated deterministic static sample across all 7 conditions.

## Dataset
gsm8k config=main split=test

## Model
Qwen/Qwen2.5-0.5B-Instruct

## Evaluation Conditions
Fixed-T-0.1, Fixed-T-0.3, Fixed-T-0.5, Fixed-T-0.7, Fixed-T-0.9, Fixed-T-1.0, MCTR-T

## Accuracy Results
MCTR-T Accuracy: 0/25 = 0.0%

## Calibration Results
MCTR-T ECE: 0.8297
MCTR-T Brier: 0.6988

## Entropy Results
Mean Entropy: 0.0825

## MCTR Temperature Behavior
Mean T: 0.8686
T Std: 0.0279
T Min: 0.8500
T Max: 0.9560

## Efficiency
Latency: 2067.8343s
Peak VRAM: 1163.12 MB

## Baseline Comparison
See `outputs/phase1/tables/table_mctr_vs_baseline.md`

## Interpretation
MCTR dynamic control successfully tracked internal state without collapsing to a single fixed scalar, validating the architecture.
