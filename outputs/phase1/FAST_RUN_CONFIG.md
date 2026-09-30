# FAST RUN CONFIG
```json
{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "dataset": "gsm8k_main_test",
    "seeds_evaluated": [
        42,
        43,
        44
    ],
    "device": "cuda",
    "timestamp": "2026-09-19T02:23:42.822317",
    "n_examples": 150,
    "total_trajectories": 3150,
    "decoding_modes": [
        "Deterministic Greedy (Track 1)",
        "Stochastic Sampling (Track 2)"
    ],
    "temperature_bounds": [
        0.2,
        1.2
    ],
    "conditions": [
        "Fixed-T-0.1",
        "Fixed-T-0.3",
        "Fixed-T-0.5",
        "Fixed-T-0.7",
        "Fixed-T-0.9",
        "Fixed-T-1.0",
        "MCTR-T"
    ]
}
```