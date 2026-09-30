"""
MCTR Phase 2: Full Run
=======================
Full experiment with GSM8K + BBH tasks.
After pilot is validated, runs generalization benchmarks.
"""

import os
import sys
import argparse

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import pilot logic and extend
from mctr.scripts.run_phase2_pilot import run_pilot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTR Phase 2 Full Run")
    parser.add_argument("--config", default="mctr/configs/phase2.yaml")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Override max examples (default: all from dataset)")
    args = parser.parse_args()

    print("[FULL RUN] Running Phase 2 full experiment (may take several hours on GPU)")
    run_pilot(args.config, smoke_test=False, max_examples=args.max_examples)
