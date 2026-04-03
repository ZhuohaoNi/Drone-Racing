#!/bin/bash

# Fine-tune V26 best model with wider DR + lower entropy
# Strategy: "over-prepare" with harder DR so TA's ranges feel easy
# Changes from V26 training:
#   - DR ranges widened (TWR ±8%, Aero 0.3-2.5x, PID kp/ki ±25%, kd ±40%)
#   - entropy_coef: 0.005 → 0.001 (more deterministic)
#   - Ground spawn: 15% + 10% ≈ 24% (moderate)

set -e

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

# Change to the project root directory automatically
cd "$(dirname "$0")/../.."

# Export PYTHONPATH to include the project root so Python can find the 'src' module
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run fine-tuning from original V26 checkpoint (not the first finetune)
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 16384 \
    --max_iterations 2000 \
    --headless \
    --logger wandb \
    --resume True \
    --load_run best_v26 \
    --checkpoint best_model.pt \
    --run_name finetune_v26_robust
