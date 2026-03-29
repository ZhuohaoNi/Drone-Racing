#!/bin/bash

# Self-evaluation script: tests policy with default params + DR stability
# Usage: ./scripts/run/eval.sh 2026-03-29_12-00-00

set -e

RUN_DIR=${1:-"[YYYY-MM-DD_XX-XX-XX]"}

if [ "$RUN_DIR" == "[YYYY-MM-DD_XX-XX-XX]" ]; then
    echo "Warning: No run directory provided!"
    echo "Example: ./scripts/run/eval.sh 2026-02-15_12-30-00"
    exit 1
fi

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

# Change to the project root directory automatically
cd "$(dirname "$0")/../.."

# Export PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run two-phase evaluation:
#   Phase 1: Default params, 1 env (mimics TA conditions)
#   Phase 2: DR, 100 envs (stability testing)
python scripts/rsl_rl/eval_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --load_run "$RUN_DIR" \
    --checkpoint best_model.pt \
    --headless \
    --num_dr_envs 100 \
    --max_steps 2000
