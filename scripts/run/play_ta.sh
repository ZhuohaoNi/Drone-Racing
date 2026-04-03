#!/bin/bash

# Ensure the script exits on errors
set -e

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

# Change to the project root directory automatically
cd "$(dirname "$0")/../.."

# Export PYTHONPATH to include the project root so Python can find the 'src' module
export PYTHONPATH="$(pwd):$PYTHONPATH"

# First argument is seed, default is 42 if not provided
SEED=${1:-42}

# Run the TA-style randomized eval script
python scripts/rsl_rl/play_ta_style.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 1 \
    --experiment_name best_v26 \
    --checkpoint best_model.pt \
    --seed $SEED \
    --video_length 2000 \
    --headless
