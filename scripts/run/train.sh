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

# Run the training script
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 8192  \
    --max_iterations 5000 \
    --headless \
    --logger wandb
