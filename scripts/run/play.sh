#!/bin/bash

# Ensure the script exits on errors
set -e

# Instructions:
# Replace the placeholder [YYYY-MM-DD_XX-XX-XX] with the actual run directory 
# string found in the logs directory before running this script.

RUN_DIR=${1:-"[YYYY-MM-DD_XX-XX-XX]"}

if [ "$RUN_DIR" == "[YYYY-MM-DD_XX-XX-XX]" ]; then
    echo "Warning: No run directory provided! Please pass your run directory as an argument."
    echo "Example: ./scripts/run/play.sh 2026-02-15_12-30-00"
    exit 1
fi

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

# Change to the project root directory automatically
cd "$(dirname "$0")/../.."

# Export PYTHONPATH to include the project root so Python can find the 'src' module
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the evaluation script
python scripts/rsl_rl/play_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 1 \
    --load_run "$RUN_DIR" \
    --checkpoint best_model.pt \
    --headless \
    --video \
    --video_length 1600 \
    --follow_robot 0
    
# # Run the evaluation script
# python scripts/rsl_rl/play_race.py \
#     --task Isaac-Quadcopter-Race-v0 \
#     --num_envs 1 \
#     --load_run "$RUN_DIR" \
#     --checkpoint best_model.pt \
#     --headless \
#     --video \
#     --video_length 1600 \
#     --follow_robot 0

# # Run the evaluation script
# python scripts/rsl_rl/play_race.py \
#     --task Isaac-Quadcopter-Race-v0 \
#     --num_envs 1 \
#     --load_run "$RUN_DIR" \
#     --checkpoint model_2650_29922.pt \
#     --headless \
#     --video \
#     --video_length 1600
