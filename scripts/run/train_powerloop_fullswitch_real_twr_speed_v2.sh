#!/bin/bash

# Powerloop fullswitch speed-v2 candidate on real-effective TWR.
#
# This is the conservative replacement for the broad reward-pressure speed
# candidate. It keeps the validated TWR-1.87 fullswitch base reward unchanged
# and adds only a small racing-line velocity reward after the loop.
#
# Why:
#   - real bag analysis showed fullswitch mainly loses time after the loop:
#     4->5, 5->6, 6->0
#   - old powerloop V19/V26 improved speed with a current/next-gate velocity
#     blend, but aggressive global velocity/heading rewards caused crashes
#   - this version limits the shaping to idx_wp 4,5,6,0 and skips the initial
#     takeoff-to-gate0 segment
#
# Usage:
#   ./scripts/run/train_powerloop_fullswitch_real_twr_speed_v2.sh [max_iterations] [num_envs]

set -euo pipefail

MAX_ITERS=${1:-3000}
NUM_ENVS=${2:-8192}

source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate ese6510

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

REWARD_OVERRIDES='{"progress_goal_reward_scale":20.0,"lap_complete_reward_scale":0.5,"progress_skip_gate_indices":[3],"vel_toward_gate_reward_scale":0.75,"vel_reward_gate_indices":[4,5,6,0],"vel_reward_min_gates_passed":1,"vel_reward_next_weight":0.375,"vel_reward_clamp_min":-2.0,"vel_reward_clamp_max":8.0}'
ENV_OVERRIDES='{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,"mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,"motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,"replay_reset_ratio":0.25,"ground_reset_ratio":0.10,"staged_replay_reset":true,"replay_warmup_iterations":500,"replay_full_iterations":2000,"aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,"pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30}'

echo "========================================"
echo "  Powerloop Training (fullswitch + real-effective TWR + speed-v2)"
echo "  Iterations: $MAX_ITERS"
echo "  Envs:       $NUM_ENVS"
echo "  Reward:     $REWARD_OVERRIDES"
echo "  Env:        $ENV_OVERRIDES"
echo "  Speed-v2:   post-loop vel reward only; base progress/time/control penalties unchanged"
echo "========================================"

REWARD_OVERRIDES="$REWARD_OVERRIDES" \
ENV_OVERRIDES="$ENV_OVERRIDES" \
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --headless \
    --logger wandb \
    --run_name "powerloop-r1d1-gate3mask-fullswitch-twr1p87-speedv2-postloopvel"
