# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESE 651 drone racing project using NVIDIA Isaac Lab. Trains a quadcopter policy via PPO to race through gate tracks in simulation. Uses a local fork of the `rsl_rl` library (not the pip version).

**Requirement:** The Isaac Lab installation must be at the same directory level as this repo.

## Commands

### Training
```bash
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 8192 \
    --max_iterations 5000 \
    --headless
```

### Evaluation
```bash
python scripts/rsl_rl/play_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 1 \
    --load_run [YYYY-MM-DD_XX-XX-XX] \
    --checkpoint best_model.pt \
    --headless \
    --video \
    --video_length 800
```
[text](docs)
Run logs are stored in `logs/rsl_rl/quadcopter_direct/<timestamp>_<run_name>/`.

### Environment
Conda environment: `isaac_quad_sim2real` (activated via `.envrc`)

## Architecture

### Entry Points
- `scripts/rsl_rl/train_race.py` — Training with Hydra config. Defines reward scales (progress, crash, death cost) and overrides env/algorithm params.
- `scripts/rsl_rl/play_race.py` — Evaluation/video recording. Exports trained policy as JIT and ONNX.
- `scripts/rsl_rl/cli_args.py` — CLI argument parsing (experiment name, resume, checkpoint, logger selection).

### Environment (`src/isaac_quad_sim2real/tasks/race/config/crazyflie/`)
- **`quadcopter_env.py`** — Core Isaac Lab environment (`QuadcopterEnv`/`QuadcopterEnvCfg`). Handles physics simulation (500Hz), PID angular rate control, motor dynamics, aerodynamic drag, contact-based crash detection. Policy runs at 50Hz (decimation=10). Action space: 4D (thrust, roll/pitch/yaw rates).
- **`quadcopter_strategies.py`** — Strategy pattern with `DefaultQuadcopterStrategy`. **Primary student implementation file.** Contains `get_rewards()`, `get_observations()`, and `reset_idx()` methods that define the learning task.
- **`agents/rsl_rl_ppo_cfg.py`** — PPO hyperparameters: actor (128-128), critic (512-256-128-128), ELU activation, adaptive LR schedule, γ=0.99, λ=0.95.

### RL Library (`src/third_parties/rsl_rl_local/rsl_rl/`)
- **`algorithms/ppo.py`** — PPO algorithm. **Contains a TODO block for the update step** that students must implement (policy loss with clipping, value loss, entropy loss, gradient step).
- **`storage/rollout_storage.py`** — Rollout buffer with GAE advantage computation.
- **`modules/actor_critic.py`** — Feed-forward and recurrent (LSTM) actor-critic networks.
- **`runners/on_policy_runner.py`** — Main training loop: collect rollouts → compute returns → PPO update → log/checkpoint.

### Task Registration
`Isaac-Quadcopter-Race-v0` is registered in `src/isaac_quad_sim2real/tasks/race/config/crazyflie/__init__.py`.

### Race Tracks
Three tracks defined in `quadcopter_env.py`: **powerloop** (default, 7 gates with vertical loop and chicane), **complex** (6 gates), **lemniscate** (6 gates). Gate assets are in `usd/`.

### Domain Randomization (Evaluation Only)
During eval, randomizes thrust-to-weight (0.95–1.05×), aerodynamic drag (0.5–2.0×), and PID gains (0.7–1.3×).

## Key Concepts

- **Strategy pattern**: The env delegates reward, observation, and reset logic to a strategy object, keeping core physics separate from the learning task definition.
- **Gate passage**: Tracked by drone position in gate-local frame. A gate is passed when `dist_to_gate < 0.1`. Same physical gate can serve as two waypoints with different pass directions (e.g., gates 3 & 6 on powerloop).
- **Crash detection**: Contact sensor on drone body; crash triggered when contact forces exceed threshold for 100 consecutive timesteps.
