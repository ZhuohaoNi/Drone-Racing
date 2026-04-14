# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESE 651 drone racing project using NVIDIA Isaac Lab. Trains a quadcopter policy via PPO to race through gate tracks in simulation. Uses a local fork of the `rsl_rl` library (not the pip version).

**Requirement:** The Isaac Lab installation must be at the same directory level as this repo.

## Current Phase: Sim-to-Real (Phase II)

**Goal:** Deploy trained policies on real Crazyflie drones at Pennovation. Priority is **robustness and lap completion**, not speed. Zero tolerance for crashes.

**Two-repo workflow:** This repo (`Drone-Racing/`) is the **training environment**. The sibling repo (`../sim2real/`) is the **real-world deployment** codebase (ROS2, Vicon, Crazyradio). Both repos are needed throughout.

**The training→deployment chain:**
```
Drone-Racing/                              sim2real/
quadcopter_strategies.py                   controller_simple_policy.py
  get_observations() → 40-dim obs    ←→     update() → 40-dim obs (must match exactly)
  get_rewards() shapes behavior              (no rewards in real)
  Train → export .pt checkpoint        →     Load .pt → run on drone
```

**Critical constraint:** The observation space and network architecture in this repo must **exactly match** `sim2real/src/controller/controller/controller_simple_policy.py`. The real controller uses:
- **Obs (40-dim):** body linear velocity (3) + rotation matrix (9) + current gate 4 corners in body frame (12) + next gate 4 corners in body frame (12) + previous action (4)
- **Actor:** `[512, 512, 256, 128]` with ELU activation, Tanh output
- **Actions (4-dim):** thrust, roll rate, pitch rate, yaw rate

**Stages:**
1. Circle track (4 gates) — zero-shot transfer, then iterate
2. Circle track — optimize for 3 reliable laps
3. Powerloop track (7 gates) — after circle is robust
4. Final race: 2026-04-28

**Sim2real plan:** See `../sim2real/docs/sim2real_plan.md` for the full strategy (reward structure, DR ranges, observation design, deployment pipeline).

### Sim2Real Strategy vs V30 Speed Strategy

The V30 strategy (`DefaultQuadcopterStrategy`) was optimized for speed (97.3% SR, 15.68s mean 3-lap). For sim2real, we use `CircleQuadcopterStrategy` that:
- Uses a sparse reward formulation (mostly `gate_pass` driving the behavior, no continuous `progress` shaping)
- Uses the 40-dim gate-corner + previous action observation space (matching the real controller)
- Heavily penalizes crashes and jerky commands (`cmd_reg` on body rates)
- Targets ~8-12s per lap (slow is fine)
- Adds Swift-inspired observation noise and action latency (`action_latency_max`)
- Mixes ground takeoffs and gate-replay initializations for deployment realism

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
Conda environment: `lab` (activated via `.envrc`)

## Architecture

### Entry Points
- `scripts/rsl_rl/train_race.py` — Training with Hydra config. Defines base sparse reward scales (`gate_pass`, `death_cost`, `cmd_reg`) and parses `REWARD_OVERRIDES`, `ENV_OVERRIDES`, and `PPO_OVERRIDES` from the system environment for hyperparameter sweeps.
- `scripts/rsl_rl/play_race.py` — Evaluation/video recording. Exports trained policy as JIT and ONNX.
- `scripts/rsl_rl/cli_args.py` — CLI argument parsing (experiment name, resume, checkpoint, logger selection).
- `scripts/run/sweep.py` — Executes automated parameter tuning grid search by setting environmental overrides.

### Environment (`src/isaac_quad_sim2real/tasks/race/config/crazyflie/`)
- **`quadcopter_env.py`** — Core Isaac Lab environment (`QuadcopterEnv`/`QuadcopterEnvCfg`). Handles physics simulation (500Hz), PID angular rate control, motor dynamics, aerodynamic drag, contact-based crash detection. Action latency configurable via `action_latency_max`.
- **`quadcopter_strategies.py`** — Strategy pattern handling curriculum logic. `DefaultQuadcopterStrategy` for the speed tasks and `CircleQuadcopterStrategy` for the Sim2Real sparse and robust deployment.
- **`agents/rsl_rl_ppo_cfg.py`** — PPO hyperparameters: actor (128-128), critic (512-256-128-128), ELU activation, adaptive LR schedule, γ=0.99, λ=0.95.

### RL Library (`src/third_parties/rsl_rl_local/rsl_rl/`)
- **`algorithms/ppo.py`** — PPO algorithm. **Contains a TODO block for the update step** that students must implement (policy loss with clipping, value loss, entropy loss, gradient step).
- **`storage/rollout_storage.py`** — Rollout buffer with GAE advantage computation.
- **`modules/actor_critic.py`** — Feed-forward and recurrent (LSTM) actor-critic networks.
- **`runners/on_policy_runner.py`** — Main training loop: collect rollouts → compute returns → PPO update → log/checkpoint.

### Task Registration
`Isaac-Quadcopter-Race-v0` is registered in `src/isaac_quad_sim2real/tasks/race/config/crazyflie/__init__.py`.

### Race Tracks
Four tracks defined in `quadcopter_env.py`: **powerloop** (default, 7 gates with vertical loop and chicane), **complex** (6 gates), **lemniscate** (6 gates), **circle** (4 gates, used for sim2real). Gate assets are in `usd/`.

For sim2real work, use `track_name = 'circle'` in `QuadcopterEnvCfg`. The circle track matches the physical gate layout at Pennovation (waypoints in `../sim2real/src/jirl_bringup/config/config.yaml`).

### Domain Randomization (Evaluation Only)
During eval, randomizes thrust-to-weight (0.95–1.05×), aerodynamic drag (0.5–2.0×), and PID gains (0.7–1.3×).

## Key Concepts

- **Strategy pattern**: The env delegates reward, observation, and reset logic to a strategy object, keeping core physics separate from the learning task definition.
- **Gate passage**: Tracked by drone position in gate-local frame. A gate is passed when `dist_to_gate < 0.1`. Same physical gate can serve as two waypoints with different pass directions (e.g., gates 3 & 6 on powerloop).
- **Crash detection**: Contact sensor on drone body; crash triggered when contact forces exceed threshold for 100 consecutive timesteps.

## Instructions to follow for experiments
Every time we tune the reward or change the experimentation and get a version, we will record it in @docs/changelog.md. Also, when finding the best reward configuration, we need to do it in a systematic way, e.g. add a script or grid search to find the best hyperparameter. Also we should record the metrics/observation, and coding agent should provide guidance on how to interpret the metrics in @doc/metrics.md (create one if it's not here).

**IMPORTANT: Modifying Rewards**
If you add or modify a new reward term (e.g., in `quadcopter_strategies.py`), you **MUST** update all related evaluation scripts to include the new reward key with a dummy `0.0` value in their manually hard-coded reward dictionaries. Failure to do so will cause a `KeyError` during validation. The related files include:
- `scripts/rsl_rl/eval_race.py`
- `scripts/rsl_rl/batch_eval_race.py`
- (And potentially `scripts/rsl_rl/paired_eval.py`)