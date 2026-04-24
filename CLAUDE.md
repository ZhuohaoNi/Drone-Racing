# CLAUDE.md

This file gives coding agents a practical map of the repository so they can make changes without re-discovering the architecture every time.

## Purpose

This repo contains the Isaac Lab training and evaluation stack for quadcopter gate-racing with PPO.

At a high level:
- Isaac Lab simulates Crazyflie dynamics and gate tracks.
- `QuadcopterEnv` owns physics, PID rate control, motor dynamics, resets, and termination.
- A strategy object owns reward shaping, observation construction, and reset sampling policy.
- Training and evaluation scripts wrap the env with the local `rsl_rl` fork.
- Run scripts in `scripts/run/` are thin launchers that mostly inject JSON overrides through environment variables.

This is the training-side repo. Real-world deployment lives in the sibling `../sim2real/` repo.

## Environment Assumptions

- Isaac Lab must be installed and discoverable by the Python environment used to run this repo.
- Most project shell scripts activate the `ese6510` conda environment.
- The repo uses the local RL fork at `src/third_parties/rsl_rl_local/` instead of relying on a pip-installed `rsl_rl`.

## Top-Level Layout

- `src/isaac_quad_sim2real/`
  Main Python package for task registration and the race environment.
- `src/third_parties/rsl_rl_local/`
  Local fork of `rsl_rl` used by training, play, and evaluation scripts.
- `scripts/rsl_rl/`
  Python entry points for training, playback, evaluation, and robustness analysis.
- `scripts/run/`
  Bash and Python wrappers that assemble concrete experiments by setting `ENV_OVERRIDES`, `REWARD_OVERRIDES`, and `PPO_OVERRIDES`.
- `scripts/analysis/`
  Offline analysis helpers, especially for rosbag-derived metrics.
- `docs/`
  Notes, archived design docs, metric definitions, and local snapshots of related code.
- `usd/`
  Sim assets such as the Crazyflie and gate models.
- `logs/rsl_rl/`
  Training outputs, checkpoints, videos, exports, and evaluation artifacts.

## End-to-End Control Flow

Training path:
1. A launcher script in `scripts/run/` optionally activates conda and sets JSON overrides in environment variables.
2. `scripts/rsl_rl/train_race.py` loads the Isaac task config and PPO config.
3. `train_race.py` applies `REWARD_OVERRIDES`, `ENV_OVERRIDES`, and `PPO_OVERRIDES`.
4. Gym task registration resolves `Isaac-Quadcopter-Race-v0` to `QuadcopterEnv`.
5. `QuadcopterEnv` instantiates its configured strategy class.
6. The env is wrapped by `RslRlVecEnvWrapper`.
7. `rsl_rl.runners.OnPolicyRunner` performs rollout collection and PPO updates.
8. Logs, checkpoints, and config snapshots are written under `logs/rsl_rl/<experiment>/<timestamp>_<run_name>/`.

Evaluation path:
1. `play_race.py`, `eval_race.py`, `batch_eval_race.py`, or `paired_eval.py` load a saved checkpoint from `logs/rsl_rl/...`.
2. They build the same task with `env_cfg.is_train = False`.
3. Reward dictionaries are usually zeroed out because evaluation is about behavior, not optimization.
4. The env still uses the same observation pipeline and reset logic, but strategies switch to nominal dynamics instead of training-time randomization.

## Task Registration

Relevant files:
- `src/isaac_quad_sim2real/tasks/__init__.py`
- `src/isaac_quad_sim2real/tasks/race/config/crazyflie/__init__.py`

How it works:
- Importing `src.isaac_quad_sim2real.tasks` triggers package discovery via Isaac Lab utilities.
- `Isaac-Quadcopter-Race-v0` is registered in `tasks/race/config/crazyflie/__init__.py`.
- The registration points Gym at:
  - env entry point: `quadcopter_env.QuadcopterEnv`
  - env config entry point: `quadcopter_env.QuadcopterEnvCfg`
  - PPO config entry point: `agents.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg`

This means almost every script only needs `--task Isaac-Quadcopter-Race-v0`; the rest is resolved from registration.

## Core Environment Architecture

Main files:
- `src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_env.py`
- `src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py`

### `QuadcopterEnvCfg`

`QuadcopterEnvCfg` is the central configuration object for the task. It contains:
- Simulation rates and decimation.
- Robot physical constants.
- PID gains and thrust scaling.
- Domain-randomization knobs.
- Reset-mode knobs.
- Observation mismatch knobs.
- Track selection via `track_name`.
- Strategy selection via `strategy_class`.

Important defaults:
- `track_name = "circle"`
- `strategy_class = CircleQuadcopterStrategy`

`strategy_class` is the real behavior switch for reward/observation/reset logic. The class name `CircleQuadcopterStrategy` is historical; it is the default sim2real strategy for multiple tracks, not only the circle track.

### `QuadcopterEnv`

`QuadcopterEnv` subclasses `DirectRLEnv` and owns the simulation-side state:
- Scene setup and USD asset placement.
- Track waypoint definitions.
- Per-env tensors for actions, motor states, desired position, crash flags, gate indices, and replay buffers needed by the strategies.
- PID body-rate control and motor mixing.
- Action latency buffering.
- Done logic and contact-based termination.

Core methods:
- `_setup_scene()`
  Spawns the robot, terrain, contact sensors, and gate assets.
- `_pre_physics_step(actions)`
  Stores incoming policy actions and handles action latency.
- `_apply_action()`
  Runs the PID/motor pipeline and applies forces/torques.
- `_get_dones()`
  Returns timeout and termination masks.
- `_get_rewards()`
  Delegates to `self.strategy.get_rewards()`.
- `_reset_idx(env_ids)`
  Delegates reset sampling and state reset to `self.strategy.reset_idx(...)`.
- `_get_observations()`
  Delegates to `self.strategy.get_observations()`.

The env owns the physics. The strategy owns the task definition layered on top of those physics.

### Tracks and Loops

Waypoint loops are defined directly in `quadcopter_env.py`:
- `circle`
- `complex`
- `powerloop`
- `lemniscate`

`track_name` selects which waypoint list becomes `self._waypoints`.

Example:
- `scripts/run/train_powerloop_fullswitch_real_twr.sh` sets `"track_name":"powerloop"` in `ENV_OVERRIDES`.
- It does not override `strategy_class`.
- So it runs the `powerloop` loop with the default `CircleQuadcopterStrategy`.

## Strategy Architecture

`quadcopter_strategies.py` contains two strategy classes.

### `DefaultQuadcopterStrategy`

This is the older dense/speed-oriented strategy. It still exists in the codebase and is useful as a reference when comparing reward formulations or older experiments.

Responsibilities:
- Dense reward terms such as progress and velocity shaping.
- An older observation layout.
- A simpler reset path.
- Its own training/eval dynamics randomization behavior.

### `CircleQuadcopterStrategy`

This is the current sim2real-oriented strategy and the default selected by `QuadcopterEnvCfg`.

Responsibilities:
- Sparse or semi-sparse gate-centric reward logic.
- 40-dimensional observation construction.
- Gate replay reset sampling.
- Ground-start reset sampling.
- Optional staged replay reset scheduling.
- Training-time randomization of thrust, aero, PID, mass, and motor time constants.
- Evaluation-time nominal parameter setup.
- Robust gate-passing logic with approach checks and backtrack detection.

Important note:
- The class name is misleading.
- It is not tied to the `circle` loop.
- Treat it as the main sim2real race strategy unless the config explicitly chooses something else.

## Observations, Actions, Rewards, and Resets

### Observation Contract

The sim2real strategy builds a 40-dimensional observation:
- body linear velocity: 3
- rotation matrix: 9
- current gate corners in body frame: 12
- next gate corners in body frame: 12
- previous action: 4

This is a critical interface. If deployment compatibility matters, keep it aligned with the real controller implementation in the sibling `../sim2real/` repo. There is also a local snapshot/reference in `docs/controller_simple_policy.py`.

### Action Contract

The policy outputs 4 actions:
- normalized thrust
- roll rate command
- pitch rate command
- yaw rate command

The env converts these into:
- desired wrench
- motor-speed targets
- applied thrust and moments through the internal PID and motor model

### Reward Ownership

Rewards are not defined in `QuadcopterEnv` itself. They are owned by the strategy.

Training scripts build a reward dictionary and place it on `env_cfg.rewards`. Strategy code reads keys from that dictionary. Because several eval scripts hard-code zero reward dictionaries, adding a new reward key usually requires updating those eval scripts too.

### Reset Ownership

Reset sampling also belongs to the strategy.

The current sim2real strategy combines:
- direct gate-relative reset sampling
- replay resets from successful gate-pass states
- optional staged replay scheduling
- ground-start resets for takeoff coverage

## Override Mechanism

Most experiments are configured by environment variables rather than by editing the Python config classes directly.

Common override channels:
- `ENV_OVERRIDES`
- `REWARD_OVERRIDES`
- `PPO_OVERRIDES`

These are JSON objects parsed by `scripts/rsl_rl/train_race.py` and some evaluation scripts.

Typical use:
- `scripts/run/*.sh` exports one or more override blobs.
- The Python entry point loads the base config from task registration.
- The script mutates config attributes with `setattr` and updates reward dictionaries.

This is why many experiment scripts are small wrappers instead of separate Python programs.

## Training and Evaluation Scripts

### `scripts/rsl_rl/`

- `train_race.py`
  Main training entry point. Loads task and PPO config, applies overrides, builds the env, and launches PPO learning.
- `eval_race.py`
  Two-phase evaluation helper for default-parameter and DR-style checks.
- `batch_eval_race.py`
  Batch evaluation over multiple trials and parameter perturbations.
- `cli_args.py`
  Shared CLI argument helpers for the RSL-RL scripts.

### `scripts/run/`

These are experiment launchers, not core logic. They usually do some combination of:
- activate conda
- set `PYTHONPATH`
- build `ENV_OVERRIDES`, `REWARD_OVERRIDES`, `PPO_OVERRIDES`
- call one of the Python entry points

If behavior differs between two run scripts, the first place to look is the JSON override blobs they define.

### `scripts/analysis/`

- `bag_ordered_metrics.py`
  Offline bag-analysis helper for sim2real/system-identification style debugging.

## PPO and the Local RL Fork

The repo uses `src/third_parties/rsl_rl_local/`.

Files that matter most:
- `rsl_rl/runners/on_policy_runner.py`
  Rollout and training orchestration.
- `rsl_rl/algorithms/ppo.py`
  PPO update implementation.
- `rsl_rl/storage/rollout_storage.py`
  Buffer and GAE machinery.
- `rsl_rl/modules/actor_critic.py`
  Policy/value network definitions.

Task-side PPO config lives in:
- `src/isaac_quad_sim2real/tasks/race/config/crazyflie/agents/rsl_rl_ppo_cfg.py`

Current task defaults:
- actor hidden dims: `[512, 512, 256, 128]`
- critic hidden dims: `[512, 512, 256, 256, 128, 128]`
- activation: `elu`
- experiment name: `quadcopter_direct`

## Domain Randomization and Robustness Knobs

Most robustness knobs live on `QuadcopterEnvCfg` and are consumed by the active strategy.

Examples:
- thrust-to-weight randomization
- aerodynamic scale randomization
- PID gain randomization
- mass variation
- motor time-constant variation
- action latency
- observation delay
- observation noise and bias
- deterministic mismatch scales for evaluation

Training mode typically randomizes within configured ranges. Evaluation mode usually snaps to nominal parameters unless a script explicitly injects deterministic mismatch overrides.

## Instructions to follow for experiments
Every time we tune the reward or change the experimentation and get a version, we will record it in @docs/changelog.md. Also, when finding the best reward configuration, we need to do it in a systematic way, e.g. add a script or grid search to find the best hyperparameter

## Editing Rules That Matter in This Repo

- When changing reward keys, check all evaluation scripts with hard-coded reward dictionaries.
- When changing observation layout or actor dimensions, consider compatibility with the deployment controller.
- When investigating experiment differences, inspect the override blobs in `scripts/run/*.sh` before changing Python logic.
- Do not assume `CircleQuadcopterStrategy` means the `circle` track; track selection is separate.
- The authoritative runtime behavior comes from the Python code in `src/` and `scripts/rsl_rl/`, not from old archived docs in `docs/archive/`.

## Useful Reference Files

- `src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_env.py`
- `src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py`
- `src/isaac_quad_sim2real/tasks/race/config/crazyflie/agents/rsl_rl_ppo_cfg.py`
- `scripts/rsl_rl/train_race.py`
- `scripts/rsl_rl/eval_race.py`
- `scripts/run/train_powerloop_fullswitch_real_twr.sh`
- `docs/changelog.md`
