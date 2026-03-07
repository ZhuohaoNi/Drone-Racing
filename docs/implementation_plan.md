# Drone Race Project

## Goal Description
Implement Proximal Policy Optimization (PPO) and design a learning strategy (rewards, observations, resets) for a drone racing environment using NVIDIA Isaac Lab. The objective is to train a quadcopter to navigate a race track containing powerloops and chicanes while being robust to domain randomization.

## Proposed Changes

### 1. PPO Algorithm Implementation
Implement the core PPO update step and optionally optimize advantage computation.

#### [MODIFY] [ppo.py](file:///home/peterni/Documents/ese6510/ese651_project/src/third_parties/rsl_rl_local/rsl_rl/algorithms/ppo.py)
- Complete the `#TODO` in the `update()` method to perform the PPO actor-critic network updates, calculate surrogate loss, value loss, and entropy bonus.

#### [MODIFY] [rollout_storage.py](file:///home/peterni/Documents/ese6510/ese651_project/src/third_parties/rsl_rl_local/rsl_rl/storage/rollout_storage.py)
- (Optional) Explore and implement Generalized Advantage Estimation (GAE) at `compute_returns()`.

### 2. Environment Strategy Implementation
Design the observation space, reward structure, and reset strategy for the drone racing task.

#### [MODIFY] [quadcopter_strategies.py](file:///home/peterni/Documents/ese6510/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py)
- **`get_rewards()`**: Implement logic to detect gate passes, calculate progress metrics, penalize crashes, and combine components into a per-timestep reward.
- **`get_observations()`**: Extract drone state (position, orientation, velocity) and target gate information, carefully handling coordinate frames (world, body, or gate-relative).
- **`reset_idx()`**: Define initial states (position and orientation) when an episode resets, adding appropriate domain randomization to initial states for robustness.

#### [MODIFY] [rsl_rl_ppo_cfg.py](file:///home/peterni/Documents/ese6510/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/agents/rsl_rl_ppo_cfg.py)
- Tune hyperparameters for the PPO agent tailored to the custom reward and observation scales.

### 3. Domain Randomization (For Sim2Real Gap)
Ensure the policy is robust to aerodynamic and mass variations introduced in evaluation.

#### [MODIFY] Environment Config files (if applicable)
- Add or modify domain randomization settings for drone mass (`twr`), aerodynamics (`k_aero_xy`, `k_aero_z`), and PID gains.

## Verification Plan

### Automated Tests
- Run typical PPO training script: 
  `python scripts/rsl_rl/train_race.py --task Isaac-Quadcopter-Race-v0 --num_envs 8192 --max_iterations 1000 --headless --logger wandb`
- Initially, visually confirm the policy learns to hover near the zeroth gate (as per handout) to verify PPO correctness, then train the full racing setup.
- Monitor Weights & Biases (wandb) for increasing episode rewards and sensible loss metrics.
- Evaluate the trained policy with video rendering:
  `python scripts/rsl_rl/play_race.py --task Isaac-Quadcopter-Race-v0 --num_envs 1 --load_run [RUN_DIR] --checkpoint best_model.pt --headless --video --video_length 800`
