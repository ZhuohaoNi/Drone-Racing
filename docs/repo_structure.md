# Project Repository Structure

This document outlines the main components and directory structure of the `ese651_project` repository.

## Overview
The repository is structured to separate the core environment definitions from the reinforcement learning algorithms and execution scripts.

## Directory Tree
```text
ese651_project/
├── config/                  # General project configuration files.
├── docs/                    # Documentation files (like this one and the implementation plan).
├── scripts/                 # Execution scripts for training and evaluating policies.
│   └── rsl_rl/              # Scripts specific to the rsl_rl library.
│       ├── train_race.py    # Main script to train the drone racing policy.
│       └── play_race.py     # Main script to evaluate/visualize the trained policy.
├── src/                     # Core source code for the project.
│   ├── isaac_quad_sim2real/ # Isaac Sim/Lab environment definitions.
│   │   └── tasks/
│   │       └── race/        # Specific code for the "Race" task.
│   │           └── config/crazyflie/
│   │               ├── agents/  # Hyperparameter configs (e.g., rsl_rl_ppo_cfg.py).
│   │               └── quadcopter_strategies.py # Contains get_rewards, get_observations, and reset_idx.
│   └── third_parties/       # External libraries.
│       └── rsl_rl_local/    # Local copy of the rsl_rl reinforcement learning library.
│           └── rsl_rl/
│               ├── algorithms/ # RL Algorithms (contains ppo.py where the PPO update is implemented).
│               ├── modules/    # Neural network modules (actor-critic networks).
│               └── storage/    # Rollout buffer implementation.
└── usd/                     # Universal Scene Description (USD) files for 3D assets and environments.
```

## Key Files to Focus On

1. **`src/third_parties/rsl_rl_local/rsl_rl/algorithms/ppo.py`**:
   - Contains the Proximal Policy Optimization algorithm. This is where the core PPO update step was implemented (Phase 1).

2. **`src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py`**:
   - Defines the learning strategy for the drone. You will work heavily here to define `get_rewards()`, `get_observations()`, and `reset_idx()` (Phase 2).

3. **`scripts/rsl_rl/train_race.py` & `play_race.py`**:
   - The main entry points to start training on the GPU and for viewing the drone racing with your trained policy.

4. **`src/isaac_quad_sim2real/tasks/race/config/crazyflie/agents/rsl_rl_ppo_cfg.py`**:
   - The configuration file for PPO hyperparameters (learning rate, entropy coefficient, clip parameter, etc.).
