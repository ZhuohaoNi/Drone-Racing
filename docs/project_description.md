# Drone Race Project Instructions

**ESE 6510: Physical Intelligence**  
**February 2026**

## 1 Introduction

In this project, you will use NVIDIA Isaac Lab to train a drone racing policy! You will write the PPO algorithm into the `rsl_rl` learning library, construct an observation space, shape rewards, and define an episode reset strategy.

*Figure 1: Powerloop race track and example raceline in blue. The green arrows indicate the gate-passing directions and green numbers indicate the gate-passing sequence.*

We have designed a brand-new Powerloop race track this semester! A powerloop is a maneuver where a drone performs a vertical loop in order to enter two adjacent gates from the same side, as illustrated with gates 2 and 3. Although this maneuver is not necessary to complete the track, executing it well will significantly improve your lap time. A chicane is a high-speed dash through offset gates that require subtle but rapidly alternating turns, as illustrated with gates 5, 6, and 0. Notice that gate 3 and gate 6 are physically the same object, but represent a different gate-passing direction in the track sequence.

### 1.1 Accessing Project Repo

The project repository contains the code for the drone racing environment, training, and evaluation. The repo also includes a custom copy of the `rsl_rl` robot learning library where you will implement Proximal Policy Optimization.

Enter your home directory, then git clone the project repository. It is critical that the project repo and the Isaac Lab directory are at the same level.

```bash
git clone git@github.com:Jirl-upenn/ese651_project.git
```

### 1.2 Running the Code

In order to train, we can call the following command from the terminal:

```bash
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 8192 \
    --max_iterations 1000 \
    --headless \
    --logger wandb
```

In order to play, we can call the following example command from the terminal:

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

> **Note:** Neither command will work until PPO is implemented as per the next section.

## 2 PPO

For this project, you will use a local copy of the `rsl_rl` robot learning library, an open-source reinforcement learning library optimized for GPU-based training. You will find our class local copy at `src/third_parties/rsl_rl_local`. Before you begin, you should create a Weights and Biases (wandb) account. This is a free cloud platform to monitor your neural network training, and you will rely heavily on it to gain insight on your drone's performance.

1. Your objective is to implement the Proximal Policy Optimization (PPO) RL algorithm by writing the `update()` method marked as `#TODO` in `src/third_parties/rsl_rl_local/rsl_rl/algorithms/ppo.py`.

2. Additionally, consider exploring different ways to compute the advantage at `compute_returns()` in `src/third_parties/rsl_rl_local/rsl_rl/storage/rollout_storage.py`. *(Optional)*

> **Important:** Since our local `rsl_rl` library is modified, directly copying from the current `rsl_rl` repository on Github will not work (and is a breach of academic integrity). Reading the `rsl_rl` repository to understand how PPO is implemented is okay, and can be useful to learn best practices for writing GPU-optimized code.

Once your PPO implementation is complete, you can immediately run a training with the command from Sec 1.2. This will produce a policy where the drone hovers near the zeroth gate. You should be able to match the final timestep of `best_policy.pt` and your wandb training logs with the following, respectively:

*Figure 2: After implementing PPO, your first trained policy should produce a drone hovering near the zeroth gate without triggering the gate-pass condition.*

*Figure 3: The episode reward plots in wandb should appear as such when training the hovering drone policy.*

## 3 Strategy

In this section, you will complete the code marked as `#TODO` in `src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py` in order to implement a reward structure in `get_rewards()`, observation space in `get_observations()`, and drone reset strategy in `reset_idx()` for effective training. All three methods contain example code that will run and produce a simple hovering-to-zeroth-gate policy. They require significant rework to produce a strong drone racing policy. You are encouraged to further tune hyperparameters in `src/isaac_quad_sim2real/tasks/race/config/crazyflie/agents/rsl_rl_ppo_cfg.py`.

1. Design and implement a reward structure in `get_rewards()` that encourages the drone to race through gates with minimal lap time. Your implementation should:
   * Implement logic to detect when a gate is successfully traversed
   * Calculate meaningful progress metrics that reward forward movement through the course
   * Detect and penalize crashes using contact sensor data
   * Compute per-timestep rewards by multiplying your reward components with the corresponding reward scales defined in `train_race.py`

   > **Note:** The provided example code only produces a policy that hovers near the zeroth gate and does not race. You must significantly modify or replace this code!

2. Create an observation space `get_observations()` that provides the policy with sufficient information for navigation and control. Your implementation should:
   * Extract relevant drone state information from the simulation
   * Be careful with frame transformations: decide whether observations should be in world, body, or gate-relative frames
   * Concatenate all observation tensors into a single observation vector

3. Implement a reset strategy `reset_idx()` that determines initial drone states when episodes begin. Your implementation should:
   * Define initial drone positions relative to waypoints/gates
   * Set appropriate initial orientations
   * Consider adding randomization to initial states to improve policy robustness

## 3.1 Evaluation

Your drone racing policy will be evaluated through time-trials on the same track that you trained on. The primary metric is time to complete 3 laps. You can view your policy's performance on a live leaderboard which we will release on Ed. 

In order to mimic the sim2real gap, the TAs will alter the drone dynamics in `quadcopter_env.py` so a selection of the following parameters will be sampled from anywhere within these ranges:

**TWR**
```python
self._twr_min = self.cfg.thrust_to_weight * 0.95
self._twr_max = self.cfg.thrust_to_weight * 1.05
```

**Aerodynamics**
```python
self._k_aero_xy_min = self.cfg.k_aero_xy * 0.5
self._k_aero_xy_max = self.cfg.k_aero_xy * 2.0
self._k_aero_z_min = self.cfg.k_aero_z * 0.5
self._k_aero_z_max = self.cfg.k_aero_z * 2.0
```

**PID gains**
```python
self._kp_omega_rp_min = self.cfg.kp_omega_rp * 0.85
self._kp_omega_rp_max = self.cfg.kp_omega_rp * 1.15
self._ki_omega_rp_min = self.cfg.ki_omega_rp * 0.85
self._ki_omega_rp_max = self.cfg.ki_omega_rp * 1.15
self._kd_omega_rp_min = self.cfg.kd_omega_rp * 0.7
self._kd_omega_rp_max = self.cfg.kd_omega_rp * 1.3
self._kp_omega_y_min = self.cfg.kp_omega_y * 0.85
self._kp_omega_y_max = self.cfg.kp_omega_y * 1.15
self._ki_omega_y_min = self.cfg.ki_omega_y * 0.85
self._ki_omega_y_max = self.cfg.ki_omega_y * 1.15
self._kd_omega_y_min = self.cfg.kd_omega_y * 0.7
self._kd_omega_y_max = self.cfg.kd_omega_y * 1.3
```

Consider strategies like domain randomization and adaptation so your policy can still succeed despite a dynamics mismatch. The evaluation environment will be held constant for all students.

There are only two differences between your eval and the TAs' eval: 

The initial pose: When you run play_race.py yourself, you execute the code within if not self.cfg.is_train. That code uniformly samples x_local and y_local within (-3.0, -0.5) and (-1.0, 1.0) respectively. When the TA's run play_race.py, we will hard code a value for x_local and y_local. This hard coded value will be within the same bounds (-3.0, -0.5) and (-1.0, 1.0).

The TA's eval code will randomly sample 3 parameters from the list of parameters in section 3.1 of the project handout. They will be sampled only once within the bounds shown in the handout, and will be held consistent for all students.

I highly recommend that you "self-evaluate" with various combinations of randomly selected parameters. If your policy performs well on 5 such combinations, it is very likely robust enough for the TAs' evaluation environment.

You will separately submit a 1-2 page write-up documenting what was implemented and how your racing strategy was developed.


In the Canvas code submission, you will upload the following files:

src/third_parties/rsl_rl_local/rsl_rl/algorithms/ppo.py

src/third_parties/rsl_rl_local/rsl_rl/storage/rollout_storage.py (If edited the advantage computation)

src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py

a .zip of your policy directory, containing the params folder as well as ONLY ONE .pt file (your best one). This directory already exists as logs/rsl_rl/quadcopter_direct/2025-MM-DD_HH-MM-SS

Recognize that you cannot submit quadcopter_env.py! If your policy during inference is dependent on changes to that code, your policy will break or fail to execute when the TAs run it. Consider backing up your work and doing a fresh git pull of the project repo to ensure your policy folder and quadcopter_strategies.py can run before submitting.

