# Design: PPO and Racing Strategy

This document covers the two core implementation areas of the drone racing project. For each area it follows the same structure: (A) what to implement per the project requirements, (B) what the current implementation does, (C) how to evaluate it, and (D) what can be improved.

---

## Part 1: Proximal Policy Optimization (PPO)

### 1.1 What to Implement (Project Requirements)

The project description (Section 2) requires two deliverables inside the local `rsl_rl` library:

1. **`update()` in `ppo.py`** (required) — The PPO gradient step that trains the actor-critic networks. Must implement:
   - Re-evaluate the current policy on stored transitions (forward pass through actor and critic)
   - Clipped surrogate policy loss
   - Value function loss (optionally clipped)
   - Entropy bonus
   - Combined loss, backpropagation, and optimizer step with gradient clipping

2. **`compute_returns()` in `rollout_storage.py`** (optional) — Generalized Advantage Estimation (GAE). Must compute:
   - TD errors (`delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)`)
   - Advantages via backward cumulative sum (`A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}`)
   - Returns as critic training targets (`returns_t = A_t + V(s_t)`)

PPO sits inside the `OnPolicyRunner` training loop:

```
for each iteration:
    1. Collect rollouts  — 8192 drones x 24 steps at 50 Hz   (act / process_env_step)
    2. Compute returns   — GAE advantage estimation            (compute_returns)
    3. PPO update        — 5 epochs x 4 mini-batches           (update)
    4. Checkpoint        — save model if reward improved
```

PPO trains two networks:
- **Actor** (policy): `[obs_dim] -> 128 -> 128 -> [4]` with ELU + final Tanh. Outputs mean of a Gaussian; actions sampled with learnable std (`init_noise_std=1.0`).
- **Critic** (value function): `[obs_dim] -> 512 -> 256 -> 128 -> 128 -> [1]` with ELU. Predicts expected cumulative reward.

#### How the `update()` method works step-by-step

**Step 1 — Mini-batch generation.** The rollout buffer holds `8192 * 24 = 196,608` transitions. `mini_batch_generator` shuffles and yields 4 chunks, repeated for 5 epochs (20 total gradient steps per iteration). Each mini-batch provides:

| Variable | Shape | Description |
|---|---|---|
| `observations` | (batch, obs_dim) | States seen by the actor |
| `critic_observations` | (batch, obs_dim) | States seen by the critic |
| `sampled_actions` | (batch, 4) | Actions that were taken during rollout |
| `value_targets` | (batch, 1) | Critic's prediction at collection time |
| `advantage_estimates` | (batch, 1) | GAE advantages |
| `discounted_returns` | (batch, 1) | Critic training target (advantage + value) |
| `prev_log_probs` | (batch, 1) | Log-prob of action under the old policy |

**Step 2 — Forward pass.** Re-evaluate the current (possibly updated) networks on the batch:
1. `update_distribution(observations)` — run actor to get new action distribution
2. `get_actions_log_prob(sampled_actions)` — log-prob of old actions under new distribution
3. `evaluate(critic_observations)` — new value estimates from critic

**Step 3 — Losses.**

Surrogate policy loss (clipped):
```
ratio = exp(new_log_prob - old_log_prob)
surrogate_1 = ratio * advantage
surrogate_2 = clamp(ratio, 1-eps, 1+eps) * advantage
policy_loss = -min(surrogate_1, surrogate_2).mean()
```
The clipping (`eps=0.2`) bounds the policy change per update. The negative sign converts maximization to minimization.

Value loss (clipped):
```
value_clipped = old_value + clamp(new_value - old_value, -eps, +eps)
value_loss = max((new_value - returns)^2, (value_clipped - returns)^2).mean()
```
Prevents the critic from overshooting and destabilizing advantage estimation.

Entropy loss:
```
entropy_loss = -entropy_coef * entropy.mean()
```
Encourages exploration by penalizing low-entropy (overly confident) distributions.

Total: `loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy`

**Step 4 — Gradient step.** `zero_grad() -> backward() -> clip_grad_norm_(max=1.0) -> step()`

**Step 5 — Adaptive LR (post-update).** Compute mean KL divergence across all mini-batches. If KL > 2 * `desired_kl`: reduce LR (policy changing too fast). If KL < `desired_kl` / 2: increase LR (could learn faster). Clamped to [1e-5, 1e-2].

#### How GAE works

```
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)     # TD error
A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}         # advantage (backward pass)
return_t = A_t + V(s_t)                                           # critic target
```

- `lam=0.95` balances bias-variance: `lam=1` gives high-variance Monte Carlo; `lam=0` gives biased single-step TD.
- The backward cumulative sum has a temporal dependency, but with only 24 steps the loop is cheap. The parallelizable part is computing all `delta_t` at once.

### 1.2 Current Implementation

Based on the changelog and code inspection, the teammate has completed both deliverables:

#### PPO `update()` (`ppo.py:123-215`) — Complete

**Forward pass (lines 152-156):**
```python
self.actor_critic.update_distribution(observations)
actions_log_prob_batch = self.actor_critic.get_actions_log_prob(sampled_actions)
value_batch = self.actor_critic.evaluate(critic_observations)
actions_log_prob_batch = actions_log_prob_batch.view(-1, 1)
```

**KL divergence tracking (lines 158-162):**
```python
with torch.no_grad():
    log_ratio = actions_log_prob_batch - prev_log_probs
    kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
    mean_kl += kl.item()
```
Uses the numerically stable approximation `KL approx E[(exp(r) - 1) - r]` where `r = log(pi_new / pi_old)`. Computed under `no_grad()` since it only feeds the adaptive LR schedule, not backprop.

**Surrogate loss (lines 164-168):** Textbook PPO-Clip. `min` selects the more pessimistic surrogate; negation converts to minimization.

**Value loss (lines 170-177):** Clipped value loss enabled by default. Old predictions clamped within `clip_param`, then `max` of clipped/unclipped MSE.

**Entropy and total loss (lines 179-184):** Entropy subtracted to encourage exploration. Currently `entropy_coef=0.0`, so this term is inactive.

**Gradient step (lines 186-194):** Standard `zero_grad -> backward -> clip_grad_norm(1.0) -> step`.

**Adaptive LR (lines 203-210):**
```python
if mean_kl > self.desired_kl * 2.0:
    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
elif mean_kl < self.desired_kl / 2.0:
    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
```
Uses a 1.5x factor (upstream `rsl_rl` uses 2x). This is a deliberate choice for smoother adjustment.

#### GAE `compute_returns()` (`rollout_storage.py:132-158`) — Complete, GPU-optimized

```python
# Vectorized: all TD errors in one operation
next_values = torch.cat([self.values[1:], last_values.unsqueeze(0)])
next_is_not_terminal = 1.0 - self.dones.float()
deltas = self.rewards + next_is_not_terminal * gamma * next_values - self.values

# Sequential backward pass (24 iterations, each over 8192 envs in parallel)
advantage = torch.zeros_like(last_values)
for step in reversed(range(self.num_transitions_per_env)):
    advantage = deltas[step] + next_is_not_terminal[step] * gamma * lam * advantage
    self.returns[step] = advantage + self.values[step]

self.advantages = self.returns - self.values
```

The delta computation is fully vectorized instead of being inside the loop. With `num_transitions_per_env=24` and `num_envs=8192`, each loop iteration processes 8192 environments in parallel, so the 24-step loop is fast.

Advantage normalization (`(adv - mean) / (std + 1e-8)`) applied when `normalize_advantage=True` (default), stabilizing training regardless of reward scale.

#### Hyperparameters (`rsl_rl_ppo_cfg.py`)

| Parameter | Value | Role |
|---|---|---|
| Actor | [128, 128] ELU + Tanh | Policy network |
| Critic | [512, 256, 128, 128] ELU | Value network |
| `clip_param` | 0.2 | Surrogate/value clipping |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `desired_kl` | 0.01 | Adaptive LR target |
| `entropy_coef` | 0.0 | Entropy bonus (disabled) |
| `num_steps_per_env` | 24 | Rollout length |
| `learning_rate` | 5e-4 | Initial LR |
| `num_learning_epochs` | 5 | Epochs per update |
| `num_mini_batches` | 4 | Mini-batches per epoch |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `init_noise_std` | 1.0 | Initial action noise |
| `min_std` | 0.0 | Minimum action std |

### 1.3 How to Evaluate PPO (WandB Metrics)

#### Verification: PPO correctness check

Per the project description, after implementing PPO with the default (hovering) strategy, the drone should hover near gate 0 and the WandB reward curve should match the reference figure. This confirms the `update()` and `compute_returns()` are working before moving to the strategy.

#### PPO-specific metrics

| Metric | Healthy Range | What it Means |
|---|---|---|
| **Surrogate loss** (policy loss) | Small, stable (-0.05 to 0.05) | How much the policy changes per update. Large oscillations = instability. |
| **Value loss** | Decreasing over training | Critic prediction error. Persistently high = critic can't learn the value function. |
| **Entropy** | Gradually decreasing, never 0 | Action randomness. Collapse to 0 = premature convergence. |
| **KL divergence** | Near `desired_kl=0.01` | Effective policy change rate. Spikes = update too aggressive. |
| **Learning rate** | Adjusting within [1e-5, 1e-2] | Stuck at 1e-5 = unstable. Stuck at 1e-2 = initial LR too low. |
| **Mean reward** | Increasing | Primary signal that learning is working. |
| **Mean episode length** | Context-dependent | Decreasing = faster laps (or more crashes — cross-reference with death rate). |

#### Diagnosing PPO problems

| WandB Pattern | Likely Cause | Fix |
|---|---|---|
| Value loss stays high or increases | Reward scale too large/variable, critic too small, LR too high | Scale rewards, increase critic size, reduce LR |
| KL spikes repeatedly | Updates too aggressive, LR at minimum | Reduce `clip_param` (0.2 -> 0.1), widen adaptive band |
| Entropy collapses to ~0 early | Policy locked in before sufficient exploration | Increase `entropy_coef`, increase `init_noise_std` |
| Surrogate loss oscillates | Noisy advantage estimates | Increase `num_steps_per_env`, lower `lam` |
| Reward plateaus after initial rise | Local optimum | Check entropy; if nonzero, issue is strategy not PPO |

### 1.4 PPO Improvements

#### 1.4.1 High: Enable Entropy Bonus

Currently `entropy_coef = 0.0` (`rsl_rl_ppo_cfg.py:30`). No exploration incentive — the policy can collapse to deterministic behavior before discovering the full racing trajectory.

**Recommended:** `entropy_coef = 0.005`. Small enough to not destabilize, sufficient to maintain exploration through the powerloop and chicane.

#### 1.4.2 Medium: Increase Rollout Length

`num_steps_per_env = 24` covers 0.48s of flight at 50Hz. A gate-to-gate segment takes 1-3s. Short rollouts mean GAE can't observe complete gate passages, making advantage estimates noisier.

**Consider:** `num_steps_per_env = 48` (0.96s). Better return estimation at the cost of more memory and slightly slower iterations.

#### 1.4.3 Medium: Tune Adaptive LR Factor

The 1.5x factor is more conservative than the upstream 2x. If WandB shows the LR oscillating frequently, consider:
- Widening the dead band (adjust only if KL > 3x or < 1/3x `desired_kl`)
- Or reducing factor to 1.25x for even smoother control

#### 1.4.4 Low: Per-Mini-Batch Advantage Normalization

`normalize_advantage_per_mini_batch = False` — advantages normalized once over the full batch. When transitions span diverse track sections (high-reward gate passes vs. low-reward cruising), per-mini-batch normalization ensures balanced gradients.

#### 1.4.5 Low: Actor Network Sizing

Actor [128, 128] is relatively small. If the policy plateaus with healthy entropy, consider [256, 128, 64] for more capacity. The critic [512, 256, 128, 128] is already large.

#### PPO Improvement Priority Summary

| Priority | Change | Impact | Effort |
|---|---|---|---|
| High | `entropy_coef = 0.005` | Prevents premature convergence | Trivial |
| Medium | `num_steps_per_env = 48` | Better advantage estimation | Trivial |
| Medium | Tune adaptive LR band/factor | Smoother LR control | Low |
| Low | Per-mini-batch advantage normalization | More balanced gradients | Trivial |
| Low | Actor network sizing [256, 128, 64] | Higher policy capacity | Low |

---

## Part 2: Racing Strategy (Rewards, Observations, Resets)

### 2.1 What to Implement (Project Requirements)

The project description (Section 3) requires completing three methods in `quadcopter_strategies.py`. The provided starter code produces only a hover-to-gate-0 policy and "requires significant rework to produce a strong drone racing policy."

1. **`get_rewards()`** — Reward structure that encourages racing through gates with minimal lap time:
   - Detect when a gate is successfully traversed
   - Calculate progress metrics rewarding forward movement through the course
   - Detect and penalize crashes using contact sensor data
   - Combine components with reward scales from `train_race.py`

2. **`get_observations()`** — Observation space providing sufficient info for navigation:
   - Extract drone state (position, orientation, velocity)
   - Handle frame transformations carefully (world vs. body vs. gate-relative)
   - Concatenate into a single observation vector

3. **`reset_idx()`** — Reset strategy for initial drone states:
   - Define positions relative to waypoints/gates
   - Set initial orientations
   - Add randomization for robustness

Additionally, the project encourages tuning hyperparameters in `rsl_rl_ppo_cfg.py`.

#### Evaluation criteria (Section 3.1)

- **Primary metric:** 3-lap completion time on the powerloop track
- **Sim2real gap:** TAs alter dynamics during evaluation — TWR (+/-5%), aero drag (0.5-2x), PID gains (roll/pitch kp/ki +/-15%, kd +/-30%; yaw same ranges)
- Domain randomization and adaptation are explicitly recommended

### 2.2 Current Implementation

Based on the changelog (V1) and code inspection, the teammate has implemented all three methods:

#### 2.2.1 Reward Structure (`get_rewards()`, lines 68-139)

Three reward components plus a death cost:

**Gate passing detection (lines 75-103):**
Tracks the drone's x-coordinate in the gate-local frame. A gate is passed when x transitions from positive to negative (crossed forward through the gate plane) AND the drone's y/z coordinates are within `gate_side / 2` (within the gate opening). On passage:
- Waypoint index advances: `idx_wp = (idx_wp + 1) % num_gates`
- Gate counter increments
- Target position updates to the new gate

**Progress reward (lines 105-112):**
```python
current_distance = ||desired_pos_w - root_link_pos_w||   # full 3D distance
progress = (last_distance - current_distance).clamp(-1.0, 1.0)
```
Dense signal rewarding distance reduction toward the current target gate.

**Crash detection (lines 114-118):**
```python
crashed = (||contact_forces|| > 1e-8).int()
mask = (episode_length_buf > 100).int()    # 100-step grace period
_crashed += crashed * mask                  # accumulates; dies at _crashed > 100
```

**Reward scales (from `train_race.py`):**

| Component | Scale | Type |
|---|---|---|
| `progress_goal` | 50.0 | Dense, per-timestep |
| `gate_pass` | 200.0 | Sparse, on gate passage |
| `crash` | -1.0 | Penalty per contact timestep |
| `death_cost` | -10.0 | One-time on termination |

On termination, the entire timestep reward is replaced with `death_cost`.

#### 2.2.2 Observation Space (`get_observations()`, lines 141-195)

23-dimensional ego-centric vector:

| Observation | Dims | Frame | Source |
|---|---|---|---|
| Linear velocity | 3 | Body | `root_com_lin_vel_b` |
| Angular velocity | 3 | Body | `root_ang_vel_b` |
| Orientation quaternion | 4 | World | `root_quat_w` |
| Current gate position | 3 | Body | `subtract_frame_transforms(drone_pos, drone_quat, gate_pos)` |
| Next gate position | 3 | Body | Same transform for `(idx_wp + 1) % num_gates` |
| Current gate normal | 3 | World | `_normal_vectors[current_gate_idx]` |
| Previous actions | 4 | N/A | `_previous_actions` |

Design choices: body-frame velocities match the body-relative control inputs; gate positions in body frame avoid needing global coordinates; next gate enables planning ahead; previous actions account for action smoothing.

**Known issue:** Gate normal is in world frame while gate positions are in body frame. The policy must implicitly learn the rotation to use the normal.

#### 2.2.3 Reset Strategy (`reset_idx()`, lines 197-353)

**Training mode:**
1. Random gate: uniform over `[0, num_gates)`
2. Position: 1.5-3.0m behind gate (gate-local -x), +/-0.5m lateral, +/-0.3m vertical (clamped above 0.15m)
3. Orientation: facing target gate with +/-0.15 rad (~8.6 deg) yaw noise
4. Velocity: zero

**Play mode:** Fixed spawn near gate 0 at z=0.05m with random lateral/longitudinal offsets.

#### 2.2.4 Physics Parameters (`__init__`, lines 45-66)

All parameters set to **fixed nominal values** — no domain randomization:
```python
_K_aero[:, :2] = k_aero_xy_value       # fixed drag
_kp_omega[:, :2] = kp_omega_rp_value   # fixed PID gains
_thrust_to_weight[:] = twr_value        # fixed TWR
_tau_m[:] = tau_m_value                 # fixed motor time constant
```

### 2.3 How to Evaluate the Strategy (WandB Metrics)

#### Reward curves

| Metric | What it tells you | Healthy sign |
|---|---|---|
| `Episode_Reward/progress_goal` | Is the drone approaching gates? | Positive, growing |
| `Episode_Reward/gate_pass` | Gates passed per episode | Increasing in discrete steps |
| `Episode_Reward/crash` | Collision frequency | Decreasing toward zero |
| Total episode reward | Overall learning progress | Steady upward trend |

#### Termination stats

| Metric | What it tells you |
|---|---|
| `Episode_Termination/died` | Crash/boundary deaths per batch — should decrease |
| `Episode_Termination/time_out` | Episodes hitting 30s limit — should decrease as drone gets faster |

#### Diagnosing strategy problems

| WandB Pattern | Likely Cause | Fix |
|---|---|---|
| Reward up but `gate_pass` flat | Reward hacking — oscillating near gate without passing | Increase `gate_pass_reward_scale`, decrease `progress_goal_reward_scale` |
| High death rate throughout | Drone too aggressive or unstable | Increase `crash_reward_scale` magnitude, add orientation penalty |
| `gate_pass` rises then drops | Catastrophic forgetting | Reduce LR, increase `clip_param` |
| Plateau with healthy entropy | Strategy ceiling — PPO is fine but reward/obs limit what's learnable | Add speed reward, improve observations |

#### Evaluation-time testing

After training, run `play_race.py` and check:
- Does the drone complete 3 laps on the powerloop track?
- What is the total lap time?
- Does it survive under domain randomization (altered TWR, drag, PID gains)?

If the policy fails under randomized dynamics, domain randomization must be added to training.

### 2.4 Strategy Improvements

#### 2.4.1 Critical: Domain Randomization

The TAs will randomize dynamics during evaluation per the exact ranges in Section 3.1 of the project description. The current implementation uses fixed nominal values. Without training-time randomization, the policy will be brittle.

**Change needed in `reset_idx()` for each resetting environment:**

```python
# TWR: +/-5%
self.env._thrust_to_weight[env_ids] = uniform(twr * 0.95, twr * 1.05)

# Aero drag: 0.5-2.0x
self.env._K_aero[env_ids, :2] = uniform(k_xy * 0.5, k_xy * 2.0)
self.env._K_aero[env_ids, 2]  = uniform(k_z  * 0.5, k_z  * 2.0)

# PID gains — roll/pitch
self.env._kp_omega[env_ids, :2] = uniform(kp_rp * 0.85, kp_rp * 1.15)
self.env._ki_omega[env_ids, :2] = uniform(ki_rp * 0.85, ki_rp * 1.15)
self.env._kd_omega[env_ids, :2] = uniform(kd_rp * 0.7,  kd_rp * 1.3)

# PID gains — yaw
self.env._kp_omega[env_ids, 2] = uniform(kp_y * 0.85, kp_y * 1.15)
self.env._ki_omega[env_ids, 2] = uniform(ki_y * 0.85, ki_y * 1.15)
self.env._kd_omega[env_ids, 2] = uniform(kd_y * 0.7,  kd_y * 1.3)
```

#### 2.4.2 High: Speed Reward

The competition metric is lap time, but the current reward has no explicit speed incentive. The policy can score well by slowly drifting toward gates.

```python
to_gate = desired_pos_w - root_link_pos_w
to_gate_dir = to_gate / (to_gate.norm(dim=1, keepdim=True) + 1e-6)
speed_toward_gate = (root_com_lin_vel_w * to_gate_dir).sum(dim=1)
speed_reward = speed_toward_gate.clamp(0.0, max_speed)
```

Scale suggestion: `speed_reward_scale = 5.0-15.0`.

#### 2.4.3 High: Gate Normal in Body Frame

The gate normal is in world frame (line 173) while gate positions are in body frame — a frame inconsistency.

```python
rot_matrix = matrix_from_quat(drone_quat_w)
gate_normal_b = torch.bmm(rot_matrix.transpose(1, 2), gate_normal_w.unsqueeze(-1)).squeeze(-1)
```

#### 2.4.4 High: Gravity Vector in Body Frame

Replace or supplement the world-frame quaternion (4 dims) with gravity in body frame (3 dims). Directly tells the policy "which way is down" without requiring quaternion decoding:

```python
gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(num_envs, 3)
rot_matrix = matrix_from_quat(drone_quat_w)
gravity_b = torch.bmm(rot_matrix.transpose(1, 2), gravity_world.unsqueeze(-1)).squeeze(-1)
```

#### 2.4.5 Medium: Non-Zero Initial Velocity

The drone always spawns stationary, but in a race it's always in motion. Training from static starts means the policy never practices flying through gates at speed.

```python
forward_speed = torch.empty(n_reset, device=self.device).uniform_(0.0, 3.0)
default_root_state[:, 7] = forward_speed * torch.cos(initial_yaw)
default_root_state[:, 8] = forward_speed * torch.sin(initial_yaw)
```

#### 2.4.6 Medium: Orientation Alignment Reward

Reward the drone for facing the gate approach direction:

```python
alignment = (drone_forward_b @ to_gate_dir).clamp(-1.0, 1.0)
alignment_reward = alignment * alignment_reward_scale
```

#### 2.4.7 Medium: Additional Look-Ahead Gate

The powerloop (gates 2-3) and chicane (gates 5-6-0) benefit from seeing further ahead. Add a third look-ahead gate (+3 dims):

```python
third_gate_idx = (current_gate_idx + 2) % num_gates
third_gate_pos_b, _ = subtract_frame_transforms(drone_pos, drone_quat, waypoints[third_gate_idx, :3])
```

#### 2.4.8 Low: Weighted Gate Sampling

Over-sample harder gates during reset:

```python
weights = torch.ones(num_gates, device=self.device)
weights[[2, 3, 5, 6, 0]] = 2.0  # powerloop and chicane gates
waypoint_indices = torch.multinomial(weights.expand(n_reset, -1), 1).squeeze(1)
```

#### 2.4.9 Low: Action Smoothness Penalty

```python
action_diff = self.env._actions - self.env._previous_actions
smoothness_penalty = action_diff.norm(dim=1) * smoothness_penalty_scale  # e.g. -0.1
```

#### Strategy Improvement Priority Summary

| Priority | Change | Impact | Effort |
|---|---|---|---|
| Critical | Domain randomization (all eval ranges) | Policy survives eval dynamics | Low |
| High | Speed reward toward gate | Faster lap times | Low |
| High | Gate normal in body frame | Faster learning, consistent frames | Low |
| High | Gravity vector in body frame | Better orientation signal | Low |
| Medium | Non-zero initial velocity | Trains in-flight behavior | Low |
| Medium | Orientation alignment reward | Cleaner gate approaches | Low |
| Medium | Third look-ahead gate | Better planning for powerloop/chicane | Low |
| Low | Weighted gate sampling | More practice on hard gates | Low |
| Low | Action smoothness penalty | Smoother, more efficient flight | Low |

---

## Overall

Both PPO and the strategy must work together:

- **PPO** determines how efficiently the policy learns from the reward signal.
- **The strategy** determines what the policy learns and how robust it is to eval-time dynamics.

A well-tuned PPO with a bad strategy converges quickly to a bad policy. A good strategy with a buggy PPO doesn't converge at all.

**Primary evaluation metric:** 3-lap completion time on the powerloop track under randomized dynamics.
