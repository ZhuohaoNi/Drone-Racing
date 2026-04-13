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

Based on the changelog and code inspection, both PPO deliverables are complete and stable since V1.

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
| `empirical_normalization` | True | Running obs normalization |

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

**Caution:** V2 achieved the best results (3 laps, ~17s) without entropy bonus. Only enable if training shows premature convergence (entropy collapse on WandB). May not be needed.

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
- 3 parameters randomly sampled once from those ranges, held constant for all students
- Initial pose: x_local in (-3.0, -0.5), y_local in (-1.0, 1.0) — hard-coded by TAs within those bounds
- Domain randomization and adaptation are explicitly recommended

### 2.2 Current Implementation (V9 = V2 Revert)

The current code matches V2 exactly. V3-V8 experimented with modifications to the reward structure, observation normalization, and orientation penalty — all degraded performance relative to V2 and were reverted. See Section 2.5 for lessons learned.

**Best recorded performance (V2):** 3 laps completed in ~17s with `empirical_normalization = True`.

#### 2.2.1 Reward Structure (`get_rewards()`, lines 85-161)

Six reward components plus a death cost:

**Gate passing detection (lines 92-107):**
Tracks the drone's x-coordinate in the gate-local frame. A gate is passed when x transitions from positive to negative (crossed forward through the gate plane) AND the drone's y/z coordinates are within `gate_side / 2` (within the gate opening). On passage:
- Waypoint index advances: `idx_wp = (idx_wp + 1) % num_gates`
- Gate counter increments
- Target position updates to the new gate

**Progress reward (lines 109-114):**
```python
current_distance = ||desired_pos_w - root_link_pos_w||   # full 3D distance
progress = (last_distance - current_distance).clamp(-1.0, 1.0)
```
Dense signal rewarding distance reduction toward the current target gate.

**Velocity toward gate reward (lines 116-121):**
```python
direction_to_gate = (desired_pos_w - root_link_pos_w) / (||...|| + 1e-8)
vel_toward_gate = dot(drone_vel_w, direction_to_gate).clamp(-2.0, 5.0)
```
Dot product of world-frame velocity with unit direction to gate. Clamped to [-2, 5] to cap outliers while still penalizing moving away from the gate.

**Orientation penalty (lines 123-127):**
```python
tilt_penalty = (|roll| + |pitch|)   # only when > 0.5 rad (~30 deg)
```
Penalizes excessive tilt angles. Threshold at 0.5 rad so normal flight is unpenalized but extreme orientations (pre-crash) are discouraged.

**Smoothness penalty (lines 129-131):**
```python
smoothness_penalty = ||actions - previous_actions||
```
Penalizes action jitter (large frame-to-frame changes).

**Crash detection (lines 133-137):**
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
| `vel_toward_gate` | 5.0 | Dense, per-timestep |
| `orientation` | -2.0 | Penalty when tilt > 0.5 rad |
| `smoothness` | -0.5 | Penalty per-timestep |
| `crash` | -1.0 | Penalty per contact timestep |
| `death_cost` | -10.0 | One-time on termination |

On termination, the entire timestep reward is replaced with `death_cost`.

#### 2.2.2 Observation Space (`get_observations()`, lines 163-226)

25-dimensional ego-centric vector:

| Observation | Dims | Frame | Source |
|---|---|---|---|
| Linear velocity | 3 | Body | `root_com_lin_vel_b` |
| Angular velocity | 3 | Body | `root_ang_vel_b` |
| Orientation quaternion | 4 | World | `root_quat_w` |
| Current gate position | 3 | Body | `subtract_frame_transforms(drone_pos, drone_quat, gate_pos)` |
| Next gate position | 3 | Body | Same transform for `(idx_wp + 1) % num_gates` |
| Current gate normal | 3 | World | `_normal_vectors[current_gate_idx]` |
| Sin/cos yaw error | 2 | N/A | `sin/cos(wrap_to_pi(gate_yaw - drone_yaw))` |
| Previous actions | 4 | N/A | `_previous_actions` |

The sin/cos yaw error provides a continuous, wrap-around-safe heading alignment signal without the discontinuity at +/-pi.

**Known issue:** Gate normal is in world frame while gate positions are in body frame — a frame inconsistency.

#### 2.2.3 Reset Strategy (`reset_idx()`, lines 228-387)

**Training mode:**
1. Random gate: uniform over `[0, num_gates)`
2. **Domain randomization** on every reset via `_randomize_dynamics(env_ids)`
3. Position: 1.5-3.0m behind gate (gate-local -x), +/-0.5m lateral, +/-0.3m vertical (clamped above 0.15m)
4. Orientation: facing target gate with +/-0.15 rad (~8.6 deg) yaw noise
5. Velocity: zero

**Play mode:** Fixed spawn near gate 0 at z=0.05m with random lateral/longitudinal offsets within (-3.0, -0.5) x (-1.0, 1.0).

#### 2.2.4 Domain Randomization (`_randomize_dynamics()`, lines 49-83)

Physics parameters randomized per-environment on every reset, matching TA evaluation ranges:

| Parameter | Range | Notes |
|---|---|---|
| TWR | ±5% | Thrust-to-weight ratio |
| Aero drag x/y | 0.5x - 2.0x | Sampled independently |
| Aero drag z | 0.5x - 2.0x | Independent from x/y |
| PID kp/ki (roll/pitch) | ±15% | Roll = pitch (physical symmetry) |
| PID kd (roll/pitch) | ±30% | Roll = pitch |
| PID kp/ki (yaw) | ±15% | Independent |
| PID kd (yaw) | ±30% | Independent |
| Motor time constant | Fixed | Not randomized |

#### 2.2.5 PPO Configuration

- `empirical_normalization = True` — Running normalization of observations using mean/std statistics. Essential for V2-level performance; V7's experiment confirmed that disabling it collapses all reward metrics. V8 showed that manual normalization (dividing by fixed constants) cannot replicate the benefit of running statistics.

**Important caveat:** `rsl_rl_ppo_cfg.py` is NOT submitted to TAs. The normalizer state must be saved in the checkpoint for inference to work correctly. Need to verify the TA's runner loads normalizer weights from the `.pt` file automatically.

### 2.3 How to Evaluate the Strategy (WandB Metrics)

#### V2 baseline metrics (best known)

| Metric | V2 Value | Notes |
|---|---|---|
| `vel_toward_gate` | ~800 | Strong speed signal |
| `progress_goal` | ~100-125 | Consistent approach |
| `gate_pass` | ~200 (saturated) | Completing all gates |
| `orientation` | ~-60 | Large but drone still raced well |
| `smoothness` | ~0 | Naturally smooth actions |
| `crash` | -0.2 to -0.8 | Minimal contact |
| 3-lap time | ~17s | Primary evaluation metric |

#### Reward curves

| Metric | What it tells you | Healthy sign |
|---|---|---|
| `Episode_Reward/progress_goal` | Is the drone approaching gates? | Positive, growing |
| `Episode_Reward/gate_pass` | Gates passed per episode | Increasing in discrete steps |
| `Episode_Reward/vel_toward_gate` | Speed toward the current gate | Positive and growing |
| `Episode_Reward/orientation` | Tilt penalty magnitude | Small/near-zero |
| `Episode_Reward/smoothness` | Action jitter penalty | Decreasing |
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
| High death rate throughout | Drone too aggressive or unstable | Increase crash/death penalty scales |
| `gate_pass` rises then drops | Catastrophic forgetting | Reduce LR, reduce `clip_param` |
| Plateau with healthy entropy | Strategy ceiling — PPO is fine but reward/obs limit what's learnable | Add look-ahead gate, fix gate normal frame |
| `vel_toward_gate` positive but `gate_pass` flat | Drone flying toward gate but not through it | Check gate normal alignment — may need body-frame normal |
| `orientation` large and persistent | Drone frequently over-tilting | Increase `orientation_reward_scale` magnitude or lower tilt threshold |
| `smoothness` stays high | Jerky control not converging | Increase `smoothness_reward_scale` magnitude |

#### Evaluation-time testing

After training, run `play_race.py` and check:
- Does the drone complete 3 laps on the powerloop track?
- What is the total lap time?
- Does it survive under domain randomization (altered TWR, drag, PID gains)?

Recommended self-evaluation: test with 5 random combinations of 3 altered parameters within TA ranges.

### 2.4 Strategy Improvements

Items marked **DONE** are already implemented. Items marked **TRIED & REVERTED** were tested in V3-V8 and degraded performance. Remaining items are prioritized for the next iteration.

#### 2.4.1 ~~Critical: Domain Randomization~~ — DONE (V2)

Implemented in `_randomize_dynamics()` (lines 49-83). TWR ±5%, aero drag 0.5-2x, PID gains ±15%/±30%. Called on every `reset_idx()`. Matches TA evaluation ranges exactly.

#### 2.4.2 ~~High: Speed Reward~~ — DONE (V2)

Implemented as `vel_toward_gate` reward (lines 116-121). Dot product of velocity with direction to gate, clamped [-2, 5], scale=5.0.

**Warning from V4-V6:** Do NOT increase this scale. At scale 5.0 combined with heading reward (V4), the velocity contribution dominated at ~750, drowning out gate_pass (200). The drone became speed-obsessed and crashed at tight turns (gate 3->4). Even at scale 2.0 (V5) it still crashed frequently. Scale 5.0 alone (without heading reward) is the validated sweet spot.

#### 2.4.3 ~~Medium: Orientation Penalty~~ — DONE (V2)

Implemented as tilt penalty (lines 123-127). Penalizes |roll| + |pitch| when > 0.5 rad, scale=-2.0.

**Warning from V3:** Reducing to -0.5 with threshold 1.0 rad caused the drone to flip at the gate 3->4 transition. The V2 values (-2.0 scale, 0.5 rad threshold) are critical for stability through tight turns. Do not weaken.

#### 2.4.4 ~~Low: Action Smoothness Penalty~~ — DONE (V2)

Implemented (lines 129-131). Penalizes ||actions - previous_actions||, scale=-0.5. V2 showed the policy naturally learns smooth actions (metric ~0), so the penalty is lightweight.

#### 2.4.5 High: Gate Normal in Body Frame

**Still needed.** The gate normal is in world frame (line ~195) while gate positions are in body frame — a frame inconsistency. The sin/cos yaw error partially addresses heading alignment, but the full 3D normal in body frame would help with the powerloop's vertical gates.

```python
rot_matrix = matrix_from_quat(drone_quat_w)
gate_normal_b = torch.bmm(rot_matrix.transpose(1, 2), gate_normal_w.unsqueeze(-1)).squeeze(-1)
```

#### 2.4.6 High: Gravity Vector in Body Frame

**Still needed.** Replace or supplement the world-frame quaternion (4 dims) with gravity in body frame (3 dims). Directly tells the policy "which way is down" without requiring quaternion decoding:

```python
gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(num_envs, 3)
rot_matrix = matrix_from_quat(drone_quat_w)
gravity_b = torch.bmm(rot_matrix.transpose(1, 2), gravity_world.unsqueeze(-1)).squeeze(-1)
```

This saves 1 obs dimension (4 -> 3) while providing a more directly useful signal.

#### 2.4.7 Medium: Non-Zero Initial Velocity

**Still needed.** The drone always spawns stationary, but in a race it's always in motion. Training from static starts means the policy never practices flying through gates at speed.

```python
forward_speed = torch.empty(n_reset, device=self.device).uniform_(0.0, 3.0)
default_root_state[:, 7] = forward_speed * torch.cos(initial_yaw)
default_root_state[:, 8] = forward_speed * torch.sin(initial_yaw)
```

#### 2.4.8 Medium: Additional Look-Ahead Gate

**Still needed.** The powerloop (gates 2-3) and chicane (gates 5-6-0) benefit from seeing further ahead. Add a third look-ahead gate (+3 dims):

```python
third_gate_idx = (current_gate_idx + 2) % num_gates
third_gate_pos_b, _ = subtract_frame_transforms(drone_pos, drone_quat, waypoints[third_gate_idx, :3])
```

This is especially important if `gate_pass` plateaus at 3-5 gates.

#### 2.4.9 Low: Weighted Gate Sampling

**Still needed.** Over-sample harder gates during reset:

```python
weights = torch.ones(num_gates, device=self.device)
weights[[2, 3, 5, 6, 0]] = 2.0  # powerloop and chicane gates
waypoint_indices = torch.multinomial(weights.expand(n_reset, -1), 1).squeeze(1)
```

#### 2.4.10 TRIED & REVERTED: Heading Alignment Reward (V4-V6)

Tested `cos(yaw_error) * 2.0` as a reward in V4. Combined with vel_toward_gate, it over-constrained the policy and caused constant crashing (passes 1-3 gates then dies). Reduced to scale 1.0 (V5) and 0.5 (V6) — still degraded performance. The sin/cos yaw error in the observation space is sufficient directional guidance; an explicit heading reward is harmful.

#### 2.4.11 TRIED & REVERTED: Manual Observation Normalization (V8)

Tried dividing velocities by 3.0, angular velocities by 5.0, gate positions by 5.0 in `get_observations()` to replicate `empirical_normalization` within the submitted file. Performance was worse (vel: 800->200, progress: 100->30, gate_pass: 200->50 at V7 level). Running mean/std normalization cannot be replicated by fixed constants because the distribution shifts during training.

#### 2.4.12 TRIED & REVERTED: Higher Death Cost (V6)

Increased death_cost from -10 to -50. Made the drone overly cautious — 3 laps in ~26s vs V2's ~17s. The -10 value provides sufficient crash avoidance without sacrificing speed.

#### Strategy Improvement Priority Summary

| Priority | Change | Status | Impact |
|---|---|---|---|
| ~~Critical~~ | ~~Domain randomization~~ | **DONE** | ~~Policy survives eval dynamics~~ |
| ~~High~~ | ~~Speed reward (vel_toward_gate)~~ | **DONE** | ~~Faster lap times~~ |
| ~~Medium~~ | ~~Orientation penalty~~ | **DONE** | ~~Reduces crashes~~ |
| ~~Low~~ | ~~Action smoothness penalty~~ | **DONE** | ~~Smoother flight~~ |
| High | Gate normal in body frame | **TODO** | Consistent obs frames |
| High | Gravity vector in body frame | **TODO** | Better orientation signal, -1 dim |
| Medium | Non-zero initial velocity | **TODO** | Trains in-flight behavior |
| Medium | Third look-ahead gate | **TODO** | Better planning for powerloop/chicane |
| Low | Weighted gate sampling | **TODO** | More practice on hard gates |
| N/A | Heading alignment reward | **REVERTED** | Harmful — over-constrains policy |
| N/A | Manual obs normalization | **REVERTED** | Cannot replicate running stats |
| N/A | Higher death cost (-50) | **REVERTED** | Makes drone too cautious |

**Next priorities:** Gate normal in body frame (2.4.5) and gravity vector (2.4.6) are the highest-impact remaining changes. Both are observation-space improvements that can be done together without touching the reward structure that V2 validated.

### 2.5 Lessons Learned from V3-V8 Experiments

These experiments collectively demonstrate important principles for reward/strategy tuning:

1. **Reward scale interactions are non-linear.** vel_toward_gate at scale 5.0 works alone, but combined with heading reward the velocity contribution (~750) drowns out gate_pass (200). Adding rewards is not additive — always check the magnitude of each component on WandB.

2. **Orientation penalty is load-bearing.** The -2.0 scale at 0.5 rad threshold is critical. Reducing to -0.5 (V3) caused flips; the drone needs strong tilt penalty to survive the gate 3->4 tight turn. This is not just a "nice to have" — it's structural.

3. **`empirical_normalization` is essential and cannot be manually replicated.** Running mean/std normalization adapts as the observation distribution shifts during training. Fixed divisors (V8) cannot match this. Since `rsl_rl_ppo_cfg.py` isn't submitted, the normalizer state must be saved in and loaded from the checkpoint.

4. **Penalty magnitude affects speed.** death_cost=-50 (V6) made the drone too cautious (~26s vs ~17s). Penalizing crashes is necessary, but excessive penalty teaches the policy to avoid risk rather than fly fast.

5. **Don't fix what works.** V2's reward structure is validated at ~17s/3-laps. Future improvements should focus on observations and resets (which don't risk destabilizing the reward signal) rather than reward tuning.

---

## Overall

Both PPO and the strategy must work together:

- **PPO** determines how efficiently the policy learns from the reward signal.
- **The strategy** determines what the policy learns and how robust it is to eval-time dynamics.

A well-tuned PPO with a bad strategy converges quickly to a bad policy. A good strategy with a buggy PPO doesn't converge at all.

**Current status:** V2/V9 achieves 3 laps in ~17s. PPO is stable and complete. The reward structure is validated — further speed improvements should come from observation-space enhancements (body-frame gate normals, gravity vector) and reset improvements (non-zero initial velocity), not reward re-tuning.

**Primary evaluation metric:** 3-lap completion time on the powerloop track under randomized dynamics.

**Key risk:** `empirical_normalization = True` is set in the non-submitted `rsl_rl_ppo_cfg.py`. Must verify that the normalizer state is persisted in the checkpoint and loaded by the TA's evaluation runner.
