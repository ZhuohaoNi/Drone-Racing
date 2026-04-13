# Drone Racing RL Policy: Implementation Write-Up

**ESE 6510 — Physical Intelligence, Spring 2026**

---

## 1. Project Overview

This project trains a neural-network policy to fly a simulated Crazyflie quadcopter through a 7-gate racing track in NVIDIA Isaac Lab. The track features two signature maneuvers: a **powerloop** (gates 2→3, where the drone loops vertically over a double-gate structure) and a **chicane** (gates 5→6→0, rapid direction changes through offset gates). The policy is evaluated on a 3-lap time trial under randomized physical parameters that simulate a sim-to-real gap.

The implementation spans three files:

| File | Role |
|------|------|
| `rsl_rl/algorithms/ppo.py` | PPO algorithm (the `update()` method) |
| `rsl_rl/storage/rollout_storage.py` | GAE advantage computation (`compute_returns()`) |
| `quadcopter_strategies.py` | Reward, observation, and reset strategy |
| `agents/rsl_rl_ppo_cfg.py` | PPO and network hyperparameters |

The final submission (**V30**) achieves **97.3% success rate** with a **mean 3-lap time of 15.68s** (median 15.58s) across 5000 evaluation environments under TA-style domain randomization.

---

## 2. PPO Algorithm Implementation

### 2.1 Starting Point

The starter code provided an empty `update()` method inside `ppo.py`. The method receives mini-batches from the rollout storage (observations, actions, old log-probs, value targets, advantages, returns) and must implement the full PPO optimization step. Everything outside `update()` — action sampling, transition recording, return computation — was already provided.

### 2.2 What We Implemented

The `update()` method in `src/third_parties/rsl_rl_local/rsl_rl/algorithms/ppo.py` (lines 123–215) performs the following for each mini-batch:

**Step 1 — Recompute current policy outputs:**
```python
self.actor_critic.update_distribution(observations)
actions_log_prob_batch = self.actor_critic.get_actions_log_prob(sampled_actions)
value_batch = self.actor_critic.evaluate(critic_observations)
```
The actor-critic's `update_distribution()` forward-passes observations through the actor network to produce a Gaussian distribution, then `get_actions_log_prob()` evaluates the log-probability of the *previously sampled* actions under the *current* policy. `evaluate()` passes through the critic to get value predictions.

**Step 2 — KL divergence (for adaptive LR):**
```python
log_ratio = actions_log_prob_batch - prev_log_probs
kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
```
This is the approximation `KL ≈ E[(r-1) - log(r)]` where `r = π_new/π_old`. It measures how far the new policy has drifted from the old, and is used post-update to adjust the learning rate.

**Step 3 — Clipped surrogate policy loss (the core of PPO):**
```python
ratio = torch.exp(actions_log_prob_batch - prev_log_probs)
surrogate_1 = ratio * advantage_estimates
surrogate_2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantage_estimates
surrogate_loss = -torch.min(surrogate_1, surrogate_2).mean()
```
The probability ratio `π_new(a|s) / π_old(a|s)` scales the advantage. Clamping the ratio to `[1 - ε, 1 + ε]` (ε = 0.2) prevents destructively large policy updates. Taking the `min` of the clipped and unclipped objectives creates a pessimistic bound — the policy only gets credit for an improvement if *both* the clipped and unclipped versions agree.

**Step 4 — Clipped value loss:**
```python
value_clipped = value_targets + (value_batch - value_targets).clamp(-clip_param, clip_param)
value_loss = torch.max((value_batch - returns)^2, (value_clipped - returns)^2).mean()
```
Analogous clipping for the value function: the new value prediction cannot deviate from the old value prediction by more than ε. This prevents the critic from making large jumps that destabilize advantage estimation.

**Step 5 — Entropy bonus:**
```python
entropy_loss = self.actor_critic.entropy.mean()
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss
```
The entropy of the action distribution is *subtracted* from the loss (equivalently, maximized). This encourages exploration by penalizing policies that collapse to deterministic actions. `entropy_coef = 0.005` was found through experimentation (0.0 led to premature convergence; 0.01 caused instability).

**Step 6 — Gradient step:**
```python
self.optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
self.optimizer.step()
```
Gradient clipping at `max_grad_norm = 1.0` prevents exploding gradients from rare high-advantage transitions.

**Step 7 — Adaptive learning rate schedule (after all epochs):**
```python
if mean_kl > desired_kl * 2.0:
    learning_rate = max(1e-5, learning_rate / 1.5)
elif mean_kl < desired_kl / 2.0:
    learning_rate = min(1e-2, learning_rate * 1.5)
```
If the average KL divergence across the update exceeds `2 × desired_kl` (0.01), the learning rate is halved (divided by 1.5). If KL is too small, the LR is increased. This auto-tunes step sizes: early training uses larger LR for fast learning, late training uses smaller LR for fine-tuning. This was critical — without adaptive LR, training would collapse in later iterations as fixed-LR updates became too aggressive.

### 2.3 Design Decisions

- **Why clipped value loss?** Standard MSE value loss can cause the critic to overshoot on stale data. Clipping stabilizes training, especially with the `num_learning_epochs = 8` we use (more epochs = more stale data per epoch).
- **Why adaptive LR over fixed?** Drone racing reward landscapes are non-stationary (the drone learns new gates over time). A fixed LR that works at iteration 100 is too large at iteration 3000. The KL-adaptive schedule (V1) fixed repeated late-training collapses observed without it.

---

## 3. GAE Advantage Computation

### 3.1 Original vs. Modified `compute_returns()`

The original `rollout_storage.py` computed GAE inside a single reverse loop, recalculating `next_values` and `next_is_not_terminal` at every timestep. Our modification in `src/third_parties/rsl_rl_local/rsl_rl/storage/rollout_storage.py` (lines 132–158) pre-computes these quantities as tensors:

```python
# Vectorized: compute all TD errors in parallel
next_values = torch.cat([self.values[1:], last_values.unsqueeze(0)])
next_is_not_terminal = 1.0 - self.dones.float()
deltas = self.rewards + next_is_not_terminal * gamma * next_values - self.values
```

The loop still exists (GAE has a temporal dependency that prevents full parallelization), but it now only accumulates the advantage using pre-computed deltas:

```python
advantage = torch.zeros_like(last_values)
for step in reversed(range(self.num_transitions_per_env)):
    advantage = deltas[step] + next_is_not_terminal[step] * gamma * lam * advantage
    self.returns[step] = advantage + self.values[step]
```

**Why this matters:** With 16,384 parallel environments and 24 steps per rollout, the delta tensor is `(24, 16384, 1)`. Pre-computing it as a single batched operation avoids 24 separate `self.values[step+1]` lookups and conditional branches inside the loop, reducing Python overhead and improving GPU utilization.

### 3.2 GAE Formula

For each timestep `t`, the generalized advantage estimate is:

```
δ_t = r_t + γ(1 - d_t)V(s_{t+1}) - V(s_t)           [TD error]
A_t = δ_t + γλ(1 - d_t)A_{t+1}                        [GAE recursive]
R_t = A_t + V(s_t)                                      [return target]
```

Where `γ = 0.99` (discount factor) and `λ = 0.95` (GAE lambda, trading off bias vs. variance in advantage estimation). The `(1 - d_t)` term zeroes out bootstrapped values at episode boundaries. Advantages are then normalized to zero mean and unit variance before PPO uses them.

---

## 4. Network Architecture and Hyperparameters

Configured in `src/isaac_quad_sim2real/tasks/race/config/crazyflie/agents/rsl_rl_ppo_cfg.py`:

| Parameter | Original | Final | Rationale |
|-----------|----------|-------|-----------|
| `actor_hidden_dims` | [128, 128] | **[256, 256]** | Larger network capacity for complex gate-specific maneuvers (powerloop, chicane) |
| `critic_hidden_dims` | [512, 256, 128, 128] | [512, 256, 128, 128] | Unchanged; asymmetric (bigger critic) helps value estimation |
| `empirical_normalization` | False | **True** | Running mean/std normalization of observations; critical for stable training |
| `entropy_coef` | 0.0 | **0.005** | Prevents premature entropy collapse; 0.01 was too high (V20) |
| `num_learning_epochs` | 5 | **8** | More gradient steps per rollout; enabled by clipped losses |
| `init_noise_std` | 1.0 | 1.0 | Unchanged; `min_std = 0.0` allows std to shrink freely |
| `activation` | ELU | ELU | Unchanged |
| `clip_param` | 0.2 | 0.2 | Standard PPO clipping |
| `learning_rate` | 5e-4 | 5e-4 | Initial LR; adapted via KL schedule |
| `desired_kl` | 0.01 | 0.01 | Target KL for adaptive LR |
| `gamma` | 0.99 | 0.99 | 0.995 tested in V17, caused instability |

The actor outputs a 4D action vector (thrust, roll rate, pitch rate, yaw rate) through a final `Tanh` layer that bounds actions to [-1, 1]. The action distribution is Gaussian with learned per-action standard deviation.

**Training configuration:** 16,384 parallel environments, 24 policy steps per rollout (0.48s at 50Hz), 5,000 iterations for the base V26 policy, plus 2,000 fine-tuning iterations for V30.

---

## 5. Racing Strategy: Reward Design

### 5.1 Reward Components

Defined in `get_rewards()` of `quadcopter_strategies.py` (lines 127–322), with scales in `train_race.py`:

| Reward | Scale | Type | Description |
|--------|-------|------|-------------|
| **Progress** | +50.0 | Dense | `Δd = d_{t-1} - d_t`, clamped to [-1, 1]. Distance reduction toward current target (gate center or powerloop waypoint). Computed every timestep. |
| **Velocity toward gate** | +8.0 | Dense | Dot product of drone velocity with direction to target, clamped to [-2, 8]. For non-powerloop gates, blended 5/8 toward current gate + 3/8 toward next gate. Computed every timestep. |
| **Orientation** | -1.0 | Dense | Sum of |roll| + |pitch| when exceeding 0.5 rad (~30°). Zero below threshold. Computed every timestep. |
| **Smoothness** | -0.2 | Dense | L2 norm of action difference `‖a_t - a_{t-1}‖`. Penalizes jerky control. Computed every timestep. |
| **Crash** | -1.0 | Dense | Per-step penalty when contact forces exceed threshold for >100 steps. Computed every timestep. |
| **Gate Pass** | +200.0 | Sparse | Binary 0/1 when the drone crosses the gate plane from the correct direction within the gate opening bounds. Only fires at the instant of gate passage. |
| **Death cost** | -10.0 | Sparse | One-time penalty applied on episode termination (crash or out-of-bounds). Only fires once per episode. |

**Dense vs. sparse reward balance:** Five of the seven reward components are **dense** — they provide a continuous learning signal at every one of the 50 Hz policy timesteps. These guide the drone's moment-to-moment flight behavior (fly toward the gate, stay upright, fly smoothly, avoid contacts). Two components are **sparse** — they fire only at discrete events (passing a gate, dying). The sparse gate-pass reward has the largest single-event magnitude (200.0) to compensate for its infrequency, ensuring the policy receives a strong gradient signal for the behavior that matters most (actually completing gates). The dense rewards act as **reward shaping** that bridges the gap between sparse gate-pass events, giving the optimizer a smooth gradient landscape to follow rather than requiring the policy to discover gate passages through random exploration alone.

### 5.2 Gate-Passing Detection

The original starter code used a simple proximity check (`dist_to_gate < 0.1`), which is unreliable because the drone may approach a gate from the wrong side or pass nearby without going through. Our implementation uses a **plane-crossing method in gate-local coordinates**:

```python
curr_x = self.env._pose_drone_wrt_gate[:, 0]   # drone x in gate frame
prev_x = self.env._prev_x_drone_wrt_gate        # previous step
crossed_plane = (prev_x > 0) & (curr_x <= 0)    # positive → negative = correct direction
within_bounds = (|gate_y| < side/2) & (|gate_z| < side/2)  # within gate opening
gate_passed = crossed_plane & within_bounds
```

This detects when the drone crosses the gate plane from the correct side (positive-x → negative-x in gate frame) while within the gate's physical opening. After a gate pass, `prev_x` is recomputed in the *new* target gate's frame to avoid false positives.

### 5.3 Wrong-Side Detection

If the drone crosses the gate plane in the *reverse* direction (negative-x → positive-x), it is immediately terminated:

```python
wrong_side_crossed = (prev_x < 0) & (curr_x >= 0) & within_bounds
```

A 5-step cooldown after each gate pass prevents the gate-clearing transient from triggering a false wrong-side detection.

### 5.4 Racing-Line Velocity Blend

A key insight (V19): rewarding velocity toward only the *current* gate causes the drone to approach each gate head-on, then make sharp turns to the next gate. By blending in a component toward the *next* gate:

```
vel_reward = (5/8) × vel_toward_current + (3/8) × vel_toward_next
```

the drone learns to **cut corners** — approaching each gate at an angle that naturally transitions toward the next gate, like a racing line. The 3/8 weight was tuned experimentally: 2/8 (V19) was too gentle, 4/8 (V27) caused late-stage collapse by pulling the drone too far off the racing line.

**Exception:** During the powerloop segment (Gate 3), the blend is disabled (`vel_toward_current` only) because the powerloop requires precise vertical maneuvering toward virtual waypoints, not corner-cutting toward Gate 4.

### 5.5 Powerloop Guide System

The powerloop is the most complex maneuver: the drone must fly over the double-gate structure and enter Gate 3 from the same side as Gate 2. This was implemented as a **virtual waypoint system** that overrides `_desired_pos_w` for gates 2 and 3:

**Gate 2 (pre-powerloop redirect):** When the drone targets Gate 2, `_desired_pos_w` is overridden to the powerloop apex `[0.0, -0.3, 1.6]` instead of Gate 2's actual center. The drone clips through Gate 2's opening while already climbing (gate-pass detection uses gate-frame coordinates, not `_desired_pos_w`). This prevents the "pass Gate 2 horizontally then circle back" pattern observed in V21.

**Gate 3 (2-phase guide):**
- **Phase 0 (climb):** Target = apex `[0.0, -0.3, 1.6]`. Transitions to Phase 1 when `z > 1.3m` OR `dist_to_apex < 0.8m`.
- **Phase 1 (descent):** Target = offset center `[0.425, 0.0, 0.75]`, shifted toward Gate 4 for a smooth exit.

An earlier 3-phase design (V14) included a `pre_entry` waypoint on the +y side of Gate 3, but V24 removed it — the drone learned to fly around the side of the gate structure instead, which is a shorter and equally valid path (17.34s vs 17.50s).

---

## 6. Observation Space

### 6.1 Design: 31-Dimensional Ego-Centric Observations

Defined in `get_observations()` (lines 324–412):

| Dims | Feature | Frame | Purpose |
|------|---------|-------|---------|
| 3 | Linear velocity | Body | Drone's speed relative to its own orientation |
| 3 | Angular velocity | Body | Roll/pitch/yaw rates for attitude awareness |
| 3 | Gravity vector | Body | Replaces quaternion (4D→3D); encodes orientation w.r.t. up-direction |
| 3 | Current gate position | Body | Where the target gate is relative to the drone |
| 3 | Next gate position | Body | 1-gate lookahead for trajectory planning |
| 3 | Next-next gate position | Body | 2-gate lookahead for anticipating turns |
| 3 | Current gate normal | Body | Which direction the drone must fly through the gate |
| 3 | Next gate normal | Body | Approach direction for the next gate |
| 1 | sin(yaw error) | — | Continuous heading alignment (no wraparound discontinuity) |
| 1 | cos(yaw error) | — | Continuous heading alignment |
| 1 | Normalized gate index | — | `idx / 7` — tells policy which gate it's targeting |
| 4 | Previous actions | — | Action history for smoothness and proprioception |

### 6.2 Key Design Decisions

**Body frame, not world frame:** All spatial observations (velocities, gate positions, normals) are expressed in the drone's body frame. This makes the policy ego-centric — it doesn't need to learn different behaviors for different absolute positions on the track, only relative spatial relationships.

**Gravity vector replaces quaternion:** A quaternion (4D) is a redundant representation for attitude. The gravity vector in body frame (3D) directly encodes the drone's tilt relative to "up", which is what the policy actually needs (how much roll/pitch correction is needed). This saved 1 observation dimension and improved learning signal clarity.

**Gate normal vectors:** Knowing the gate's normal vector tells the policy *which direction to fly through the gate*. Without this, the policy would need to infer entry direction from position alone, which is ambiguous for the double-gate structure (same physical position, different entry directions for Gates 3 vs 6).

**2-gate lookahead:** The `next_next_gate_pos_b` (added in V12) gives the policy advance notice of upcoming turns, enabling anticipatory banking and speed adjustment. This improved lap times by allowing smoother transitions between gate segments.

**Normalized gate index:** A single scalar `idx / 7` tells the policy which gate it's targeting. This is essential because gates 2/3 require powerloop behavior while other gates need different strategies. The policy uses this to condition its behavior on track position.

**Empirical normalization:** The runner applies running mean/variance normalization to observations before feeding them to the network. This was critical (V7-V9 experiments showed 4× worse reward without it). The normalizer state is saved with the model checkpoint, ensuring consistent behavior at evaluation time.

---

## 7. Reset Strategy

### 7.1 Randomized Episode Initialization

Defined in `reset_idx()` (lines 414–673), the reset strategy determines where and how drones respawn at the start of each episode. This is critical because the policy must generalize across all starting conditions, not just a single spawn point.

**Random starting gate:** Each resetting environment selects a random gate index (0–6) as its target. This ensures the policy trains on all gate segments equally, not just the first few gates.

**Curriculum-based spawn distance:** Distance behind the target gate ramps over 800 iterations:
- Early training: [1.0, 2.0]m — close spawns for easy initial learning
- Late training: [1.5, 4.0]m — farther spawns for longer approach segments

**Spawn noise:** Lateral noise ±1.0m, vertical ±0.5m, yaw ±0.3 rad. This prevents the policy from memorizing exact spawn-to-gate trajectories.

**40% mid-track spawns (V26):** 40% of resets spawn the drone between two consecutive gates (linearly interpolated at a random 20–80% position). This trains the policy on gate-to-gate transitions, which are critical for continuous multi-lap racing. This was the single most impactful reset change — increasing from 10% (V10) to 25% (V25) to 40% (V26) improved success rate from 86.7% to 96.3%.

**~20% ground-level spawns:** 10% at `z = 0.05m` with current velocity + 10% "ground takeoff" at `z = 0.05m` with zero velocity. These simulate the TA evaluation condition where the drone starts on the ground and must take off before racing.

**50% velocity-initialized spawns:** Half of air spawns start with 0.5–3.0 m/s forward velocity, simulating mid-flight conditions. Ground takeoff spawns always start at zero velocity.

### 7.2 Domain Randomization

On every episode reset during training, `_randomize_dynamics()` re-samples physical parameters:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Thrust-to-weight ratio | ±5% of nominal | Motor strength variation |
| Aerodynamic drag XY | 0.5× – 2.0× nominal | Air resistance variation |
| Aerodynamic drag Z | 0.5× – 2.0× nominal | Vertical drag variation |
| PID gains kp, ki (roll/pitch/yaw) | ±15% of nominal | Controller response variation |
| PID gains kd (roll/pitch/yaw) | ±30% of nominal | Damping variation |

These ranges match the TA's evaluation randomization bounds (Section 3.1 of the project description). During evaluation (`is_train = False`), a separate `_set_default_dynamics()` sets all parameters to nominal values, allowing the TA's evaluation script to apply its own fixed randomization.

---

## 8. Iterative Development Process

The final V30 policy was the result of 31 experiment versions over 7 days. Below is a summary of the key milestones and lessons learned:

### Phase 1: Foundation (V1–V2)
- **V1:** PPO implemented, adaptive LR added, 23-dim observations, basic gate-pass and progress rewards. Successfully completed laps but slowly.
- **V2:** Added velocity-toward-gate reward (scale 5.0), orientation penalty (-2.0 above 30°), smoothness penalty (-0.2), sin/cos yaw error observations, domain randomization. **Best early result:** 3 laps consistently completed with stable training. This became the baseline that all future versions were compared against.

### Phase 2: Reward Tuning Failures (V3–V9)
- **V3:** Relaxed orientation penalty (0.5→1.0 rad threshold) — drone started flipping at tight turns.
- **V4–V5:** Added heading reward — combined with velocity reward, policy became "speed-obsessed" and crashed at every turn.
- **V6:** Removed velocity reward entirely — drone became too conservative (26s vs 17s).
- **V7–V9:** Reverted to V2. Discovered empirical normalization was essential (4× reward drop without it).

**Lesson:** Reward components interact non-linearly. The V2 combination of strong orientation penalty + moderate velocity reward + no heading reward was a stable equilibrium that shouldn't be disturbed independently.

### Phase 3: Observation & Detection Improvements (V10–V13)
- **V10:** Domain randomization guarded with `is_train`, curriculum spawn distance, 20% velocity-initialized spawns.
- **V11:** Gate index observation + wrong-side detection. Policy now knows which gate it targets.
- **V12:** Gravity-in-body-frame replaces quaternion, 2-gate lookahead, next gate normal.
- **V13:** 50% velocity-initialized spawns, unified body-frame normals, 3D distance for progress.

### Phase 4: Powerloop Development (V14–V18)
- **V14:** First powerloop implementation (3-phase guide: apex → pre-entry → gate center). **First successful powerloop: 21.6s.**
- **V15–V16:** Speed tuning (velocity scale 5→8, velocity clamp 5→8), tighter powerloop apex (2.5→1.6m), larger actor network (128→256). **17.78s.**
- **V17:** 2-phase powerloop (removed pre-entry) + gamma 0.995 — gamma change caused collapse, reverted.
- **V18:** 10% ground spawn + Gate 3 offset center. **Critical fix**: without ground spawns, the drone crashed immediately in TA evaluation.

### Phase 5: Racing Line Optimization (V19–V26)
- **V19:** Racing-line velocity blend (6/8 current + 2/8 next gate). **17.56s.** Late-training instability.
- **V20:** Attempted entropy/rollout fixes — 3× repeated collapses. Reverted.
- **V21–V23:** Gate 2 pre-powerloop redirect, velocity blend tuning. **V23: 17.50s, 92.8% SR.**
- **V24:** 2-phase powerloop (removed pre-entry entirely). Drone uses alternative path. **17.34s.**
- **V25:** Mid-track spawn 10→25%. **SR dropped to 86.7%** but median improved (16.46s).
- **V26:** Mid-track spawn 25→**40%**. **SR recovered to 96.3%, mean 15.77s.** This became the base policy.

**Key finding:** Mid-track spawning rate was the single most important factor for success rate. More mid-track training taught the policy to handle all gate-to-gate transitions robustly.

### Phase 6: Robustness Fine-Tuning (V27–V30)
- **V27:** Velocity blend 3/8→4/8 — late-stage collapse, reverted.
- **V28–V29:** Gate 2 backward-pass detection experiments (added then removed for safety).
- **V30 (submission):** **Fine-tuned V26** with wider domain randomization (TWR ±8%, aero 0.3–2.5×, PID kp/ki ±25%, kd ±40%) and lower entropy (0.005→0.001 for fine-tuning). Trained 2000 additional iterations from V26 checkpoint.

**V30 result: 97.3% SR, mean 15.68s, median 15.58s** — best ever across all versions.

### Final Submission Cleanup
- Removed Gate 2 backward-pass detection (could cause false positives under extreme parameter randomization).
- Kept evaluation `_desired_pos_w` override so TA visualizer markers point to actual gate centers rather than virtual powerloop targets.

---

## 9. Robustness, Performance Distribution, and Fine-Tuning

A fast policy that fails under parameter perturbation is useless in evaluation. This section describes how we ensured robustness, analyzed the performance distribution under domain randomization, and used fine-tuning to produce the final V30 submission.

### 9.1 Ensuring Robustness

Robustness was addressed through three complementary mechanisms:

**a) Training-time domain randomization (DR).** On every episode reset during training, `_randomize_dynamics()` re-samples all physical parameters (TWR, aerodynamic drag, PID gains) from uniform distributions matching the TA's evaluation bounds. Because each of the 16,384 parallel environments independently re-samples on reset, the policy sees tens of thousands of distinct parameter configurations per training run. This forces the neural network to learn a single policy that works across the entire parameter space, rather than overfitting to a single dynamics setting.

**b) Diverse reset conditions.** The policy must handle arbitrary starting states, not just a clean launch from Gate 0. Our reset strategy (Section 7) creates diversity along multiple axes: random starting gate (any of 7), random distance/angle behind the gate, 40% mid-track spawns, ~20% ground-level spawns (simulating the TA's ground-start evaluation), and 50% velocity-initialized spawns. This makes the policy robust to the unknown initial pose the TA will use during evaluation (hard-coded within `x_local ∈ [-3.0, -0.5]` and `y_local ∈ [-1.0, 1.0]`).

**c) Over-preparation in DR ranges (V30 fine-tuning).** For the final fine-tuning phase, we deliberately *widened* the DR ranges beyond the TA's actual bounds:

| Parameter | TA bounds | V26 training | V30 fine-tuning |
|-----------|-----------|--------------|-----------------|
| TWR | ±5% | ±5% | **±8%** |
| Aero drag | 0.5–2.0× | 0.5–2.0× | **0.3–2.5×** |
| PID kp, ki | ±15% | ±15% | **±25%** |
| PID kd | ±30% | ±30% | **±40%** |

The rationale is "over-preparation": if the policy can handle ±8% TWR variation, the TA's ±5% feels like easy mode. This is analogous to training at altitude for a sea-level race.

### 9.2 Batch Evaluation and Performance Distribution Analysis

To understand how the policy performs across the space of possible TA parameter selections, we built a batch evaluation tool (`scripts/rsl_rl/batch_eval_race.py`). The key design mirrors the TA's protocol:

- **TA protocol:** Randomly select 3 parameters from the pool (TWR, aero_xy, aero_z, kp_rp, ki_rp, kd_rp, kp_y, ki_y, kd_y), sample values within bounds, hold constant for all students.
- **Our self-evaluation:** Run N parallel environments (up to 5000), each with an *independently* sampled 3-parameter subset and values. This covers far more parameter combinations in a single run than sequential testing.

**Metrics collected:**
- **Success rate (SR):** Fraction of environments completing 3 full laps within the time limit.
- **3-lap time distribution:** Mean, median, standard deviation, min, max of completion times for successful environments.
- **Per-bucket analysis:** Environments grouped by lap time into performance buckets to identify where the policy struggles.

**V26 baseline distribution (5000 envs):**
- SR = 94.9%, mean = 15.78s, median = 15.68s, std = 0.54s
- The ~5% failure cases were concentrated in specific parameter combinations (high aero drag + low TWR + aggressive PID perturbation), forming a **long tail** in the time distribution.

**V30 distribution (5000 envs):**
- SR = **97.3%**, mean = **15.68s**, median = **15.58s**, std = **0.50s**
- Best: 14.82s, worst: 21.26s
- **P(time < 16.06s) = 84.0%** — a key target threshold derived from leaderboard analysis.

### 9.3 Paired Comparison: V26 vs V30

To understand *how* V30 improved over V26, we ran a **paired evaluation**: 1000 environments with identical DR parameters for both policies. This eliminates randomness and reveals which parameter regions each policy handles better.

**Key findings:**

1. **Near-zero rank correlation** (Pearson r = -0.016, Spearman ρ = 0.048): V30 is not a uniform shift that makes everything faster. It *reshuffled* which parameter combinations are hard vs. easy. This is expected — wider DR training trades peak performance on easy configs for robustness on hard ones.

2. **Bucketed analysis by V26 performance:**

| Bucket | V26 time range | V30 change | Interpretation |
|--------|---------------|------------|----------------|
| A (easy) | < 15.6s | +0.87s slower | V30 regressed on V26's easiest configs |
| B (normal) | 15.6–15.9s | ~neutral | No significant change |
| C (borderline) | 15.9–16.1s | **-1.32s faster** | V30 rescues borderline cases |
| D (risk zone) | 16.1–16.4s | **-0.99s faster** | Pass rate improved 0% → 77% |
| E (worst) | ≥ 16.4s | **-1.38s faster** | Largest improvement on hardest configs |

3. **Threshold crossing analysis** (target = 16.06s):
   - 168 environments that failed under V26 now pass under V30 (78.5% of V26 failures rescued)
   - 137 environments regressed (V26 pass → V30 fail) — new blind spots from redistributed difficulty
   - **Net gain: +31 environments passing**

**Interpretation:** V30's wider DR training specifically repaired V26's weak spots (buckets C/D/E) at the cost of marginal regression in V26's easiest parameter combinations. Since the TA's evaluation parameters are unknown but fixed, and the risk of landing in V26's bucket D (16.1s) is high, V30's robust tail behavior makes it the superior submission.

### 9.4 Fine-Tuning Methodology (V26 → V30)

Rather than training V30 from scratch, we used **continual learning** (fine-tuning from the V26 checkpoint). This was both more sample-efficient and produced a better result than from-scratch training under the wider DR ranges.

**What was loaded from V26:**
- Actor and critic network weights
- Adam optimizer state (momentum and variance accumulators)
- Observation normalizer running mean and variance

**What was changed for fine-tuning:**
1. **Wider DR ranges** (see table in Section 9.1) — the policy encounters harder dynamics variations than it saw during V26 training.
2. **Lower entropy coefficient** (0.005 → 0.001) — the policy is already well-explored from V26's 5000 iterations; during fine-tuning we want to *consolidate* learned behavior rather than explore further. Higher entropy during fine-tuning was wasteful and introduced unnecessary variance.
3. **2000 additional iterations** — enough to adapt to the wider DR ranges without catastrophic forgetting of the core racing strategy.
4. **Reduced ground spawn** (24% combined, down from 44% in an intermediate V30a attempt that increased time variance).

**Why fine-tuning rather than from-scratch?**
- V26 already had a strong racing policy (96.3% SR, 15.77s mean). Training from scratch with wider DR ranges would require the policy to re-learn basic flight, gate-passing, and powerloop maneuvers — all of which V26 already mastered.
- The optimizer state carries valuable information: Adam's per-parameter learning rate adaptation (momentum/variance) from 5000 iterations of V26 training helps the fine-tuning phase converge faster and more stably.
- The observation normalizer statistics (running mean/variance over millions of transitions) provide well-calibrated normalization that a fresh run would need hundreds of iterations to re-estimate.
- **Empirical validation:** V31 attempted from-scratch training with V26's config + wider DR ranges under a different seed. While it was intended to test whether a different random seed would find different blind spots, the from-scratch approach required the full 5000 iterations to converge, confirming that fine-tuning was the more efficient path.

**Risk of fine-tuning — catastrophic forgetting:** When the distribution of training data changes (wider DR), the policy can "forget" previously learned behavior. We mitigated this by:
- Keeping the DR range widening moderate (not extreme)
- Using a low learning rate (inherited from V26's adaptive schedule, which had already decreased from the initial 5e-4)
- Limiting fine-tuning to 2000 iterations (not enough to overwrite V26's core representations)
- Monitoring success rate on WandB to detect any collapse early

---

## 10. Summary of Changes from Starter Code

### `ppo.py` — From empty TODO to full PPO update
- Implemented clipped surrogate policy loss, clipped value loss, entropy bonus
- Added KL-adaptive learning rate schedule
- Added `mean_kl` tracking variable

### `rollout_storage.py` — Vectorized GAE
- Pre-computed `next_values` and `next_is_not_terminal` as tensors
- Vectorized TD error (`deltas`) computation
- Simplified loop to accumulate advantages from pre-computed deltas

### `rsl_rl_ppo_cfg.py` — Hyperparameter tuning
- `empirical_normalization`: False → True
- `actor_hidden_dims`: [128,128] → [256,256]
- `entropy_coef`: 0.0 → 0.005
- `num_learning_epochs`: 5 → 8

### `quadcopter_strategies.py` — Complete rewrite
**Rewards** (original: proximity-based hover → final: 6-component racing reward):
- Plane-crossing gate detection replacing distance threshold
- Wrong-side crash detection with cooldown
- Velocity-toward-gate with 5/8+3/8 racing-line blend
- Powerloop 2-phase virtual waypoint system
- Orientation, smoothness, crash penalties
- Lap time and success rate tracking

**Observations** (original: 13-dim world-frame → final: 31-dim body-frame):
- Body-frame velocities, angular velocities
- Gravity vector (replacing quaternion)
- 3-gate lookahead (current, next, next-next) in body frame
- Gate normals in body frame
- Sin/cos yaw error, normalized gate index, previous actions

**Reset** (original: fixed Gate 0, 2m behind → final: randomized multi-strategy):
- Random starting gate, curriculum spawn distance
- 40% mid-track interpolation, ~20% ground spawns, 50% velocity init
- Per-reset domain randomization of physics parameters
- Wider noise in position (±1.0m lateral) and yaw (±0.3 rad)

### `train_race.py` — Reward scales
- Added `gate_pass_reward_scale` (200.0), `vel_toward_gate_reward_scale` (8.0), `orientation_reward_scale` (-1.0), `smoothness_reward_scale` (-0.2)
- Progress scale: 50.0, crash: -1.0, death cost: -10.0
