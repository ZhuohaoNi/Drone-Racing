# Metrics Guide — Sim2Real Drone Racing

How to interpret the wandb/tensorboard metrics during training and evaluation. Organized by priority: check the top section first, then drill down if something looks wrong.

---

## Primary Metrics (check these first)

### `Lap/success_rate_3lap`
**What:** Rolling window (last 100 episodes) percentage of episodes where the drone completed 3 full laps without crashing or timing out.

**Target:** >95% before deploying to real. >99% ideal.

**How to read:**
- Should climb steadily during training. If it plateaus below 50%, the policy is stuck (see troubleshooting below).
- A sudden drop usually means the policy discovered an exploit that earns reward without completing laps, or the learning rate kicked it out of a good basin.

### `Lap/mean_lap_time`
**What:** Average time (seconds) per completed lap across the logging window.

**Target:** 8–12s for sim2real safety. Faster is fine only if success_rate stays high.

**How to read:**
- Decreasing = policy getting faster. This is good as long as success_rate doesn't drop.
- If lap time drops but success_rate also drops, the policy is getting aggressive — check `Episode_Reward/cmd_reg`.

### `Lap/best_lap_time`
**What:** Fastest single lap ever observed during training.

**Use:** Sanity check. If this is <4s the policy is probably exploiting something (e.g., cutting gates at extreme angles). If >20s the policy is too conservative or hovering.

---

## Reward Components

All `Episode_Reward/*` metrics are **mean episodic sum divided by max episode length** (normalized to per-second scale). This makes them comparable across runs with different episode lengths.

### `Episode_Reward/gate_pass`
**What:** Accumulated gate-pass reward per episode. This is the primary learning signal (+200 per gate).

**How to read:**
- Should increase steadily. This is the single most important reward curve.
- If flat at ~0 for >300 iterations: the policy is not passing any gates. Likely causes:
  - Spawn positions too far from gates (check `Reset/spline_ratio`)
  - `num_steps_per_env` too short for the sparse reward to appear in rollouts
  - Gate-pass reward dominated by negative terms (compare magnitude vs cmd_reg + crash)

### `Episode_Reward/cmd_reg`
**What:** Accumulated command regularization penalty (body-rate squared on roll/pitch/yaw).

**How to read:**
- Should be moderately negative (−0.5 to −3.0 range typical).
- If close to 0: policy is hovering / not moving.
- If very negative (< −5.0): policy is jerky. Increase `cmd_reg_rp_scale` / `cmd_reg_yaw_scale`, or check if the action latency is causing oscillations.
- **Sim2real importance:** If this is too large in sim, the real drone will overheat motors or oscillate. This is the main smoothness indicator.

### `Episode_Reward/crash`
**What:** Accumulated contact penalty (−0.1 per step while in contact with environment).

**How to read:**
- Should be close to 0 in a well-trained policy.
- If persistently negative: policy is brushing walls/ground. Check if spawn positions are valid.

### `Episode_Reward/lap_incomplete`
**What:** Constant per-step penalty (−0.05/step). Pushes the policy to finish laps rather than hover.

**How to read:**
- This is always negative and roughly constant across runs. Don't worry about its absolute value.
- Its purpose is to break ties: when the policy can either hover safely or attempt a gate, the per-step cost makes attempting the gate marginally better.

---

## Episode Termination

### `Episode_Termination/died`
**What:** Count of episodes in the logging batch that ended in a crash (contact forces exceeded threshold for 100+ consecutive steps).

**How to read:**
- Should decrease over training as the policy learns to avoid crashes.
- If increasing late in training: the policy is getting aggressive. May need to increase `death_cost`.

### `Episode_Termination/time_out`
**What:** Count of episodes that hit the 30s time limit without crashing.

**How to read:**
- Early in training: high time_out is fine (policy hasn't learned to fly yet, but at least it's not crashing).
- Late in training: high time_out means the policy is hovering or flying slowly. The `lap_incomplete` penalty should prevent this, but if it persists, check if `gate_pass` reward magnitude is large enough.

### `Episode_Termination/wrong_side`
**What:** Count of episodes terminated because the drone flew backward through a gate.

**How to read:**
- Should be 0 or very small. A spike means the policy discovered a reverse-gate exploit. The cooldown logic should prevent this, but watch for it.

---

## Reset Distribution

### `Reset/spline_ratio`
**What:** Fraction of resets using spline-based sampling (vs replay or ground).

**Expected:** ~0.50 (rest is 0.30 replay + 0.20 ground, but replay is only available when the buffer has data).

**How to read:**
- Early training: close to 0.80 (replay buffer empty, so only spline + ground).
- Mid/late training: should settle near 0.50 as replay buffer fills.

### `Reset/replay_ratio`
**What:** Fraction of resets using the gate-pass replay buffer.

**Expected:** 0 early, rising to ~0.30 as the policy starts passing gates.

**How to read:**
- If stuck at 0 after many iterations: the policy isn't passing gates at all.
- Healthy training shows this climbing from 0 to 0.20–0.30 over the first ~500 iterations.

### `Reset/ground_ratio`
**What:** Fraction of resets starting on the ground (z=0.03–0.06m).

**Expected:** ~0.20 (fixed ratio).

### `Reset/use_spline`
**What:** Binary flag (1.0 = spline resets enabled, 0.0 = V3-style linear). Useful for confirming A/B test configs.

---

## PPO / Training Health

### `Loss/surrogate`
**What:** PPO clipped surrogate loss (policy gradient objective).

**How to read:**
- Should decrease and stabilize. Large oscillations mean learning is unstable.
- If it goes to 0 and stays there: policy has collapsed to a deterministic action (check `Policy/mean_noise_std`).

### `Loss/value_function`
**What:** Critic's MSE loss on predicted vs actual returns.

**How to read:**
- Should decrease over training as the critic learns to predict returns.
- If it stays high while `surrogate` loss is low: the value function is struggling with sparse rewards. Consider deeper critic or longer rollouts.

### `Loss/entropy`
**What:** Mean entropy of the policy's action distribution.

**How to read:**
- Should start high (exploring) and gradually decrease (exploiting).
- If it drops to near 0 very quickly (<100 iterations): the policy collapsed prematurely. Increase `entropy_coef`.
- If it stays high after 1000+ iterations: the policy isn't converging. May need more training or stronger reward signal.

### `Loss/learning_rate`
**What:** Current learning rate (adaptive schedule targets `desired_kl=0.01`).

**How to read:**
- With adaptive schedule, LR increases when KL divergence is low (policy updates are small) and decreases when KL is high (policy updates are too large).
- If LR drops to near 0: the policy is making large updates each step — training is unstable. Lower `learning_rate` initial value or increase `num_mini_batches`.

### `Policy/mean_noise_std`
**What:** Mean standard deviation of the policy's action distribution (exploration noise).

**How to read:**
- Starts at `init_noise_std` (1.0) and should gradually decrease.
- If it drops below 0.1 very early: premature convergence.
- Typical healthy range by end of training: 0.2–0.5.

### `Train/mean_reward`
**What:** Mean total reward per episode across the rollout buffer.

**How to read:**
- The single summary number. Should trend upward.
- Absolute value depends on reward scale. Compare across runs with same reward config.

### `Train/mean_episode_length`
**What:** Mean episode length in steps.

**How to read:**
- Max is `episode_length_s / dt / decimation` = 30s / 0.002 / 10 = 1500 steps.
- Short episodes = lots of crashes (dying early). Long episodes = completing laps or hovering.
- Ideal: long episodes with high gate_pass reward (completing laps, not hovering).

---

## Troubleshooting Patterns

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `gate_pass` flat at 0, `time_out` high | Policy hovers, never reaches gates | Check spawn positions, increase rollout length, increase gate_pass scale |
| `gate_pass` rises then crashes to 0 | Policy found and lost a good strategy | Lower LR, increase `num_mini_batches`, check KL divergence |
| `success_rate` high but `cmd_reg` very negative | Policy is jerky but completes laps | Increase `cmd_reg_rp_scale` — smooth commands are critical for real deployment |
| `success_rate` oscillates ±20% | Training instability from sparse rewards | Increase `num_mini_batches`, lower LR, or try longer `num_steps_per_env` |
| `died` high throughout training | Spawns in bad positions or DR too extreme | Check DR ranges, tighten mass/tau variation, verify spline positions are valid |
| `replay_ratio` stuck at 0 | No gates being passed → empty buffer | Same as "gate_pass flat at 0" |
| `mean_noise_std` near 0, reward flat | Premature convergence | Increase `entropy_coef`, restart with higher `init_noise_std` |
| `lap_time` <4s | Possible exploit (cutting corners) | Check video, verify gate-pass detection bounds |

---

## Comparing Runs (A/B Testing)

When comparing configs (e.g., spline vs no-spline via `sweep.py --config 0 7`):

1. **Primary comparison:** `Lap/success_rate_3lap` at convergence (iteration 2000+). Higher is better.
2. **Convergence speed:** At what iteration does `gate_pass` first become consistently positive? Faster is better (more sample-efficient).
3. **Stability:** How much does `success_rate` oscillate in the last 500 iterations? Less is better.
4. **Smoothness:** Compare `Episode_Reward/cmd_reg` — less negative means smoother commands, better for real deployment.
5. **Speed (secondary):** Compare `Lap/mean_lap_time` only after confirming success_rate is comparable.

Use wandb's run comparison view: group runs by sweep config name, overlay the key metrics above.
