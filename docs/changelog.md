# Changelog

All notable changes to this project will be documented in this file.

## [V23] - 2026-03-31

### Changed
- **Gate 2 vel_next restored**: `is_powerloop_segment` changed from `(idx_wp==2) | (idx_wp==3)` → `(idx_wp==3)` only. Gate 2's `vel_next` component (3/8 weight) now points toward Gate 3/apex, naturally providing an upward bias during the Gate 2 approach — complementing the Gate 2 pre-powerloop redirect (→ apex) added in V22. Gate 3's active powerloop phase still uses pure `vel_toward_current`.

### Experiment Results
- **3 laps in 17.50s** ✅ — 0.38s faster than V22 (17.88s). Dual upward bias on Gate 2 (desired_pos_w → apex + vel_next → Gate 3) producing cleaner powerloop entry without the horizontal-pass-then-return issue from V21.


## [V22] - 2026-03-31


### Changed
- **Gate 2 pre-powerloop guide**: When `idx_wp == 2`, `desired_pos_w` is overridden to the powerloop apex `[0.0, -0.3, 1.6]` instead of Gate 2 center. The drone clips through Gate 2's opening while already climbing (gate_pass detection uses gate-frame y/z, unaffected by desired_pos_w). Fixes the "pass Gate 2 horizontally → circle back to Gate 3" pattern observed in V21.

### Experiment Results
- **3 laps in 17.88s** ✅ — training very stable (no collapses over 2.5k steps), `success_rate_3lap` ~80-85% throughout.
- Slower than V21 (16.96s): Gate 2 redirect changes training dynamics — drone now takes a more deliberate powerloop approach vs V21's faster (if visually suboptimal) path.
- `best_lap_time` in training converges at ~5.5s/lap (~16.5s potential), eval at 17.88s — gap suggests robustness difference or reset misalignment.

## [V21] - 2026-03-31

### Experiment Results
- **3 laps in 16.96s** ✅ — 0.6s faster than V19 (17.56s).
- **Gate 2 issue**: Drone passes Gate 2 horizontally then returns to Gate 3 instead of looping over. Fixed in V22. Which is not valid.

## [V19] - 2026-03-31


### Changed
- **vel_next blend**: 2/8 → **3/8** (current: 6/8 → 5/8). Same PPO config as V19 (entropy=0.005, num_steps_per_env=24) after V20 proved unstable.
- **V20 reverted**: `entropy_coef=0.01` and `num_steps_per_env=48` both reverted — caused 3× repeated collapses vs V19's single one.

## [V20] - 2026-03-31 *(REVERTED)*

### Changed (reverted)
- `entropy_coef`: 0.005 → 0.01 — caused repeated collapses at steps ~350, ~650, ~800.
- `num_steps_per_env`: 24 → 48 — longer rollouts increased advantage variance, amplifying instability.

### Experiment Results
- Multiple `success_rate_3lap` collapses; training never stabilized. Reverted to V19 PPO params.

## [V19] - 2026-03-31


### Changed
- **`entropy_coef`**: 0.005 → 0.01 — stronger exploration bonus to prevent the late-training entropy collapse observed in V19 (~step 2.5k `success_rate_3lap` drop).
- **`num_steps_per_env`**: 24 → 48 — doubles rollout length from 0.48s to 0.96s at 50Hz. At 17s/lap the drone now covers a near-complete gate-to-gate segment per rollout, giving GAE cleaner advantage estimates and reducing gradient noise from truncated trajectories.

## [V19] - 2026-03-31


### Changed
- **Racing-line velocity reward** — blend `vel_toward_current` (6/8) + `vel_toward_next` (2/8), applied only to non-powerloop gates:
  - `idx_wp == 2` (approach) and `idx_wp == 3` (powerloop) both use pure `vel_toward_current` (weight = 1.0), fully preserving existing Gate 3 behavior.
  - All other gates: `vel_toward_gate = (6/8) * vel_toward_current + (2/8) * vel_toward_next`.
  - Pre-normalized (weights sum to 1) so effective scale relative to `vel_toward_gate_reward_scale` (8.0) is unchanged.
- **Motivation**: the 2/8 next-gate component biases approach trajectories toward the inside corner of bend gates (e.g. Gate 0, Gate 5), encouraging natural racing-line corner clipping without manual waypoint tuning.
- **Extra WandB diagnostics**: `Episode_Reward/vel_current_mean` and `Episode_Reward/vel_next_mean` logged separately to monitor component balance.

### Experiment Results
- **3 laps in 17.56s** ✅ — 0.22s faster than V18 (17.78s); `vel_current_mean` ~150 vs `vel_next_mean` ~60, next-gate component not dominating.
- **Late-training instability** observed around step 2.5k: `success_rate_3lap` drops sharply before partially recovering. Likely entropy collapse (`entropy_coef=0.005` may be insufficient at this stage). Flagged for next experiment.


## [V18] - 2026-03-31

### Changed
- **Gate 3 Powerloop**: Retained 3-phase (apex, pre-entry, offset center). Reverted apex to `[0.0, -0.3, 1.6]` and pre-entry to `[0.0, 1.0, 1.2]` (identical to V16) but added Phase 2 offset gate center target `[0.425, 0.0, 0.75]` to shift the drone towards Gate 4 without losing the Gate 3 trigger.
- **Training Robustness (Crucial Fix)**: Added 10% chance of spawning near the ground `[z=0.05]` with 0 initial velocity during training resets. This simulates the exact evaluation condition used by TAs and prevents the drone from instantly crashing on the floor during `play`.
- **Gamma**: Reverted back to 0.99 (from 0.995) after observing training instability/collapse in V17.

## [V17] - 2026-03-31

### Changed
- **2-phase powerloop** (removed pre-entry) — apex `[-0.3, -0.3, 1.6]` → exit target `[0.9, -0.2, 0.75]`
  - Apex shifted toward Gate 2 for better loop entry
  - Exit target offset toward Gate 4 for racing line (no center detour)
- **gamma**: 0.99 → 0.995 (longer horizon planning)

## [V16] - 2026-03-30

### Changed
- **Speed tuning (stabilized)**: vel_toward_gate 7→8, vel clamp 7→8 (10 caused collapse)
- **Bigger actor network**: [128, 128] → [256, 256] for more policy capacity
- **PPO improvements**: epochs 5→8, entropy_coef 0.0→0.005 (prevents collapse)
- **Lower apex**: `[0, -0.3, 2.0]` → `[0, -0.3, 1.6]`, z threshold 1.5 → 1.3

### Experiment Results
- **3 laps in 17.78s** ✅ — 2.6s faster than V15, powerloop + speed tuning working

## [V15] - 2026-03-30

### Changed
- **Speed tuning**: vel_toward_gate 5→7, orientation -2→-1, smoothness -0.5→-0.2, vel clamp 5→7
- **Tighter powerloop**:
  - Apex lowered: `[0, -0.3, 2.5]` → `[0, -0.3, 2.0]`
  - Pre-entry pulled toward Gate 2: `[0.625, 1.5, 0.75]` → `[0.0, 1.0, 1.2]`
  - Phase 0→1 z threshold: 1.8 → 1.5

### Experiment Results
- **3 laps in 20.38s** ✅ — powerloop + speed tuning, 1.2s faster than V14

## [V14] - 2026-03-30

### Added
- **Gate 3 powerloop: 3-phase guide** — only Gate 3 affected, all others unchanged:
  - Phase 0: target apex `[0, -0.3, 2.5]` (climb) → transitions when z>1.8 or dist<0.8m
  - Phase 1: target pre-entry `[0.625, 1.5, 0.75]` (+y entry side) → transitions when dist<1.0m
  - Phase 2: target gate center (default)
- Recompute `_prev_x_drone_wrt_gate` in NEW gate frame after gate pass (fix wrong-side false positives)
- Reset gate3 `_desired_pos_w` to apex on spawn (consistent `_last_distance_to_goal`)

### Experiment Results
- **3 laps in 21.6s** ✅ — successfully executes powerloop at Gate 3 (flies over double gate)

## [V13] - 2026-03-30

### Changed
- **Velocity-initialized reset** — 20% → 50%, speed 0.5–3.0 m/s, direction follows `initial_yaw` forward (world vx/vy)
- **Current gate normal → body frame** — now both current and next gate normals are in body frame (unified obs frame)
- **Wrong-side termination** — added 5-step cooldown after gate pass (no grace period). Logs `Episode_Termination/wrong_side` to wandb
- **`_last_distance_to_goal` fix** — moved init after `write_root_link_pose_to_sim`, uses 3D distance instead of XY

### Experiment Results
- **3 laps in 19.96s** ✅ — stable, no wrong-side violations, side-detour at double gate

## [V12] - 2026-03-30

### Changed
- **Minimal observation improvement** (26 → 31 dims), reward/reset unchanged from V11:
  - `quat_w` (4) → `gravity_b` (3): gravity vector in body frame
  - Added `next_next_gate_pos_b` (3): 2-gate lookahead
  - Added `next_gate_normal_b` (3): next gate entry direction in body frame
  - Current gate normal kept in world frame (unchanged from V11)


## [V11] - 2026-03-29

### Added
- **Wrong-side gate crossing detection** — if drone passes through the current target gate from the wrong direction (negative→positive x in gate frame), episode is immediately terminated (`_crashed = 200`). Prevents reverse-through exploits at the double-gate (powerloop) structure.
- **Gate index observation** — normalized `current_gate_idx / num_gates` added as 1-dim observation. Policy now knows which gate it's targeting, enabling gate-specific maneuvers (e.g., powerloop at gates 2→3). Total observation dims: 25 → 26.

### Fixed
- **Eval mode parameter initialization** — added `_set_default_dynamics()` that sets physics parameters to nominal config values during eval (`is_train=False`). Previously, guarding DR with `is_train` left all parameters at zero, causing the drone to be unable to fly.

## [V10] - 2026-03-29

### Changed
- **Domain randomization guarded with `is_train`** — DR now only runs during training. During eval (`is_train=False`), the TA's fixed parameters in `quadcopter_env.py` are preserved instead of being overwritten.
- **Improved reset strategy**:
  - Curriculum-based spawn distance: [1.0, 2.0]m early → [1.5, 4.0]m later (ramps over 800 iterations)
  - Wider spawn noise: lateral ±1.0m (was ±0.5), vertical ±0.5m (was ±0.3), yaw ±0.3 rad (was ±0.15)
  - 20% of resets have initial velocity (1-3 m/s) toward gate for momentum handling
  - 10% mid-track spawns between consecutive gates for transition learning

### Added
- **Lap time tracking** during training — detects full-lap completions, logs `Lap/mean_lap_time`, `Lap/min_lap_time`, `Lap/best_lap_time`, and `Lap/laps_completed` to wandb.
- **`eval_race.py`** — self-evaluation script with two phases:
  1. Default parameters, 1 env (mimics TA evaluation conditions)
  2. Domain randomization, 100 envs (stability testing)
  Reports gates passed, lap count, completion rate, and timing for both phases.

## [V9] - 2026-03-28

### Changed
- **Full revert to exact V2 setup** — removed V8's manual observation normalization, re-enabled `empirical_normalization = True` in `rsl_rl_ppo_cfg.py`.
- V8's manual normalization (dividing by 3.0/5.0) did not replicate V2's performance, suggesting `empirical_normalization` uses running mean/std that manual fixed constants can't match.

### Experiment Observations
- V8 manual normalization still underperformed compared to V2. Confirmed that `empirical_normalization = True` is essential for V2-level results.
- ⚠️ `rsl_rl_ppo_cfg.py` is not submitted — need to verify the TA's runner loads the normalizer from checkpoint automatically.

## [V8] - 2026-03-28

### Added
- **Manual observation normalization** in `get_observations()` — divides velocities by 3.0, angular velocities by 5.0, gate positions by 5.0. Quaternion, gate normal, sin/cos yaw error, and actions are already in [-1,1]. This replicates the benefit of `empirical_normalization` but lives in the submitted file (`quadcopter_strategies.py`), so it works regardless of the TA's config.

### Experiment Observations
- V7 showed that removing empirical normalization dramatically reduced all reward metrics (vel: 800→200, progress: 100→30, gate_pass: 200→50). Manual normalization should recover V2-level performance.

## [V7] - 2026-03-28

### Changed
- **Reverted to V2 reward structure** — V2 had the best performance (gate_pass ~200, 3 laps in ~17s). V3-V6 experiments with heading/velocity reward tuning degraded performance.
- **Restored vel_toward_gate** at scale 5.0, orientation at -2.0 (threshold 0.5 rad), death_cost at -10.0.
- **Removed heading reward** — added in V4, didn't improve over V2.
- **Reverted `empirical_normalization`** to `False` — this config file (`rsl_rl_ppo_cfg.py`) is not submitted, so the TA's default would break inference if trained with it enabled.

### Experiment Observations
- Pending training results. Expected to match V2 performance.

## [V6] - 2026-03-28

### Removed
- **vel_toward_gate reward** — caused speed-obsessed behavior leading to crashes at tight turns (gate 3→4).

### Changed
- **Reverted orientation** to V2 values: scale -2.0, threshold 0.5 rad (~30°) — V2 had the best gate completion rate.
- **death_cost**: -10 → -50 — stronger crash avoidance for dangerous gate transitions.
- **heading scale**: 1.0 → 0.5 — gentle directional guidance without dominating.

### Experiment Observations
- Completed 3 laps in ~26s (slower than V2's ~17s).
- Drone flew more cautiously due to higher death_cost (-50) and lower velocity incentive.
- No crashes observed, but significantly slower lap times due to removed velocity reward.
- Conclusion: removing vel_toward_gate made the drone too conservative; the higher death_cost added unnecessary caution.

## [V5] - 2026-03-28

### Changed
- **vel_toward_gate scale**: 5.0 → 2.0 — was dominating the reward signal (~750 contribution), causing the drone to fly too fast and crash at turns.
- **heading scale**: 2.0 → 1.0 — lighter touch to avoid over-constraining maneuvers.

### Experiment Observations
- Still crashing frequently, especially at gate 3→4 (middle-bottom tight turn).
- Passes 2-3 gates then crashes on most episodes. Never completed a full lap.
- wandb: vel_toward_gate ~60, progress ~20 (down from V2's ~100), gate_pass ~40 (down from V2's ~200).
- Conclusion: velocity reward still dominated even at scale 2.0. The heading+velocity combination overconstrained the policy.

## [V4] - 2026-03-27

### Added
- **Heading alignment reward**: `cos(yaw_error)` × 2.0 — encourages the drone to point directly toward the gate, preventing looping/circling approaches.

### Changed
- **Orientation penalty scale**: -0.5 → -1.0 (middle ground between V2's -2.0 and V3's -0.5).
- **Tilt threshold**: 1.0 rad → 0.75 rad (~43°) — prevents flips while still allowing aggressive racing lean.

### Experiment Observations
- Very poor performance — constant crashing. Passes 1-3 gates then dies every episode.
- wandb: vel_toward_gate ~150 (extremely high), gate_pass collapsed to ~40, heading dropped from ~15 to ~0.
- The velocity reward (scale 5.0) dominated the reward signal at ~750 contribution, drowning out gate_pass (200).
- The drone learned to barrel toward gates at max speed but couldn't decelerate for tight turns (gate 3→4).
- Conclusion: vel_toward_gate at scale 5.0 combined with heading reward created a speed-obsessed policy.

## [V3] - 2026-03-27

### Changed
- **Orientation penalty scale** reduced from -2.0 to -0.5 — previous value was limiting top speed by penalizing necessary racing tilt angles.
- **Tilt threshold** raised from 0.5 rad (~30°) to 1.0 rad (~57°) — allows more aggressive lean into turns while still penalizing dangerous attitudes.
- **Reverted `empirical_normalization`** to `False` — this config file is not submitted, so the TA's default would break inference if trained with it enabled.

### Experiment Observations
- Drone completed laps but developed a **flip** at the middle gate transition (gate 3→4).
- After exiting the upper gate, the drone would loop around to the right side before passing through the bottom gate, instead of taking the direct path.
- The reduced orientation penalty allowed excessive tilt angles, leading to tumbling at high-speed transitions.
- Conclusion: orientation penalty of -0.5 was too permissive; the 1.0 rad threshold allowed near-flips.

## [V2] - 2026-03-27

### Added
- **Velocity toward gate reward**: Dot product of drone velocity with direction-to-gate vector, clamped to [-2, 5], scale 5.0.
- **Orientation penalty**: Penalizes excessive roll+pitch when > 0.5 rad (~30°), scale -2.0.
- **Smoothness penalty**: Penalizes action jitter via `||a_t - a_{t-1}||`, scale -0.5.
- **Sin/cos yaw error** observations (2 dims): Continuous heading alignment signal relative to gate direction. Total observation dims: 23 → 25.
- **Domain randomization** via `_randomize_dynamics()`:
  - TWR: ±5%, Aero drag: 0.5x-2.0x, PID kp/ki: ±15%, PID kd: ±30%
  - Ranges match TA evaluation ranges from project description
  - Re-randomized on every episode reset for maximum diversity
- **Empirical observation normalization** enabled in `rsl_rl_ppo_cfg.py`.

### Experiment Observations
- **Best performing version.** Consistently completed 3 laps.
- wandb: vel_toward_gate ~800, progress ~100-125, gate_pass ~200 (saturated), crash ~-0.2 to -0.8.
- Orientation penalty was large (-60) but drone still raced well — stable flight with no flips.
- Smoothness at ~0 — policy naturally learned smooth actions without needing higher penalty.
- The strong orientation penalty (-2.0 at 30° threshold) kept the drone upright and stable through tight turns.

## [V1] - 2026-03-27

### Added
- **Adaptive LR schedule** in `ppo.py`: Computes approximate KL divergence per update. If KL > 2×desired_kl, LR is halved; if KL < desired_kl/2, LR is doubled (clamped to [1e-5, 1e-2]). Fixes training instability in later iterations.
- **23-dim ego-centric observation space** in `quadcopter_strategies.py`:
  - Body-frame linear velocity (3), angular velocity (3), quaternion (4)
  - Current gate position in body frame (3), next gate position in body frame (3)
  - Current gate normal vector (3), previous actions (4)
  - World-frame position intentionally excluded for ego-centric generalization
- **Gate-pass detection** using x-component sign change in gate frame (positive→negative = correct passage) with y/z bounds checking to ensure passage through the gate opening.
- **Reward structure**: progress toward gate (distance reduction), gate-pass bonus (200.0 scale), crash penalty (-1.0 scale), death cost (-10.0).
- **Randomized reset strategy**: Random starting gate, spawn 1.5-3m behind gate with ±0.5m lateral, ±0.3m vertical, and ±0.15 rad yaw noise. Altitude clamped above 0.15m.
- Added `gate_pass_reward_scale` to `train_race.py` reward scales.

### Experiment Observations
- Successfully completed laps with basic reward structure.
- Training was stable thanks to adaptive LR schedule (fixed prior instability in later iterations).
- No velocity or orientation shaping — drone learned to fly toward gates using only progress and gate-pass rewards.

## [Unreleased]

### Added
- Created this changelog file to track project modifications.
- Implemented the PPO algorithm update step in `ppo.py`, including surrogate loss, clipped value loss, entropy bonus, and model optimization.
- Created convenient bash scripts (`scripts/run/train.sh` and `scripts/run/play.sh`) to easily run training and evaluation.

### Changed
- Refactored `compute_returns()` in `rollout_storage.py` to use a GPU-optimized vectorized calculation for generalized advantage estimation (GAE). By computing the TD errors (deltas) in parallel across the sequence, PyTorch kernel launch overhead is minimized during loop iteration.
