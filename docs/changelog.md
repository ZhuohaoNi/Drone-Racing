# Changelog

All notable changes to this project will be documented in this file.

---

## [Circle-V5] - 2026-04-15

### Stage 2 — V4 minor variant: higher spline reset velocity

Small tweak to V4: increase spline reset velocity range to better match real flight speed (~2.9 m/s avg observed in V2/V3 real runs).

### Changed (relative to V4)

| Parameter | V4 | V5 | Reason |
|-----------|----|----|--------|
| `spline_vel_min` | 0.5 m/s | **1.0 m/s** | Remove near-hover starts from spline resets (ground resets already cover 0 m/s) |
| `spline_vel_max` | 1.5 m/s | **3.0 m/s** | Match real flight speed; policy needs to learn high-speed gate entry |

All other settings identical to V4 (spline reset, V2 DR ranges, mass ±5%, motor tau 0.7-1.3x, obs noise).

### Training
- `num_envs=8192`, `max_iterations=3000`, `seed=42`

### Experiment Results
- ⏳ Pending real-world evaluation at Pennovation

---

## [Circle-V4] - 2026-04-13

### Stage 2 — V2-based + Spline Reset + Moderate New DR

V2 outperformed V3 in real-world testing. V4 starts from V2 as the base, keeps its wider DR ranges for existing parameters, and selectively adds V4 improvements: spline reset and moderate new DR (mass, motor tau). Action delay and observation latency are removed.

### Changed

**Reset strategy: spline-based sampling + velocity init (kept from prior V4 draft)**

1. **Pre-compute periodic cubic spline** through gate positions at init time (scipy CubicSpline, closed loop). 1024 points + tangent vectors pre-sampled and stored as torch tensors. Zero runtime cost.
2. **Sample spawn positions along the spline** using gate-biased weights (Beta(0.5, 0.5) PDF on within-segment fraction — more samples near gates).
3. **Initialize velocity along spline tangent** (0.5–1.5 m/s). The drone starts in a state resembling actual flight, not hovering mid-air.
4. **Yaw aligned to spline tangent** (not pointing at gate center).
5. **Lateral/vertical noise** perpendicular to tangent for diversity (±0.3m lateral, ±0.2m vertical).

Reset mixture (unchanged ratios):
- 50% spline resets (position + velocity along tangent)
- 30% gate-replay resets (real observed states with perturbation)
- 20% ground resets (z=0.03–0.06m behind first gate)

**Domain randomization (V2 base + 2 new moderate params)**

| Parameter | V2 | V3 | V4 | Reason |
|-----------|----|----|-----|--------|
| TWR | ±15% | ±15% | **±15%** | Unchanged |
| Aero drag | 0.2–3.0× | 0.5–2.0× | **0.2–3.0× (V2)** | V2's wider range performed better in real |
| PID kp/ki | ±35% | ±25% | **±35% (V2)** | V2's wider range performed better in real |
| PID kd | ±50% | ±35% | **±50% (V2)** | V2's wider range performed better in real |
| Mass | Fixed | Fixed | **±5%** | New, moderate — real battery weight varies |
| Motor tau | Fixed | Fixed | **0.7–1.3× nominal** | New, moderate — motor response varies with wear |
| Action delay | None | 0–2 steps | **None** | Removed — not needed |
| Obs latency | None | None | **None** | Removed |

**Observation noise (kept from V3)**
- lin_vel_b: σ=0.05 m/s, rot_matrix: σ=0.01, gate_corners: σ=0.02m, prev_action: no noise

**PPO hyperparameters (reverted to V2)**

| Parameter | V2 | V3 | V4 |
|-----------|----|----|-----|
| `num_mini_batches` | 4 | 8 | **4 (V2)** |

### Training
- `num_envs=8192`, `max_iterations=3000`, `seed=42`

### Experiment Results
- ⏳ Pending real-world evaluation at Pennovation

---

## [Circle-V4-Conservative] - 2026-04-14

### Stage 2 — Ultra-Conservative Fallback for Real Deployment

V4 is a conservative variant of V3 designed as a fallback at Pennovation. Same obs noise, action latency, and DR as V3, but with stronger command smoothness penalties and less time pressure. The idea: if V3 is too aggressive in real, V4 should fly slower but survive.

Test order at Pennovation: **V4 → V3 → V2** (most conservative first).

### Changed (relative to V3)

| Reward | V3 | V4 | Reason |
|--------|----|----|--------|
| `cmd_reg_rp_scale` | -1.0 | **-2.0** | 2x stronger roll/pitch rate penalty for smoother flight |
| `cmd_reg_yaw_scale` | -0.5 | **-1.0** | 2x stronger yaw rate penalty |
| `lap_incomplete_penalty` | -0.05 | **-0.02** | Less time pressure — fly slow and safe |

All other settings (obs noise, action latency, DR ranges, network, num_mini_batches) inherited from V3.

### Training
- `num_envs=8192`, `max_iterations=3000`, `seed=42`
- `run_name=circle-v4-conservative`
- Script: `scripts/run/train_circle_v4.sh`

### Experiment Results — Real-world @ Pennovation 2026-04-15 (bag: `group30_cyclev4`)

| Metric | Value |
|--------|-------|
| Total race time (first gate → last) | 7.624 s |
| Takeoff → 3 laps done | 9.066 s |
| Best lap | 2.050 s |
| Mean lap | 2.133 s |
| Lap std (consistency) | 0.090 s |
| Path length (race) | 22.56 m |
| Mean / max speed | 2.96 / 4.05 m/s |
| Mean / max tilt | 33.7 / 51.2 deg |
| Mean / max body rate | 98.37 / 243.48 rad/s |
| Mean / max thrust | 0.62 / 1.17 N |
| Mean gate clearance | 0.436 m |

Per-lap breakdown:

| Lap | Time (s) | Path (m) | v_avg | v_max | br_avg | br_max | gate_d |
|-----|----------|----------|-------|-------|--------|--------|--------|
| 1 | 2.050 | 6.07 | 2.97 | 3.64 | 102.56 | 224.02 | 0.377 |
| 2 | 2.257 | 6.55 | 2.90 | 4.05 | 94.46 | 238.09 | 0.423 |
| 3 | 2.091 | 5.97 | 2.87 | 3.81 | 101.60 | 207.14 | 0.509 |

**Interpretation:** Strongest cmd_reg penalty produced the lowest mean body rate (98 rad/s) across all tested versions, confirming the conservative penalty worked. However, lap time (2.133s mean) and consistency (std=0.090s) were the worst — the reduced `lap_incomplete_penalty` let the policy "linger" between gates. Slower but not more consistent than V2/V3.

---

## [Circle-V3] - 2026-04-14

### Stage 2 — Training Stability & DR Calibration

V2 achieved 99.9% SR in sim with 3-param DR eval, but training curves showed large periodic oscillations. Two causes identified: (1) DR ranges too extreme (aero 0.2-3.0x, kd ±50%) creating impossible dynamics that added noise; (2) too few mini-batches for sparse reward variance.

### Changed

**Domain randomization (tightened to realistic ranges)**

| Parameter | V2 | V3 | Reason |
|-----------|----|----|--------|
| TWR | ±15% | ±15% | Unchanged, reasonable for Crazyflie |
| Aero drag | 0.2–3.0× | **0.5–2.0×** | 0.2× (near-zero drag) unrealistic; 3.0× too extreme |
| PID kp/ki | ±35% | **±25%** | Tighter, real Crazyflie PID won't deviate this far |
| PID kd | ±50% | **±35%** | kd ±50% caused extreme angular rate behavior |

**Training hyperparameters**

| Parameter | V2 | V3 | Reason |
|-----------|----|----|--------|
| `num_mini_batches` | 4 | **8** | More mini-batches = smaller batch = more stable gradient estimates |
| `max_iterations` | 5000 | **3000** | V2 curves plateau by ~1500 iter; remaining was wasted compute |

**Observation noise (new, Swift-inspired)**
- Added Gaussian noise to observations during training only:
  - `lin_vel_b`: σ=0.05 m/s (Vicon velocity from numerical differentiation is noisy)
  - `rot_matrix`: σ=0.01 (small attitude measurement noise)
  - `gate_corners_b`: σ=0.02m (gate position calibration error)
  - `prev_action`: no noise (known exactly)
- Forces policy to be robust to sensor imperfections rather than relying on perfect state

**Action latency randomization (new, Swift-inspired)**
- Each env gets a random action delay of 0–2 policy steps during training
- Simulates real communication latency: Vicon → compute → Crazyradio → Crazyflie (~10–40ms total, policy runs at 50Hz = 20ms/step)
- Delay re-randomized on each episode reset
- Disabled during eval (delay=0)

### Training
- `num_envs=8192`, `max_iterations=3000`, `seed=42`
- `run_name=circle-v3-tighter-dr`

### Experiment Results — Real-world @ Pennovation 2026-04-15 (bag: `group30_cyclev3`)

| Metric | Value |
|--------|-------|
| Total race time (first gate → last) | 6.750 s |
| Takeoff → 3 laps done | 7.359 s |
| Best lap | 1.867 s |
| Mean lap | 1.884 s |
| Lap std (consistency) | 0.013 s |
| Path length (race) | 19.95 m |
| Mean / max speed | 2.96 / 4.01 m/s |
| Mean / max tilt | 38.0 / 48.7 deg |
| Mean / max body rate | 159.66 / 234.83 rad/s |
| Mean / max thrust | 0.66 / 1.17 N |
| Mean gate clearance | 0.496 m |

Per-lap breakdown:

| Lap | Time (s) | Path (m) | v_avg | v_max | br_avg | br_max | gate_d |
|-----|----------|----------|-------|-------|--------|--------|--------|
| 1 | 1.900 | 5.86 | 3.10 | 4.01 | 155.60 | 222.21 | 0.489 |
| 2 | 1.867 | 5.34 | 2.87 | 3.14 | 165.57 | 234.83 | 0.508 |
| 3 | 1.883 | 5.40 | 2.88 | 3.25 | 160.16 | 233.01 | 0.491 |

**Interpretation:** Very consistent laps (std=0.013s). Higher body rate than V2 (159 vs 122 rad/s avg) suggests tighter DR didn't help smoothness in real. Faster overall than V4 but slightly slower than V2. Gate clearance slightly worse than V2 (0.496 vs 0.443m).

---

## [Circle-V2] - 2026-04-13

### Stage 2 — Circle Track Sparse Reward Redesign (Sim2Real)

Major reward and training overhaul based on insights from two papers:
- Kaufmann et al. "Champion-level drone racing using deep RL" (Nature, 2023)
- Pasumarti et al. "Agile Flight Emerges from Multi-Agent Competitive Racing" (2026)

Circle-V1 failed to fly in real. Two likely causes were identified: (1) dense reward shaping (progress, vel_toward_gate, orientation) encouraged sim-specific trajectories; (2) the training reset distribution did not match real deployment, which started from the ground rather than from mid-air gate approach states.

**V2 first run collapsed** at ~200 iterations: gate_pass=10.0 was too weak vs continuous negative terms (cmd_reg, crash), causing the policy to prefer hovering over attempting gates. Fixed by scaling gate_pass to 200.0, adding death_cost back, replacing lap_complete bonus with per-step lap_incomplete penalty, and increasing cmd_reg to match reference values.

### Changed

**Reward structure (dense → sparse with strong gate signal)**

| Reward | Circle-V1 | Circle-V2 | Reason |
|--------|-----------|-----------|--------|
| Progress (d_t-1 − d_t) | +20.0 | **removed** | Dense shaping constrains policy to gate-to-gate line; hurts sim2real (Pasumarti Fig.5) |
| Gate pass | +150.0 | **+200.0** | Strong sparse signal; must dominate continuous negative terms (ref uses 500, conservative for real) |
| Lap incomplete penalty | (none) | **-0.05/step** | Constant per-step cost pushes policy to complete laps, not hover |
| Vel toward gate | +1.0 | **removed** | Dense shaping; prescribes *how* to fly rather than *what* to achieve |
| Orientation penalty | -3.0 | **removed** | Over-constrains policy; let RL discover stable attitudes |
| Smoothness penalty | -1.5 | **removed** | Replaced by cmd_reg on body rates |
| Cmd regularization (rp) | (none) | **-1.0 × (ω_roll² + ω_pitch²)** | Body-rate reg for real-world smoothness (ref uses -1.0) |
| Cmd regularization (yaw) | (none) | **-0.5 × ω_yaw²** | Yaw rate reg (ref uses -0.5) |
| Crash (contact) | -5.0 | **-0.1/step** | Per-step contact penalty |
| Death cost | -50.0 | **-100.0** | Strong death penalty prevents aggressive behavior in real |

**Observation space (36-dim → 40-dim)**
- Added previous action (4D) to observation: `[thrust, roll_rate, pitch_rate, yaw_rate]`
- Helps policy infer current dynamics state (Swift uses previous action in obs)
- Layout: `lin_vel_b(3) | rot_matrix(9) | curr_gate_corners_b(12) | next_gate_corners_b(12) | prev_action(4)`

**Reset strategy (mixed sim2real distribution)**
- Replaced the previous air-only reset with a mixture of:
  - **Ground / near-ground starts (20%)**: sample behind the first gate with `z ∈ [0.03, 0.06] m` to match real deployment, where the quad starts on the ground
  - **Gate-state replay resets (up to 30% when buffer available)**: cache states observed after successful gate passes and replay them with small perturbations, following Swift's "bounded perturbation around states observed when passing gates"
  - **Gate-biased geometric resets (fallback coverage)**: Beta(0.5, 0.5) interpolation between consecutive gates, with more mass near gates and less in the middle of the segment
- Lateral noise for geometric resets: ±1.5m → ±0.3m (tighter, along perpendicular to gate-to-gate segment)
- Vertical noise for geometric resets: ±0.5m → ±0.2m
- Replay resets preserve `prev_action` and initialize motor state consistently to reduce first-step transients
- Spawn target remains the next gate in the segment for air resets; ground resets target the first gate

**Network architecture**
- Actor: `[512, 512, 256, 128]` (unchanged)
- Critic: `[512, 256, 128, 128]` → `[512, 512, 256, 256, 128, 128]` (deeper critic improves value estimation, per Pasumarti et al.)

**Training hyperparameters**
- `num_steps_per_env`: 24 → 64 (sparse rewards need longer rollouts to see gate-pass events)
- `learning_rate`: 1e-4 (unchanged; 3e-4 caused instability in first V2 run)
- `max_iterations`: 2000 → 5000 (sparse rewards need more training)
- `entropy_coef`: 0.01 (unchanged, helps exploration with sparse rewards)

### Training
- `num_envs=16384`, `max_iterations=5000`, `seed=42`
- `run_name=circle-v2-sparse`

### Experiment Results — Real-world @ Pennovation 2026-04-15 (bag: `group30_cyclev2`)

| Metric | Value |
|--------|-------|
| Total race time (first gate → last) | 6.959 s |
| Takeoff → 3 laps done | 7.459 s |
| Best lap | 1.866 s |
| Mean lap | 1.878 s |
| Lap std (consistency) | 0.011 s |
| Path length (race) | 20.34 m |
| Mean / max speed | 2.93 / 3.73 m/s |
| Mean / max tilt | 36.3 / 48.6 deg |
| Mean / max body rate | 121.67 / 243.88 rad/s |
| Mean / max thrust | 0.75 / 1.17 N |
| Mean gate clearance | 0.443 m |

Per-lap breakdown:

| Lap | Time (s) | Path (m) | v_avg | v_max | br_avg | br_max | gate_d |
|-----|----------|----------|-------|-------|--------|--------|--------|
| 1 | 1.866 | 5.72 | 3.08 | 3.73 | 118.60 | 210.67 | 0.383 |
| 2 | 1.875 | 5.39 | 2.89 | 3.12 | 113.66 | 219.98 | 0.482 |
| 3 | 1.892 | 5.38 | 2.86 | 3.13 | 109.76 | 216.73 | 0.463 |

**Interpretation:** Best consistency across all versions (std=0.011s). Lowest body rate (122 rad/s avg) — smoothest control. Best gate clearance (0.443m). Slightly slower total race time than V3 (6.959 vs 6.750s) but more reliable lap-to-lap. **V2 is the strongest baseline for real-world deployment.**

### Interpretation Guide
Key metrics to watch during training:
- **Episode_Reward/gate_pass**: should increase steadily — primary learning signal
- **Episode_Reward/lap_incomplete**: constant negative; offsets when gate_pass grows
- **Episode_Reward/cmd_reg**: should be moderately negative; if too large policy is too jerky
- **Reset/ground_ratio**: should be close to 0.20 during training
- **Reset/replay_ratio**: should rise above 0 once successful gate passes start populating the replay buffer
- **Lap/success_rate_3lap**: target > 80% before deploying to real
- **Lap/mean_lap_time**: decreasing = policy getting faster
- **Lap/best_lap_time**: tracks peak performance
- If gate_pass reward stagnates at 0 for > 500 iterations: rollout may be too short or spawn positions too far from gates

---

## [Circle-V1] - 2026-04-09 *(run: `2026-04-09_01-24-21`)*

### Stage 2 — Circle Track Zero-Shot (Sim2Real)

First circle track policy trained for real-world deployment at Pennovation.

### Changed

**Observation space (31-dim → 36-dim)**
- Replaced Stage 1 obs (vel, gravity, gate centers, normals, yaw error, gate index, prev actions) with a 36-dim obs that exactly matches `controller_simple_policy.py` in the sim2real repo:
  - Body linear velocity (3)
  - Rotation matrix body→world, flattened (9)
  - Current gate 4 corners in body frame (12)
  - Next gate 4 corners in body frame (12)
- Gate corners use identical math to the real controller: `corners_w = local_square @ R_gate.T + gate_pos`, then `corners_b = (corners_w - drone_pos) @ R_body`

**Reward structure (safety-focused for sim2real)**

| Reward | Stage 1 | Circle-V1 | Reason |
|--------|---------|-----------|--------|
| Progress | +50.0 | +20.0 | Less urgency |
| Gate pass | +200.0 | +150.0 | Reduced |
| Vel toward gate | +8.0 | +1.0 | Slow and smooth |
| Orientation | -1.0 | -3.0 | Stay level in real |
| Smoothness | -0.2 | -1.5 | Jerky commands → oscillation in real |
| Crash | -1.0 | -5.0 | Zero tolerance |
| Death cost | -10.0 | -50.0 | Strong crash avoidance |

- Removed powerloop guide (not applicable to circle track)
- Removed racing-line velocity blend (speed optimization, not needed for safety)

**Domain randomization (wider for sim2real)**

| Parameter | Stage 1 (V30) | Circle-V1 |
|-----------|--------------|-----------|
| TWR | ±8% | ±15% |
| Aero drag | 0.3–2.5× | 0.2–3.0× |
| PID kp/ki | ±25% | ±35% |
| PID kd | ±40% | ±50% |

**Reset strategy**
- Removed velocity-initialized spawns (always start from hover for safety)
- Removed ground spawns (SE3 controller handles takeoff in real)
- Wider lateral noise: ±1.0m → ±1.5m
- Mid-track spawn rate: 40% → 50%

**Network architecture**
- Actor: `[256, 256]` → `[512, 512, 256, 128]` (matches `controller_simple_policy.py`)
- `empirical_normalization`: True → False (normalizer not exported to real controller)
- `entropy_coef`: 0.005 → 0.01
- `learning_rate`: 5e-4 → 1e-4

**Track**
- Added `'circle'` track (4 gates, matches Pennovation layout)
- Set default `track_name = 'circle'`
- Implemented as `CircleQuadcopterStrategy` alongside existing `DefaultQuadcopterStrategy`

### Training
- `num_envs=16384`, `max_iterations=2000`, `seed=42`
- `run_name=` (unnamed), `run_dir=2026-04-09_01-24-21`

### Experiment Results
- ⏳ Pending zero-shot evaluation at Pennovation
