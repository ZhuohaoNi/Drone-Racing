# Changelog

All notable changes to this project will be documented in this file.

---

## [Circle-V4] - 2026-04-13

### Stage 2 — Spline Reset Strategy + Expanded DR

V4 adds two major changes: (1) spline-based reset with velocity initialization, inspired by the champion's "Spline Hallway" strategy; (2) expanded domain randomization (mass, motor tau, obs latency).

### Changed

**Reset strategy: spline-based sampling + velocity init (champion's key insight)**

The champion's takeaway: "the secret isn't a complex reward function; it's a robust reset strategy." Previous versions used linear interpolation between gate centers with zero velocity. V4 replaces this with:

1. **Pre-compute periodic cubic spline** through gate positions at init time (scipy CubicSpline, closed loop). 1024 points + tangent vectors pre-sampled and stored as torch tensors. Zero runtime cost.
2. **Sample spawn positions along the spline** using gate-biased weights (Beta(0.5, 0.5) PDF on within-segment fraction — more samples near gates).
3. **Initialize velocity along spline tangent** (0.5–1.5 m/s). The drone starts in a state resembling actual flight, not hovering mid-air.
4. **Yaw aligned to spline tangent** (not pointing at gate center).
5. **Lateral/vertical noise** perpendicular to tangent for diversity (±0.3m lateral, ±0.2m vertical).

Reset mixture (unchanged ratios):
- 50% spline resets (position + velocity along tangent)
- 30% gate-replay resets (preserved from V3 — real observed states with perturbation)
- 20% ground resets (preserved from V3 — z=0.03–0.06m behind first gate)

Configurable via `QuadcopterEnvCfg`:
- `use_spline_reset` (default True, set False for V3-style linear interp)
- `spline_vel_min` / `spline_vel_max` (default 0.5–1.5 m/s)

**Domain randomization (3 new parameters)**

| Parameter | V3 | V4 | Reason |
|-----------|----|----|--------|
| Mass | Fixed | **±10%** | Real battery weight varies; motor degradation changes effective mass |
| Motor time constant (tau_m) | Fixed | **0.5–2.0× nominal** | Real motor response varies with wear, temperature, voltage sag |
| Observation latency | None | **30% chance of 1-step-old obs** | Vicon pipeline has ~10ms latency independent of action delay |

Configurable via `QuadcopterEnvCfg` and sweepable via `ENV_OVERRIDES`:
- `mass_variation` (default 0.1 = ±10%, set 0 to disable)
- `motor_tau_scale_min` / `motor_tau_scale_max` (default 0.5–2.0, set both to 1.0 to disable)
- `obs_latency_prob` (default 0.3, set 0 to disable)

**Implementation notes**
- `_robot_weight` converted from scalar to per-env tensor for per-env mass randomization
- `_tau_m` now randomized per-episode via scale factor (was fixed)
- Observation latency stores previous obs and substitutes with probability during training
- New logging: `Reset/spline_ratio`, `Reset/use_spline` in wandb
- All new DR disabled during evaluation (nominal values used)

**Sweep infrastructure (`scripts/run/sweep.py`)**
- 12 configs total:
  - Config 0: V4 baseline (spline + all DR)
  - Configs 1–6: DR ablations (action delay, obs latency, mass, tau, all latency, minimal)
  - Config 7: No spline (V3-style linear interp for A/B comparison)
  - Config 8: Conservative spline velocity (0.2–0.8 m/s)
  - Configs 9–11: Reward tuning (gate_pass 300, tighter cmd_reg, forgiving death_cost)

### Training
- Recommended comparison: `python scripts/run/sweep.py --config 0 7 --max-iterations 3000`
  - Config 0 = V4 baseline (spline reset)
  - Config 7 = same DR but V3-style linear reset (no spline)
- This isolates the spline reset effect with all other parameters identical.

### Experiment Results
- Pending training and evaluation

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

### Experiment Results
- ⏳ Pending training and zero-shot evaluation at Pennovation

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

### Experiment Results
- ⏳ Pending training and zero-shot evaluation at Pennovation

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
