# Sim-to-Real Sysid & DR Calibration Plan

**Date:** 2026-04-19
**Scope:** Concrete plan for a dedicated system-identification flight at Pennovation, the data we must log, and how to feed it back into sim-side domain randomization (DR) and the observation pipeline.

**Motivating documents:**
- `docs/changelog.md` (2026-04-19 robustness sweep): systematic **observation bias** is the dominant failure mode, not control mismatch. `obs_vel_bias_0p20` collapsed 3-lap SR from 96.6% to 2.9%; `obs_gate_bias_5cm` dropped it to 92.4%.
- `rosbags_04_15/dynamics_analysis.md`: race rosbags are a poor sysid source (policy-biased, not persistently exciting, bang-bang thrust, controller code changed mid-session). Identified mass ~66 g vs sim's 30 g is a ~2× TWR gap no DR range currently covers.
- `docs/ideas.md`: for a Vicon setup, the big gaps are dynamics + latency, not perception. Residual dynamics and dynamics-only DR are the recommended levers.
- `docs/ros_bags_fields.md`: topic schema for collection.

---

## 0. TL;DR — What This Document Answers

1. **What to log:** every ROS topic already listed in `ros_bags_fields.md`, plus a `/sysid_reference` topic emitted by the sysid controller so we can align commanded vs actual.
2. **How to collect:** one SE3-tracked reference-player flight, split into phases ordered by priority. The same controller supports 1-min, 5-min, 10-min, and 20-min budgets — phases are gated by a time budget, so cutting short only drops low-priority phases, never leaves us without core data.
3. **How to use:** `scripts/sysid/fit_dynamics.py` consumes the bag and produces `sysid_output.yaml`. Sim reads that file to set DR *nominals* and *half-widths*. Obs-bias DR (new) is driven by measured Vicon-pipeline biases, directly addressing the top failure mode in the 2026-04-19 sweep.

---

## 1. On Vicon: Is It Accurate? Does It Lag? Does It Matter?

**Yes to all three, and we must analyze it.** Vicon *cameras* are sub-mm at 100–300 Hz, but the policy never sees raw Vicon. It sees `/observations`, which is the end of this chain:

```
Vicon cameras → Vicon server → network → ROS bridge → state estimator
→ /odom (velocity = numerical diff of position) → controller node
→ policy inference → /ctbr_cmd → Crazyradio → Crazyflie onboard rate controller
```

Known failure modes on this chain:

| Pipeline feature | Typical magnitude | Why it bites the policy |
|---|---|---|
| Pose delivery latency | 10–30 ms | Policy acts on a stale state — trained-in latency must match |
| Velocity-from-diff lag | 1–2 frames (~10–20 ms) | Body-vel in obs lags actual body-vel; creates a **systematic bias** during accel/decel |
| Velocity-from-diff noise | σ ≈ 0.02–0.10 m/s | Zero-mean; policy handles this fine per sweep |
| Vicon drift / calibration | ~1 cm position, possibly a few-degree yaw | Translates to **gate-corner bias** in body frame — sweep says 5 cm → −4.2 pp |
| Frame jitter / dropped frames | occasional | Transient outliers; action-latency DR should subsume |
| Body-frame rotation convention | discrete | If sim and real disagree on axis conventions, everything breaks — check once, then stop worrying |

These map directly onto the 2026-04-19 sweep's headline finding: **systematic observation bias is the top risk**. That is a statement about the Vicon pipeline, not about dynamics. So sysid must spend a material fraction of its time measuring this pipeline, not only the drone.

### Concrete Vicon-pipeline measurements to add

1. **Stationary noise floor** — drone held still on the ground, 60 s. Compute σ of each channel in `/observations`: `lin_vel`, `rot` (flattened), `corners_pos_b_curr/next`. Feeds zero-mean obs-noise DR.
2. **Stationary bias** — same 60 s of ground data. Compute *mean*, not just σ. If the body-frame velocity has a non-zero mean while the drone is physically still, that's the bias the sweep is warning us about. Feeds **obs-bias DR** (new).
3. **Pipeline latency** — compare `header.stamp` of `/pose` vs `/odom` vs `/observations` for the same underlying event. Gives the age of the observation the policy actually saw.
4. **Velocity-from-diff lag** — during a Phase-4 straight-line segment at known constant speed, compare `twist.linear` in `/odom` against a finite-difference of `/pose.position`. The lag between them is the `/odom` estimator's filter delay.
5. **Gate-corner calibration error** — Phase-7 gate flyby (see below). Compare the `corners_pos_b_curr` the policy would see against ground-truth `(gate_pos - drone_pos)` rotated into the body frame using `/odom` orientation. The systematic difference is the gate calibration + Vicon frame offset.

None of these require new hardware — all five fall out of logging `/pose`, `/odom`, `/observations`, and `/tf` during the existing phases.

---

## 2. What to Collect

### 2.1 Topics to record (always on, whole flight)

All topics from `docs/ros_bags_fields.md`, plus one new topic from the sysid controller:

| Topic | Required | Notes |
|---|---|---|
| `/<ns>/odom` | ✅ | 100 Hz, primary state |
| `/<ns>/pose` | ✅ | 120 Hz raw Vicon pose — needed for latency comparison |
| `/<ns>/observations` | ✅ | What the policy would see — needed for obs-bias calibration |
| `/ctbr_cmd` | ✅ | Commanded thrust + body rates |
| `/<ns>/trajectory` | ✅ | SE3 reference setpoints (empty during race but FULL during sysid) |
| `/tf` | ✅ | Frame convention check |
| `/multi_odometry` | ✅ | Gate / obstacle poses if tracked |
| `/sysid_reference` (**new**) | ✅ | What the sysid controller commanded (phase id, commanded pos/vel/accel/rate) |
| `/battery` (if available) | ⚠️ | Cheap to log, useful for TWR-vs-voltage |

If a topic is unavailable at flight time, don't block sysid on it — proceed and note it in the post-flight YAML. Don't silently drop `/observations` (that happened in some 2026-04-15 bags and cost us hardware signatures).

### 2.2 Physical metadata (log by hand once per session)

| Field | How |
|---|---|
| Drone mass (scale, battery installed) | kitchen scale, grams |
| Battery voltage start | Crazyflie sysinfo or log |
| Battery voltage end | same |
| Prop condition | visual, “fresh / worn / chipped” |
| Vicon calibration age | Vicon operator |
| Controller git commit | `git rev-parse HEAD` in `sim2real/` |
| Ambient temperature | optional, motor-τ depends on it |

Write to `<bag_dir>/session_metadata.yaml`.

### 2.3 Targets we want to identify

| Parameter | What it is | Sim DR today | Gap evidence |
|---|---|---|---|
| Mass `m` | Flying mass with battery | 30 g nominal | Bags identify ~66 g (dynamics_analysis §2.2) |
| Max thrust `T_max` | Force at PWM 60 000 | implicit from TWR | Race bags cap at 1.17 N; unknown relation to real force |
| Thrust curve `T(PWM)` | Nonlinear map | approximated linear | Controller code changed quadratic ↔ linear (§3 dyn-analysis) |
| Motor τ | Thrust response lag | 1.0× (disabled) | Drives bang-bang transfer issues |
| Rate bandwidth per axis | Closed-loop rate tracking | PID DR ±15–35 % | Yaw bandwidth ≪ roll/pitch in bags (expected hardware) |
| Linear drag `k_d` | Velocity-proportional drag | DR 0.5–2.0× | Under-damped sim → over-commit thrust |
| Command latency `τ_c` | Pipeline delay | action_latency_max=2 | Never measured open-loop |
| **Obs vel bias** | Bias on `lin_vel_b` vs true vel | **not modeled** | **Top failure mode — sweep** |
| **Obs gate-corner bias** | Systematic offset in corners_b | **not modeled** | **2nd failure mode — sweep** |
| Obs latency (age) | Staleness of `/observations` | implicit | Measured once from bag timestamps |
| Obs noise σ | Vicon + diff noise | approximated | Need fresh measurement after any Vicon recalibration |

---

## 3. How to Collect — Phase-Based Reference Player

The sysid controller tracks a canned reference via the existing SE3 controller. **No RL policy in the loop.** Phases are ordered by priority so that any truncation drops lowest-value phases first.

### 3.1 Design rule for the controller

The controller takes **one arg: `time_budget_seconds`**, and greedily runs phases in priority order until the budget is exhausted. Each phase has a `min_useful_duration` — if the remaining budget is less than that, skip the phase entirely (half a chirp is useless). If time remains after all listed phases, loop Phase 8 (repeat hover + vertical sinusoid) for battery-drift tracking.

This gives us a clean short-form fallback:

| Budget | What we get |
|---|---|
| **1 min** | Phase 0 + 1: mass + hover bias + stationary noise |
| **2 min** | + Phase 2 (vertical chirp short): thrust curve + motor τ |
| **5 min** | + Phase 3 (per-axis rate chirps, shortened): rate bandwidth per axis |
| **10 min** | + Phase 4 (drag) + Phase 6 (latency probe) |
| **20 min** | + Phase 5 (coupled figure-8) + Phase 7 (gate flybys) + Phase 8 (drift tracking) |

Crucially, **ROSbag recording runs for the whole flight**. We record *all* topics no matter how short. Analysis is offline — nothing is thrown away at collection time.

### 3.2 Phases (in priority order)

Each phase is commanded as an SE3 trajectory. Every phase emits `/sysid_reference` with `{phase_id, t_in_phase, commanded_*}`.

**Phase 0 — Static (pre-flight), min 10 s, typical 30 s.**
- Drone on ground, armed but idle. Motors off.
- Logs stationary noise on `/pose`, `/odom`, `/observations`. Establishes Vicon noise floor and any stationary bias.
- **Identifies:** obs noise σ, obs bias nominal, frame convention sanity check.

**Phase 1 — Hover, min 15 s, typical 30 s.**
- Hover at z = 1.0 m, yaw = 0.
- Mean commanded thrust at steady hover → mass (via `m = T_hover / g`).
- Second mass estimate (cross-check with ground scale).
- **Identifies:** mass, hover thrust, vertical stationary obs noise.

**Phase 2 — Vertical excitation, min 30 s, typical 120 s.**
- Sub-phase 2a: sinusoid, 0.5 Hz, ±0.3 m, 10 cycles (20 s).
- Sub-phase 2b: chirp, 0.2 → 3 Hz, ±0.3 m (30 s).
- Sub-phase 2c: step ±0.4 m, hold 2 s, 4 reps (16 s).
- **Identifies:** T(PWM) interior, motor τ, vertical drag `k_{d,z}`.

**Phase 3 — Per-axis rate excitation at hover, min 60 s, typical 240 s.**
- Roll-rate chirp 0.5 → 12 Hz, ±40 °/s, 30 s.
- Pitch-rate chirp same, 30 s.
- Yaw-rate chirp 0.5 → 8 Hz, ±60 °/s, 30 s.
- Doublets at ±20/40/60/80 °/s per axis, ~10 s each.
- **Identifies:** rate bandwidth per axis (Bode −3 dB), rate tracking linearity, yaw asymmetry, rate-command dead time.

**Phase 4 — Horizontal drag, min 60 s, typical 180 s.**
- Straight-line at 1, 2, 3 m/s, in ±x and ±y (8 segments, ~15 s each with decel and turnaround).
- Steady-state per segment: force balance identifies `k_{d,x}`, `k_{d,y}`.
- **Also gives us:** the velocity-from-diff lag measurement (Vicon §1.4) — we know the true steady-state velocity from the reference, so any offset in `/odom.twist` is estimator lag/bias.
- **Identifies:** linear drag, *and* body-vel bias (maps directly onto the sweep's headline risk).

**Phase 5 — Coupled / high-speed, min 30 s, typical 120 s.**
- Figure-8 at 2 m/s then 3 m/s.
- Hard brake: accelerate to 3 m/s, zero-thrust reference, measure deceleration.
- **Identifies:** aero coupling, drag at higher Re, nothing critical but nice-to-have.

**Phase 6 — Latency probe, min 15 s, typical 30 s.**
- From hover, command thrust step +0.2 N for 0.1 s, then back. Repeat 20×.
- Cross-correlate commanded step vs measured `a_z` onset.
- **Identifies:** total command latency τ_c, cleanly, without chirp-autocorrelation ambiguity.

**Phase 7 — Gate flyby, min 30 s, typical 60 s. (NEW, motivated by sweep.)**
- SE3-track straight lines through each gate center at 1.5 m/s, yaw aligned with gate normal. One pass per gate.
- During each pass, log what `/observations.corners_pos_b_curr` reports against ground truth `(gate_pos_world - drone_pos_world)` transformed into body frame using `/odom` orientation.
- **Identifies:** gate-corner systematic bias (the 2nd failure mode). Also sanity-checks gate USD vs real gate dimensions.

**Phase 8 — Drift tracking (fills remaining time).**
- Alternating 30 s hover and 30 s vertical sinusoid.
- Hover thrust trend vs battery voltage → voltage-dependent TWR decay.

### 3.3 Safety & practical notes

- **/race is called from the ground (SOP).** The sysid controller auto-ramps (z-only min-jerk) from the current pose up to `hover_altitude_m` (default 1.0 m) over `takeoff_duration_s` (default 3.0 s) as an implicit Phase −1 before any probe begins. No prior `takeoff` call is required.
- All chirps/steps run against a stable sysid hover center (established by the ramp above). The `max_tilt_deg = 30°` config value is used only for internal sizing; the effective hover envelope before emitting a warning is `max_tilt_deg × 1.5 = 45°`.
- **Tilt does not abort the sysid schedule.** Excursions past the 45° warning threshold emit a rate-limited stdout warning but the scheduler keeps running so data collection is not lost. If the drone is genuinely unsafe, the operator calls `/stop`.
- Velocity limits: 3 m/s horizontal, 1.5 m/s vertical. Rate chirps cap at hardware doublet maxes (±200 °/s roll/pitch, ±150 °/s yaw).
- If Vicon drops out mid-phase, abort the phase cleanly via SE3 land — do not re-enable the rate-chirp after recovery in the same bag.
- Run the full protocol **twice in one session** if possible: once on a fresh battery, once on a drained one. Gives a two-point sample of voltage-dependent thrust decay at almost no extra cost.

### 3.4 What not to try in this session

- No RL policy anywhere (even for "comparison"). Race bags already cover that.
- No gate crashes. The gate flyby must leave ≥10 cm margin; this is calibration, not racing.
- No onboard state estimator swap. Keep everything identical to race-day config except for the reference-player module.

---

## 4. How to Use the Data

### 4.1 Single output artifact — `sysid_output.yaml`

`scripts/sysid/fit_dynamics.py` (new) consumes the bag and writes **one** YAML, schema following `dynamics_analysis.md` §6.5 plus new observation-bias fields:

```yaml
# sysid_output.yaml — identified YYYY-MM-DD on crazy_jirl_b3, battery {v0}→{v1}
mass_kg: {nominal: 0.066, dr_half_width: 0.003}
twr: {nominal: 1.81, dr_half_width: 0.15}
thrust_curve:
  form: quadratic
  coefs: [a2, b1, c0]
  dr_scale_half_width: 0.15
motor_tau_s: {nominal: 0.045, dr_range: [0.030, 0.060]}
command_latency_s: {nominal: 0.035, action_latency_steps_max: 2}
rate_bandwidth_hz:
  roll: {nominal: 18, dr_half_width_pct: 25}
  pitch: {nominal: 17, dr_half_width_pct: 25}
  yaw:   {nominal: 8,  dr_half_width_pct: 25}
drag_kd_Nsm:
  x: {nominal: 0.12, dr_range: [0.5, 2.0]}
  y: {nominal: 0.12, dr_range: [0.5, 2.0]}
  z: {nominal: 0.08, dr_range: [0.5, 2.0]}
obs_noise:
  lin_vel_std_mps: 0.05
  rot_std: 0.01
  gate_corners_std_m: 0.03
obs_bias:                          # NEW — driven by 2026-04-19 robustness sweep
  lin_vel_b_bias_mps: {x: 0.03, y: 0.00, z: 0.00, dr_half_width_mps: 0.10}
  yaw_bias_deg:       {nominal: 0.0, dr_half_width_deg: 3.0}
  gate_corners_bias_m: {nominal: 0.00, dr_half_width_m: 0.04}
obs_latency_s: {nominal: 0.020, dr_half_width_s: 0.010}
```

**Rule that keeps this honest:** every DR half-width is *our uncertainty after sysid*, not *the empirical spread across race policies*. If we can't identify a parameter from one flight, its half-width stays wide; if we identify it tightly, the half-width shrinks.

### 4.2 How sim reads the YAML

1. `QuadcopterEnvCfg` gains a `sysid_yaml: Optional[Path]` field. If set, it overrides nominal mass/TWR/drag/PID/motor-τ and seeds DR half-widths from the YAML. If unset, current hardcoded defaults apply (so today's runs don't break).
2. `CircleQuadcopterStrategy.get_observations` gains **obs-bias DR**: per-episode fixed biases drawn from the ranges in `obs_bias`. These are not re-sampled per step — the whole point (per the sweep) is that *systematic* bias hurts and zero-mean noise doesn't.
3. `obs_latency_s` drives a new step-count delay buffer on the observation side, separate from action latency.

### 4.3 Residual dynamics — only if needed

Per `ideas.md` §1, we *could* train a residual NN on `(s_t, a_t) → s_{t+1}` using sysid bags. Recommendation: **do not lead with this.** A parametric fit plus bias DR is cheaper to debug and less risky for sim2real. Revisit residual learning only if:

- After retraining with the identified DR, sim-predicted trajectories still don't cover real bag state distributions on specific axes, AND
- The residual has structure (i.e., it's not just white noise) that can't be absorbed by widening a Tier-A parameter.

### 4.4 Race bags demoted to validation

After retraining: simulate 1000 trajectories under the new DR, compute state-distribution envelopes for velocity / tilt / body rate, overlay with envelopes from real race bags. Where they disagree:

- Real covers sim → DR is too narrow on that axis, widen and retrain.
- Sim covers real (extra adversarial tail) → fine.
- Zero overlap → deeper dynamics bug, block deploy.

### 4.5 Validation loop

1. Retrain one R1+D1-class checkpoint with the new YAML.
2. Re-run the 13-scenario sim robustness sweep from `docs/changelog.md` (2026-04-19).
   - **Success criterion:** `obs_vel_bias_0p20` no longer collapses to <10 % SR — targeting >60 %. This is the direct test of whether bias-DR worked.
3. Sim2sim break test (`ideas.md` §4): run the new policy with mass +10 %, latency +50 ms; if it still completes laps, ship to real.
4. Real deployment: do a pre-race sysid mini-run (Phase 0+1+6, ~90 s) as a pre-flight check. If identified mass or latency shifts more than the DR half-width, re-check before full race.

---

## 5. Concrete Next Steps (ordered by cost / impact)

1. **Weigh the drone** (5 s). If it's ~66 g, confirms the dominant-gap hypothesis.
2. **Add obs-bias DR to sim in parallel with sysid work.** This is motivated by the 2026-04-19 sweep independent of sysid — we already know the sensitivity. Cheap, doesn't need a flight.
3. **Implement `sysid_mode` in the `sim2real` controller** (SE3 reference player + time-budgeted phase scheduler). Emits `/sysid_reference`.
4. **Implement `scripts/sysid/fit_dynamics.py`** producing `sysid_output.yaml`. Start with just Phase 0, 1, 2, 6 so a 2-minute bag can be parsed end-to-end.
5. **First sysid flight** at Pennovation, 5–10 min budget is sufficient for Tier-A.
6. **Retrain** with identified DR + obs-bias DR.
7. **Re-run robustness sweep.** Require the `obs_vel_bias_0p20` collapse to be gone as go/no-go.
8. **Deploy + validate.** Pre-race 90 s sysid snapshot on the day, to catch drift.

---

## 6. Files (to be created)

| Path | Purpose |
|---|---|
| `../sim2real/src/controller/controller/sysid_controller.py` | SE3 reference player; phases + time-budget gating |
| `../sim2real/src/jirl_bringup/config/sysid_phases.yaml` | Phase definitions (durations, amplitudes, freqs) |
| `scripts/sysid/fit_dynamics.py` | Bag → `sysid_output.yaml` |
| `scripts/sysid/validate_against_race_bags.py` | Post-training envelope overlap check |
| `docs/sysid_output.yaml` | Committed artifact from most recent identification |

---

## 7. Open Questions / Risks

- **Flight-time limit at Pennovation:** unknown. Controller is designed to degrade gracefully; if we only get 2 min we still get mass + hover bias + thrust curve + latency. Do not block on this.
- **Vicon calibration drift:** if the arena is recalibrated between sysid and race, gate-corner bias changes. Run Phase 7 on race day as a 30-s pre-flight sanity check.
- **Battery-dependent thrust:** one session can't separate battery drift from motor τ. Two-battery runs help; if we can't afford two, treat thrust_curve half-width as wider (≥15 %).
- **Controller commit discipline:** `dynamics_analysis.md` §3 showed one mid-session controller change invalidated all prior DR inferences. Every sysid bag must record its controller commit SHA in `session_metadata.yaml`, and retraining must be tied to a specific SHA.
