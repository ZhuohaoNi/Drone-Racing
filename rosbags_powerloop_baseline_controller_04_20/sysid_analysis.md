# Sysid Collection Analysis — 2026-04-20

**Bag:** `rosbags_powerloop_baseline_controller_04_20/sysid/group30_sysid_0.mcap`
**Namespace:** `crazy_jirl_b3`
**Flight duration:** 139.35 s (bag); sysid active until abort at **t = 102.35 s**
**Controller:** `sim2real/src/controller/controller/controller_simple_policy.py` (sysid reference-player drop-in)
**Analysis script:** `sim2real/bin/analyze_sysid_bag.py` (artifacts in `sysid/sysid_analysis/`)

---

## 0. TL;DR

The sysid **scheduler worked** — 23 priority-ordered probes in Round 1 plus two Round-2 probes were cycled through and their phase metadata embedded correctly in `/observations.corners_pos_b_next`.

The **drone did not take off.** Altitude stayed at **0.037–0.06 m** (i.e. on the floor) for the entire flight. The only moment it left the ground was a brief jump to **0.265 m at t ≈ 100 s** during the Round-2 vertical chirp, immediately followed by a 48° tilt and the controller's tilt-abort trigger at t = 102.35 s.

As a result, **most Tier-A parameters from `docs/sysid_plan.md` §2.3 were NOT identified.** What we did learn:

| Plan target | Result |
|---|---|
| Obs noise σ (stationary) | ✅ **identified** — σ ≈ 3–5 mm/s on body velocity channels |
| Obs bias (stationary) | ✅ **identified** — bias < 1 cm/s, effectively zero |
| Mass, hover thrust | ❌ **not identified** — no stable hover ever reached |
| TWR, max thrust | ❌ **not identified** — no sustained airborne flight |
| Motor τ, thrust curve | ❌ **not identified** — no airborne excitation |
| Rate tracking bandwidth per axis | ❌ **unusable** — on-ground friction dominates response |
| Command latency (thrust step) | ❌ **not identified** — drone did not react airborne |
| Linear drag | ❌ **not identified** — no flight |
| Gate-corner bias | ❌ **not executed** — gate flyby aborted as a 1.6 s grounded scoot |

Root cause and next-step recommendations are in §5.

---

## 1. What the Bag Contains

Topic counts from `metadata.yaml` and confirmed by the reader:

| Topic | Count | Rate |
|---|---|---|
| `/crazy_jirl_b3/odom` | 13 917 | ≈ 100 Hz |
| `/crazy_jirl_b3/pose` | 16 722 | ≈ 120 Hz |
| `/crazy_jirl_b3/observations` | 13 516 | ≈ 97 Hz |
| `/ctbr_cmd` | 13 516 | ≈ 97 Hz |
| `/tf` | 16 722 | — |
| `/multi_odometry` | 13 919 | — |
| `/crazy_jirl_b3/trajectory` | **0** | (expected empty — sysid emits via obs instead) |

All required topics from `docs/sysid_plan.md` §2.1 were captured except `/trajectory`, which the sysid controller intentionally skips (the reference state is packed into `corners_pos_b_curr` and the phase metadata into `corners_pos_b_next` — see `controller_simple_policy.py` header).

---

## 2. Phase Scheduling Confirmed Working

`analyze_sysid_bag.py` segments phases by watching the `(dim_code, detail, round, phases_emitted)` signature in `corners_pos_b_next`. 31 blocks (probes + return-to-center + hover-settle) were detected, matching the design in §3.2 of the plan.

Round 1 (survey) ran end-to-end before any abort:

| # | dim | detail | round | t_start | dur | planned | note |
|---:|---|:-:|:-:|---:|---:|---:|---|
| 2 | mass | 1 | 0 | 5.8 | 7.00 | 7.00 | hover_mass_survey |
| 5 | vert | 1 | 0 | 14.7 | 6.00 | 6.00 | vert sin 0.5 Hz |
| 8 | roll | 1 | 0 | 22.5 | 5.00 | 5.00 | LateralOsc y |
| 11 | pitch | 1 | 0 | 29.3 | 4.99 | 5.00 | LateralOsc x |
| 14 | yaw | 1 | 0 | 36.1 | 5.00 | 5.00 | YawChirp |
| 17 | latency | 1 | 0 | 42.9 | 2.40 | 2.40 | ThrustStep ×3 |
| 20 | drag | 1 | 0 | 47.2 | 1.80 | 1.80 | Straight 1 m/s |
| 23 | gate_bias | 1 | 0 | 52.6 | 1.59 | 1.60 | GateFlyby g0 |

Round 2:

| # | dim | detail | round | t_start | dur | planned | note |
|---:|---|:-:|:-:|---:|---:|---:|---|
| 26 | mass | 2 | 0 | 57.7 | 25.00 | 25.00 | hover_mass_medium |
| 29 | vert | 2 | 0 | 84.5 | 17.83 | 20.00 | VerticalChirp — **aborted at 17.83 s** |
| 30 | aux | 0 | 0 | 102.3 | 37.00 | 0.00 | post-abort hover until `/stop` |

The full phase table is saved as `sysid/sysid_analysis/crazy_jirl_b3_sysid_phases.png`. Per-phase analysis JSON lives in `crazy_jirl_b3_sysid_summary.json`.

**Caveat on the abort flag:** I detected the abort from odom tilt (>45°) directly. The `aborted_flag` at `corners_pos_b_next[7]` was never flipped to 1 in this bag because `controller_simple_policy.py` `_build_obs` reads `info.get("aborted")` from the inner phase-info dict, but `SysIDController` stores `self._aborted` at the scheduler level (only surfaced in `out["aborted"]`, not in the inner `info`). This is a minor controller-side bug to fix for next collection so that downstream tools don't have to infer abort from tilt.

---

## 3. What Actually Happened Altitude-wise

Ten-second windows from `/odom`:

```
t[  0, 10): z_mean=0.037  z_max=0.037  tilt_max= 2.0°   (idle on ground)
t[ 10, 20): z_mean=0.037  z_max=0.037  tilt_max= 2.1°   (idle / mass phase)
t[ 20, 30): z_mean=0.037  z_max=0.039  tilt_max= 5.0°   (vert + roll probes)
t[ 30, 40): z_mean=0.037  z_max=0.038  tilt_max= 2.6°   (pitch + yaw probes)
t[ 40, 50): z_mean=0.036  z_max=0.053  tilt_max=22.4°   (latency + drag probes)
t[ 50, 60): z_mean=0.040  z_max=0.057  tilt_max=24.2°   (gate_bias + mass R2)
t[ 60, 70): z_mean=0.047  z_max=0.049  tilt_max=12.3°   (mass R2 continues)
t[ 70, 80): z_mean=0.044  z_max=0.045  tilt_max= 8.9°   (mass R2 continues)
t[ 80, 90): z_mean=0.041  z_max=0.047  tilt_max=12.6°   (vert chirp R2 start)
t[ 90,100): z_mean=0.061  z_max=0.174  tilt_max=29.8°   (chirp — brief lift)
t[100,110): z_mean=0.045  z_max=0.265  tilt_max=48.0°   (flip → ABORT at 102.35)
t[110,120): z_mean=0.015  ...                            (drone fell / tipped)
```

Max altitude across the entire sysid was 26.5 cm; typical altitude was 3–6 cm. Pennovation's floor-to-camera optical center offset is a few cm, so **z ≈ 0.037 m is consistent with the drone sitting on the floor** the whole time.

---

## 4. What We Did and Did Not Identify

### 4.1 Identified — obs-pipeline measurements from the on-ground idle window

Using the first 5 s of the bag (drone armed but not moving) the script produces:

```yaml
obs_lin_vel_b_bias_mean_mps: [-0.000, -0.001,  0.000]   # ≈ 0, no systematic bias
obs_lin_vel_b_noise_std_mps: [ 0.004,  0.003,  0.003]   # 3–5 mm/s std
altitude_mean_m: 0.037
world_speed_max_mps: ≈ 0
```

**Interpretation in the context of `docs/changelog.md` 2026-04-19 sweep:** the body-velocity bias sweep `obs_vel_bias_0p20` used ±0.20 m/s perturbations. What we observe on the real Vicon pipeline, at rest, is **two orders of magnitude smaller** than the sim's worst-case bias. Good news for the zero-mean-noise side (σ ≈ 3 mm/s is negligible vs the 0.02–0.10 m/s range the sweep showed is tolerated). **But this only tells us about the stationary pipeline** — the velocity-from-diff bias that shows up *during accel/decel* (see plan §1 row 2) was not measured here, because the drone never accelerated in flight.

### 4.2 Not identified — dynamics parameters that require flight

| Parameter | Why it failed |
|---|---|
| Mass (hover) | No segment passed the `z > 0.30 m ∧ |vz| < 0.15 m/s ∧ tilt < 12°` gate; the drone never hovered. |
| TWR / max thrust | No sustained airborne flight window; the brief lift at t≈100 s is too short and not steady. |
| Thrust curve, motor τ | The vertical chirp phases ran but produced `vz` std of 3 mm/s (phase 5) — the drone did not move vertically. |
| Linear drag | The "straight-line at 1 m/s" phase showed a peak body speed of 1.33 m/s, but at z ≈ 0.04 m: the drone **skidded on the floor**, not flew. Drag from that segment is aerodynamic + ground friction mixed — unusable. |
| Command latency | Thrust-step reps at z ≈ 0.04 m never triggered a body-frame a_z response. |
| Gate-corner bias | Gate-flyby phase ran for 1.6 s but the drone never got within gate altitude. No useful `corners_pos_b_curr` vs ground-truth comparison. |

### 4.3 Captured but unusable — per-axis rate tracking

The cross-correlation routine *runs* on the roll/pitch/yaw chirp phases and reports a result, but these numbers **should not be used** because the drone was on the ground:

```
phase 8  (y-osc, dim=roll):   pitch_cmd_std=29 dps, actual=9 dps,  lag=0 ms,  score=0.47
phase 11 (x-osc, dim=pitch):  roll_cmd_std=29 dps,  actual=6 dps,  lag=0 ms,  score=0.38
phase 14 (yaw chirp):         yaw_cmd_std=27 dps,   actual=3 dps,  lag=0 ms,  score=0.44
```

Two tells that this is ground-interaction, not airborne rate dynamics:

- Commanded-to-actual amplitude ratios of ~3× (roll/pitch) and ~10× (yaw) are consistent with a grounded drone whose skids are in contact — friction attenuates actual body rates. An airborne Crazyflie would track within ~20 % at these amplitudes.
- The zero-ms "lag" is suspicious: the peak of the cross-correlation is being pulled to zero by the noise floor, not a real closed-loop response.

The global pre-abort cross-correlation in the YAML digest (roll 60 ms, pitch 70 ms, yaw 90 ms, yaw score 0.61) inherits the same problem.

### 4.4 Sanity checks that do pass

- Observation rate (97 Hz) and odom rate (100 Hz) match deployment expectations.
- Phase metadata packing/unpacking round-trips correctly: detail levels, round indices, and `phases_emitted` are monotone within the bag, so the reference player and the analyzer agree on phase structure.
- The `/pose` vs `/odom` position channels track to within the Vicon noise floor (not plotted here — same as prior race bags).

---

## 5. Root Cause and Recommended Next Run

### 5.1 Why the drone never took off

The sysid controller sets its hover center to the drone's pose at the moment `/race` is first called:

```python
# controller_simple_policy.py ~L994
self._sysid.start(
    center_pos=np.asarray(state.get("x", _zeros3())).copy(),
    center_yaw=float(state.get("yaw", 0.0)),
    t_wall=now,
)
```

Combined with the safety-box clamp `(1.2, 1.2, 0.5)` m half-width, if `/race` is called from the ground (center_z ≈ 0.04 m), commanded z is clamped to `[-0.46, +0.54]` m. The hover_mass_survey phase commands x = center — i.e., just z = 0.04 m, which the rotors cannot achieve from a cold start because the SE3 controller integrates up thrust gradually and the drone has just enough lift to scrape the floor, not lift off. Subsequent phases command oscillations around the same low-altitude center and the drone stays grounded.

### 5.2 Fixes before the next sysid flight

Ordered by impact:

1. **Only call `/race` after a stable hover is reached.** This is the intended workflow described in the controller file header (`take off -> hover -> call /race`). Verify on the operator side before re-running.
2. **Alternative: make the sysid controller take off itself** to a nominal hover altitude (e.g. 1.0 m) via the existing SE3 trajectory before starting Phase 0. This removes the operator-sequencing dependency.
3. **Fix the abort-flag plumbing** in `controller_simple_policy.py` so `obs[31]` (`corners_pos_b_next[7]`) actually reflects `self._aborted`. One-line change: in `_update_impl`, merge `out.get("aborted")` into `info` before calling `_build_obs`. Lets downstream tools flag abort without tilt-inference.
4. **Add an altitude precondition** to Phase 0 — if `z < 0.5 m` at phase start, refuse to begin and print a clear message ("hover first, then /race").
5. **Keep the bag regardless.** Offline analysis is cheap, and this bag still tells us the observation pipeline is clean at rest — a non-trivial data point given the 2026-04-19 sweep's concern about obs bias.

### 5.3 What to re-run once the drone actually flies

The minimum follow-up flight that unblocks retraining is the 5-min budget from `sysid_plan.md` §3.1:

- Phase 0 (static) — already OK here
- Phase 1 (hover 15–30 s) — unlocks mass + hover thrust
- Phase 2 (vertical chirp ≥ 30 s) — unlocks motor τ, thrust curve
- Phase 3 (per-axis rate chirps short) — unlocks real rate bandwidth, not the ground-friction version in §4.3
- Phase 6 (latency probe) — unlocks command latency

Drag and gate-bias (Phases 4 & 7) can wait for a longer-budget run. Until the drone hovers, none of the plan's Tier-A parameters are identifiable and `sysid_output.yaml` will stay empty beyond the obs-noise block.

---

## 6. Artifacts

Produced by `sim2real/bin/analyze_sysid_bag.py`:

```
sysid/sysid_analysis/
├── crazy_jirl_b3_sysid_overview.png    # altitude, tilt, thrust, cmd rates, phase ribbon
├── crazy_jirl_b3_sysid_phases.png      # full phase table
├── crazy_jirl_b3_sysid_summary.json    # per-phase + cross-phase analysis
└── crazy_jirl_b3_sysid_digest.yaml     # short digest with "identified" / "gaps" lists
```

To reproduce:

```bash
python sim2real/bin/analyze_sysid_bag.py \
    Drone-Racing/rosbags_powerloop_baseline_controller_04_20/sysid \
    crazy_jirl_b3
```
