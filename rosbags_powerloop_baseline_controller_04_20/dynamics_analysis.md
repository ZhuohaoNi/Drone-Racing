# Powerloop Dynamics Analysis — Cross-Track Confirmation of the Circle-Track Sysid Story

**Date:** 2026-04-21
**Scope:** Three powerloop-track rosbags recorded on the race arena on 2026-04-20 with the pre-race baseline controller, compared against the 2026-04-08 / 2026-04-15 circle-track bags analyzed in `../rosbags_04_15/dynamics_analysis.md`. The goal is to check whether the sim-real gaps we previously saw on the circle — consistent hardware signatures alongside policy-driven metric variance — also show up on the harder powerloop track, and to pull out what transfers to sysid / DR tuning.

**Bags:**

| Bag | Duration | Complete laps (ordered) | Best lap | Notes |
|---|---:|---:|---:|---|
| `powerloop-fixed-baseline` | 21.8 s | 0 | — | Baseline circle controller ported to powerloop; fails after 1.4 pseudo-laps, 4 valid passes, 5 sequence breaks. |
| `powerloop-r1d1-gate3mask` | 46.8 s | 4 | 7.36 s | r1d1 policy with a mask on gate 3; finishes 4 ordered laps. |
| `powerloop-r1d1-gate3mask-fullswitch` | 61.0 s | 5 | 7.50 s | Same policy with "full switch" gate-progression logic; our longest clean run to date. |

**TL;DR — the hardware story is the same as on the circle, and the 2026-04-16 calibrated PWM map is *still not in these bags*.** Every hardware-level signature we identified on the circle (max thrust 1.17 N, identified mass ~64 g, ~60–90 ms rate-tracking lag, linear PWM↔thrust map) reappears unchanged on the powerloop track. Every policy-driven quantity (mean thrust, tilt extrema, commanded-rate envelope) varies as much *between the three powerloop bags* as it did across the circle bags, even though they were recorded back-to-back on the same drone. This is a second independent confirmation that **race bags cannot serve as the DR generator** and that the identified-mass / sim-mass mismatch (~64 g real vs 30 g sim) is the dominant sim-real gap. The existing `sysid/` bag recorded in the same session is the right input for DR; this race data is the right input for *validation* only.

---

## TL;DR — plain-language version

### Does the powerloop data agree with the circle data on hardware?

**Yes — on everything that should be a hardware constant, the powerloop bags and the circle bags match.**

| Quantity | Circle (04-15) | Powerloop (04-20) | Same? |
|---|---|---|---|
| Max thrust at PWM = 60000 | 1.170 ± 0.003 N | 1.174 N (all 3 bags) | ✅ |
| PWM→Newton controller formula | `2.4e-5·PWM − 0.235` (linear) on pre-04-16 bags | `2.34–2.35e-5·PWM − 0.24` (linear, quadratic coef ≈ 0) | ✅ — but this means the calibrated quadratic map from the 2026-04-16 changelog entry **is still not being used** in the 04-20 baseline controller |
| Identified drone mass | 60.5 – 69.5 g (median 65.9 g, σ 3.6 g) | 60.8 – 64.7 g (median ~64 g) | ✅ |
| Rate-tracking lag (roll / pitch) | 50 – 70 ms | 60 – 90 ms on the two r1d1 bags | ✅ |
| Max measured body rate | physically reasonable | 10315 °/s (single-sample peaks, clearly recovery-transient artefacts) | ⚠️ new: powerloop exposes post-crash spins we didn't see on circle |

### Does the powerloop data agree with the circle data on policy-driven metrics?

**Also yes — in the bad way.** The metrics that varied by 50–100 % across circle bags flown on the same hardware on the same day also vary across these three powerloop bags.

| Quantity | Circle bag spread | Powerloop bag spread |
|---|---|---|
| Mean commanded thrust | 0.65 – 0.77 N (18 % spread) | 0.52 – 0.71 N (**36 % spread**) |
| Mean body-rate RMS-ish | 81 – 180 °/s | 37 – 80 °/s (means), but max per-axis spans 299 – 10315 °/s |
| Mean tilt | 29° – 39° | 32.0° – 32.5° |
| Max tilt | 39° – 57° | 49° – 137° (**much wider** — driven by crash-recovery inversions and by the loop itself) |
| Bang-bang thrust fraction | 66 – 82 % | 64 – 77 % |

So the *shape* of the policy-vs-hardware confusion from `rosbags_04_15/dynamics_analysis.md` reproduces here: if you tuned DR ranges to any one of these three bags you would converge to that one policy's habits. The three bags disagree about the mean thrust by a third, about the max tilt by 3×, and about the peak commanded rate by ~4×. These are the same drone, same controller, same track, same day, over a 1-hour window.

### What *is* different on the powerloop vs the circle

1. **Vertical dynamics are genuinely exercised.** The powerloop's gate 2 inverted-loop segment drives pitch to ±80° and roll past 140° on every clean lap (r1d1 bags). The circle never saw more than ±57°. This is a real track effect, not a policy artefact, and the sim must cover it (currently `CircleQuadcopterStrategy`'s euler DR clips well below ±60°).
2. **Recovery transients are larger.** Both r1d1 bags show isolated angular-velocity peaks of 8 000 – 10 000 °/s. Physically these are post-contact spins after brushed gate hits; they are single-sample artefacts on a 100 Hz grid (i.e. a finite-difference numerical spike), but they indicate the real drone does make contact and recover frequently — the sim's zero-tolerance crash model can't simulate this mode.
3. **The fixed-baseline bag self-destructs.** The circle-tuned "fixed baseline" policy deployed directly onto the powerloop made only 1.4 pseudo-laps and zero ordered laps before breaking sequence — zero-shot circle→powerloop transfer does not work even in the real pipeline. This is useful negative evidence: the *sim* side of the gap is not the only thing blocking powerloop.
4. **The gate-3 mask + full-switch logic matter more than dynamics.** `r1d1-gate3mask-fullswitch` does 5 ordered laps with essentially the same policy dynamics as `r1d1-gate3mask`; the difference is in the *gate-progression state machine*, not the PID loop. Don't try to fix this with DR.

### The dominant sim-real gap, re-confirmed

On the circle we argued the ~65 g identified mass vs 30 g sim nominal is the single biggest miscalibration. The powerloop bags pin median identified mass at **63.9 g across three policies** (60.8 / 64.7 / 64.3 g). Same drone on a second, harder track, with recovery events included — same answer. Nothing ±15 % around the 30 g sim nominal can bridge a 2× mass gap. Everything downstream (TWR, hover thrust, commanded-rate feasibility, bang-bang optimality) inherits this error.

### Concrete first step (unchanged from the circle doc)

The `rosbags_powerloop_baseline_controller_04_20/sysid/` directory contains a dedicated sysid bag already recorded in this same session — exactly what the circle analysis recommended as the next action. The next step is to run `fit_dynamics.py` (to be written, see §6 of `rosbags_04_15/dynamics_analysis.md`) against **that bag**, not against any of the three race bags, to seed DR nominal + half-width values. The race bags stay in a validation role: does sim-DR-generated state coverage include what we see here?

---

## 1. Methodology

Identical pipeline to the circle analysis. All three bags were processed with `sim2real/bin/analyze_rosbag.py`, which emits `<ns>_statistics.json`, `<ns>_ctbr_full.json`, and `<ns>_odom_full.json` per bag. For each bag we then computed:

- **Policy-independent hardware signatures** — expected to match across bags / tracks:
    - Linear and quadratic fit of `thrust_n` vs `thrust_pwm` on the `/ctbr_cmd` messages.
    - Max thrust at PWM ≈ 60000.
    - Identified mass via the vertical force balance `m = cos(roll)·cos(pitch)·thrust_n / (a_z_world + g)`, filtered to `z > 0.3 m`, `thrust > 0.3 N`, `cos(roll)·cos(pitch) > 0.5`, `|a_z_world| < 5 m/s²`, and clipped to the physically plausible 12–80 g range.
    - Cross-correlation lag per axis between commanded body rate and measured body-frame angular velocity, on a 100 Hz resampled grid with ±400 ms search window.
- **Policy-dependent metrics**:
    - Mean / max commanded thrust (N) and PWM.
    - Bang-bang fraction (thrust < 0.1 N or > 1.0 N).
    - Mean / max body-frame speed.
    - Mean (|abs|) and max commanded body rate per axis.
    - Euler-angle ranges.

Track layout used: `--track powerloop` (real mocap-frame coordinates from `analyze_rosbag.py`'s preset). All three bags use the `crazy_jirl_b3` namespace.

---

## 2. What the Hardware Should Be Saying (and Is) — Powerloop Edition

### 2.1 Max thrust at saturation — identical to circle

| Bag | Max `thrust_n` at PWM = 60000 |
|---|---:|
| powerloop-fixed-baseline | **1.1738 N** |
| powerloop-r1d1-gate3mask | **1.1743 N** |
| powerloop-r1d1-gate3mask-fullswitch | **1.1743 N** |

Same 1.17 N ceiling that every circle bag produced. This confirms the controller's PWM→N table has not changed for these flights and matches the 2026-04-15 Race-1 and Race-2 (zhuohao_v7) bags.

### 2.2 PWM→Newton mapping — still the linear formula

Fitting `thrust_n = a·PWM² + b·PWM + c` to the `/ctbr_cmd` messages:

| Bag | Linear fit (`b·PWM + c`) | Quadratic coef `a` |
|---|---|---:|
| powerloop-fixed-baseline | `2.349e-5·PWM − 0.2349` | −6.6 × 10⁻¹⁵ (numerical zero) |
| powerloop-r1d1-gate3mask | `2.351e-5·PWM − 0.2381` | 2.4 × 10⁻¹¹ |
| powerloop-r1d1-gate3mask-fullswitch | `2.340e-5·PWM − 0.2712` | 2.0 × 10⁻¹⁰ |

All three bags are **linear** — the quadratic coefficient is effectively zero. This matches Race-1 bags and Zhuohao's v7 bag in `rosbags_04_15/dynamics_analysis.md`, but it does **not** match Bruce's 2026-04-16 "fixed sim2real baseline" (`bruce_v2base+dr` and `bruce_v5spline+dr`), which used a calibrated quadratic. In other words:

> **The 04-20 baseline-controller race bags use the same linear PWM map as the pre-04-16 bags. The calibrated quadratic thrust map from the 2026-04-16 changelog entry is *not active* in these flights.**

This is important because the circle analysis concluded that the linear vs quadratic controller code change was a sim-real gap larger than most DR ranges, and therefore race bags from before and after the 04-16 commit cannot be pooled. These 04-20 bags live in the *pre-fix* regime even though they were recorded later. If the race on 04-28 is still on the baseline controller, DR calibration built against the quadratic map would be aimed at the wrong deployment pipeline.

### 2.3 Identified mass — same ~64 g as on the circle

Median identified mass per bag (filtered as in §1):

| Bag | Identified mass |
|---|---:|
| powerloop-fixed-baseline | 60.8 g |
| powerloop-r1d1-gate3mask | 64.7 g |
| powerloop-r1d1-gate3mask-fullswitch | 64.3 g |

**Mean: 63.3 g.** Compare: circle bags gave 65.9 ± 3.6 g. The ~5 % cross-bag spread is again explained by battery / short segment / filter sensitivity; the central answer reproduces. Two independent tracks, two independent sessions, 11 bags total, and every one of them identifies the drone at 60 – 70 g.

The sim trains against a 30 g nominal. **The single clearest action item from this entire document is: weigh the drone and reset the sim-side mass.** A 2× mass error cannot be absorbed by any realistic DR schedule — it changes TWR, hover thrust, commanded-rate response, and the set of policies the optimizer considers locally optimal.

### 2.4 Rate-tracking latency — consistent on the working bags

Peak cross-correlation lag between commanded and actual body rate (100 Hz grid, ±400 ms search):

| Bag | Roll lag | Pitch lag | Yaw lag |
|---|---:|---:|---:|
| powerloop-fixed-baseline | −70 ms | **−270 ms** | 120 ms |
| powerloop-r1d1-gate3mask | −90 ms | 100 ms | 50 ms |
| powerloop-r1d1-gate3mask-fullswitch | 80 ms | 70 ms | 60 ms |

The two r1d1 bags give lags in the same 50–90 ms band as the circle bags. The fixed-baseline bag is dominated by a few seconds of near-tumble right before it bailed, so its cross-correlation is noise-contaminated (the −270 ms pitch lag is non-physical for this hardware) — exactly the closed-loop-on-degenerate-data failure mode we flagged for v3 in the circle analysis. This reinforces the point from §6 of the circle doc: **rate-tracking lag cannot be reliably identified from closed-loop race data**; we need an open-loop chirp.

### 2.5 Summary — hardware signatures across 11 bags, 2 tracks, 2 sessions

| Signature | Circle (8 bags) | Powerloop (3 bags) |
|---|---|---|
| Max thrust @ PWM 60000 | 1.170 ± 0.003 N | 1.174 N |
| Identified mass median | 65.9 g | 63.3 g |
| PWM map (baseline controller) | linear, `≈ 2.4e-5·PWM − 0.235` | linear, `≈ 2.35e-5·PWM − 0.24` |
| Rate lag, r/p/y (clean bags) | 50–70 / 60–90 / 50–70 ms | 60–90 / 70–100 / 50–60 ms |

The hardware speaks with one voice across everything we have.

---

## 3. What the Rosbags Say About the Policies (Not the Hardware)

| Bag | mean_thrust | mean_PWM | mean speed | max speed | mean tilt | max tilt | roll cmd max | pitch cmd max | yaw cmd max | bang-bang % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| powerloop-fixed-baseline | 0.520 N | 32138 | 2.00 m/s | 5.74 m/s | — | 49.3° | **100 °/s** | **100 °/s** | **200 °/s** | 68.3 % |
| powerloop-r1d1-gate3mask | 0.705 N | 40141 | 2.71 m/s | 5.30 m/s | 32.0° | 108.8° | 689 °/s | 1261 °/s | 402 °/s | 76.6 % |
| powerloop-r1d1-gate3mask-fullswitch | 0.660 N | 39521 | 2.54 m/s | 5.79 m/s | 32.5° | 136.8° | 368 °/s | 500 °/s | 362 °/s | 63.6 % |

(Mean-tilt entry for fixed-baseline omitted — the statistics file does not compute it, and the flight is too short and non-stationary to average meaningfully.)

Three bags, same drone, same track, same hour:

- **Mean thrust varies by 36 % (0.52 → 0.71 N).** Wider spread than on the circle. Fitting TWR DR to any single bag would contradict the other two.
- **Commanded roll / pitch rate envelope varies by ~7× across bags.** The fixed-baseline policy hard-clips at 100 °/s (its action space ceiling); the r1d1 policies emit transient peaks into the hundreds of deg/s because their action space is wider *and* because they occasionally re-stabilise after a gate brush. Neither number is "the hardware's body-rate capability" — the first is the policy's output cap, the second is a recovery event.
- **Max tilt varies between 49° and 137°.** The 137° peak in `r1d1-gate3mask-fullswitch` is a post-contact roll-over that the policy recovers from; the 49° in the fixed-baseline bag is simply that the policy crashed before ever having to invert. Setting sim orientation DR from any one percentile would give a different answer each time.
- **Bang-bang thrust fraction 64–77 %** — inside the circle's 66–82 % band. This is a sim artefact the policies inherit from the training environment: with high sim TWR and no smoothness penalty on thrust, extreme on/off is locally optimal. Same trap on both tracks.

Cross-bag disagreement is as bad on powerloop as on circle. **The policies' "operating regime" is not the drone's operating regime.**

---

## 4. What's New on Powerloop vs Circle

### 4.1 Real vertical dynamics, for the first time

The powerloop's inverted gate-2 traversal drives the euler envelope outside anything seen on the circle:

| Axis | Circle max | Powerloop max (r1d1-fullswitch) |
|---|---:|---:|
| Roll | 57° | 177° |
| Pitch | 50° | 80° |
| Yaw | full 360° (both tracks lap) | full 360° |

Roll saturating near ±180° on every successful lap is a track feature, not an outlier. This has two implications for sim:
1. `CircleQuadcopterStrategy`'s observation noise / initial-state randomization envelope may be too narrow on orientation to be meaningful for powerloop training — if we want powerloop transfer, DR should cover inverted flight.
2. The 40-dim observation passes rotation as a 3×3 matrix, which is singularity-free, so this is a ranges-only change, not a parameterisation change. No obs-layer work needed.

### 4.2 Recovery transients

Both r1d1 bags show single-sample angular-velocity peaks of 8 000 / 10 000 / 1 350 °/s on roll / roll / pitch respectively. These are almost certainly numerical artefacts of finite-differencing through a fast attitude snap (a common post-collision transient), not real 10 000 °/s hardware rates. But their existence means the real drone **is** making contact with gates and recovering, whereas the sim's contact-sensor crash threshold terminates the episode. The sim cannot be used to train the recovery, only to avoid needing it. If we want the policy to handle gate brushes on race day, we either:

1. soften the sim's crash criterion and add a large recovery reward, or
2. accept that the sim only trains the "don't touch the gate" mode and use the race bags to validate that `min(gate_distance)` stays positive on every lap.

Option 2 matches the course-stated priority ("robustness, zero tolerance for crashes").

### 4.3 Zero-shot circle→powerloop does not transfer

The `powerloop-fixed-baseline` bag is the circle-baseline controller run on the powerloop. It achieves 0 ordered laps in 22 s. This is information about the *policy*, but it's load-bearing for the project plan: **we cannot use the circle policy for the 04-28 race**. The r1d1 policy (trained specifically for powerloop) completes 4–5 ordered laps. The sysid calibration only matters conditional on finishing the training run, so the dominant short-term risk is policy design, not dynamics miscalibration. Dynamics miscalibration sets the *ceiling* on how well any trained policy can transfer; it does not produce a working policy by itself.

### 4.4 Gate-progression state machine > dynamics

`r1d1-gate3mask` (4 laps, 7.36 s best) and `r1d1-gate3mask-fullswitch` (5 laps, 7.50 s best) use the same policy weights and essentially the same dynamics (mean thrust 0.71 vs 0.66 N, max speeds 5.30 vs 5.79 m/s, identical max-thrust 1.174 N). The lap count difference comes from the "full switch" gate-progression logic, not from physics. This is a reminder that *not every sim-real gap is a dynamics gap* — some of our real-world failures are in the observation-construction / gate-switching code on the real side, which the sim cannot replicate if it uses a different switching rule.

---

## 5. What the Powerloop Race Bags Actually Tell Us

Same three things the circle bags told us, with a couple of additions:

1. **Max commanded thrust 1.174 N (controller label).** Same as circle.
2. **Identified mass 63–65 g.** Same as circle. Two tracks agreeing is stronger evidence than one track with multiple bags, because any track-specific aerodynamic bias (ground effect on the circle at 0.75 m gate height, ceiling effect on powerloop at 2.0 m gate 2) affects force balance differently yet the identified mass is unchanged.
3. **Which powerloop policies succeed and how they fail.** `fixed-baseline` breaks sequence, `r1d1-gate3mask-fullswitch` reliably laps. Useful for policy / reward design, not for sysid.
4. **A new data point: inverted-flight orientation envelope.** We now have ground-truth roll trajectories past 140° on real hardware. These are usable to check whether the sim's rotation-matrix observation pipeline actually produces valid inputs in that regime.
5. **Negative result: the calibrated quadratic PWM map is *not* in use on 04-20.** If we planned to re-fit DR against Bruce's 04-16 controller, that plan is moot for the upcoming race unless the deployment team switches to the calibrated controller before 04-28.

Everything else — mean thrust spread, tilt distributions, commanded-rate envelope — is policy behaviour, not hardware.

---

## 6. Insights for Sysid and Sim-to-Real

The circle analysis made three recommendations. Powerloop data lets us prioritise and sharpen them:

### 6.1 Priority-0: mass

Two independent tracks identify ~64 g. Weigh the drone on a kitchen scale, set `mass_kg` in the sim to the scale value (± 5 % DR for battery / prop variation), and retrain. Five-minute change. If the scale reads 30 g, something is structurally different about this drone from the Crazyflie we planned against and we must rebuild from there; but given two tracks agree the odds are the scale reads ~65 g.

### 6.2 Priority-1: PWM map clarity before 04-28

The baseline controller on 04-20 still uses the linear PWM formula. Before the race:
- Decide whether the 04-28 deployment will use the linear map or Bruce's calibrated quadratic map.
- Ensure the sim's thrust model matches the deployed map. If we train against a calibrated quadratic but race with the linear controller, we re-create the bruce-vs-zhuohao mismatch described in §3 of the circle doc.
- This is a binary decision that retires a whole category of sim-real gap without needing any sysid.

### 6.3 Priority-2: run `fit_dynamics.py` against `sysid/group30_sysid_0.mcap`

`rosbags_powerloop_baseline_controller_04_20/sysid/` already contains a dedicated sysid bag from the same session, with the same drone + battery + props as the three race bags above. That bag (not the race bags) is the correct input for identifying:
- motor time constant `τ_m`
- total command latency `τ_c`
- per-axis rate-controller bandwidth
- linear drag `k_d_{x,y,z}`

The sysid bag's analysis output isn't in this directory yet. Running `analyze_rosbag.py sysid crazy_jirl_b3` and then a chirp-aware fitter is the next concrete action. Once that JSON lands, DR ranges should be anchored to it, with the race bags (these three + the circle eight) used purely as validation.

### 6.4 Priority-3: extend DR orientation envelope for powerloop

If / when we retrain for powerloop:
- Training-time euler DR should allow at least |roll| up to ~170° for a short fraction of the episode (the inverted segment), not truncate near ±60°.
- Initial-state randomization for ground takeoffs does not need to change; the inverted segment is a trajectory feature, not a reset feature.
- This is not a DR "range widening" in the classical sense — it's making sure the sim can simulate the track. The circle-only DR schedule is physically insufficient for powerloop regardless of how well mass is identified.

### 6.5 Race bags → validation set, not generator

Same conclusion as the circle doc, now with twice the supporting evidence. Specifically the 11 bags give us:
- A hardware-parameter prior with tight error bars (mass 63.3 ± 3.6 g, max thrust 1.173 ± 0.003 N).
- A policy-regime state distribution to cover from sim.
- A failure-mode catalogue (v3 lock-in, v4 gate skipping, fixed-baseline sequence breaks, r1d1 inverted-flight recoveries) for reward / obs debugging.

Race bags do *not* give us motor τ, latency, drag, or rate bandwidths. Those require `sysid/`.

---

## 7. Files

- `plots/powerloop-fixed-baseline/crazy_jirl_b3_statistics.json` — per-bag stats.
- `plots/powerloop-r1d1-gate3mask/crazy_jirl_b3_statistics.json`
- `plots/powerloop-r1d1-gate3mask-fullswitch/crazy_jirl_b3_statistics.json`
- `plots/<bag>/crazy_jirl_b3_ctbr_full.json` — full command stream used for PWM-map fitting.
- `plots/<bag>/crazy_jirl_b3_odom_full.json` — full state stream used for mass identification + rate-lag cross-correlation.
- `summary.csv` — per-bag lap-level race summary from `batch_eval`.
- `sysid/group30_sysid_0.mcap` — dedicated sysid bag (not analysed in this document; recommended as the next step).
- `../rosbags_04_15/dynamics_analysis.md` — circle-track companion analysis; this document extends it.
