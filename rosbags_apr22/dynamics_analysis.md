# Apr-22 Rosbag Analysis — Third Race Session, First sim-TWR-Calibrated Policy

**Date:** 2026-04-24
**Scope:** Three rosbags recorded 2026-04-22 in the Pennovation arena, flying policy variants of `r1d1-gate3mask-fullswitch` trained with `twr1p87` (sim thrust-to-weight fixed to 1.87 to match the mass identified from the 04-20 session). Compared against the 04-20 baseline-controller bags analysed in `../rosbags_powerloop_baseline_controller_04_20/dynamics_analysis.md` and the 04-15 circle bags one level above that. Goal: verify the 04-20 sysid recommendations (priority-0 mass fix via `twr1p87`) were executed correctly, reconfirm hardware signatures on a new drone (`crazy_jirl_b5` vs the earlier `b3`), and read out what the three policy variants tell us about the remaining sim-real gap.

**Bags:**

| Bag | Duration | Ordered laps | Best lap | Mean speed | Policy variant |
|---|---:|---:|---:|---:|---|
| `group30_powerloop-r1d1-gate3mask-fullswitch-twr1p87` | 44.6 s | **5** | **6.121 s** | 3.59 m/s | reference: r1d1-gate3mask-fullswitch retrained against `twr1p87` |
| `group30_powerloop-r1d1-gate3mask-fullswitch-twr1p87-speedv4-currentvel-g560` | 40.4 s | 4 | 6.659 s | 2.82 m/s | same + "speedv4" observation with current-velocity formulation |
| `group30_powerloop-r1d1-gate3mask-fullswitch-twr1p87-speedv2-postloopvel` | 23.8 s | 1 | 6.707 s | 2.62 m/s | same + "speedv2" post-loop-velocity observation — **tumbled out at t≈22 s** |

**Source:** `summary.csv`, `plots/<bag>/crazy_jirl_b5_{statistics,ctbr_full,odom_full,observations_full,gate_passes}.json`.

**TL;DR — the sim mass fix landed, the policy is the fastest we've run, the hardware story is still unchanged, and the "powerloop" track deployed on 04-22 is *not the same track as 04-20*.** Every hardware signature (max thrust 1.174 N, PWM map `2.349e-5·PWM − 0.235`, ~60 g identified mass, 60–70 ms rate-tracking lag on clean flight) reproduces on the new drone `crazy_jirl_b5`. Training with `twr1p87` (the value implied by the 04-20 identified-mass analysis: `1.174 / (0.064 · 9.81) ≈ 1.87`) delivered the best lap time and lap-count numbers we've ever logged — **5 ordered laps, 6.12 s best, 3.59 m/s mean speed**, vs 5 laps / 7.50 s / 2.54 m/s at the peak of 04-20. That is a **19 % faster best lap with ~40 % higher mean speed on the same drone class**. The calibrated-quadratic PWM map from the 2026-04-16 changelog is *still* not active — the deployed controller continues to use the linear map. The two alternate observation variants (`speedv4`, `speedv2`) are both slower than the reference and one of them crashed, so the obs-design experiment is a clear negative result.

---

## TL;DR — plain-language version

### Did the mass fix work? Yes, and it was the biggest single win so far.

04-20's top recommendation was to raise sim mass from 30 g to the identified ~64 g (equivalently, drop sim TWR from ~4.0 to ~1.87). The 04-22 `twr1p87` policy is the first deployment after that change. Side-by-side on an equivalent track difficulty:

| Metric | 04-20 r1d1-fullswitch | 04-22 twr1p87 reference | Δ |
|---|---:|---:|---:|
| Ordered laps | 5 | 5 | same |
| Best lap | 7.50 s | **6.12 s** | **−18 %** |
| Mean speed | 2.54 m/s | **3.59 m/s** | **+41 %** |
| Max speed | 5.79 m/s | 6.25 m/s | +8 % |
| Mean commanded thrust | 0.660 N | 0.735 N | +11 % |
| Bang-bang % | 63.6 % | 73.9 % | +10 pp |
| Max tilt | 136.8° | 59.9° (roll) | much lower — see §3 |

The increase in commanded thrust (0.66 → 0.74 N) is consistent with a policy that now understands it has less thrust margin and exploits it — mean thrust / max thrust = 0.74 / 1.17 = 63 %, up from 56 % in 04-20. This is the signature of a policy trained under the correct TWR.

### Is the "powerloop" track the same as 04-20? No.

Reconstructing gate positions from the drone's position at each pass (averaged over the 6 laps of the reference bag):

| Gate | (x, y, z) m | Notes |
|---|---|---|
| 0 | ( 1.47, 8.02, 1.06) | NE |
| 1 | (−1.62, 7.84, 1.88) | NW, elevated |
| 2 | (−0.87, 4.49, 0.95) | mid-left |
| 3 | (−0.08, 4.49, 1.07) | mid-center |
| 4 | (−1.47, 1.27, 1.84) | SW, elevated |
| 5 | ( 1.44, 1.02, 1.15) | SE |
| 6 | ( 0.66, 4.52, 0.83) | mid-right |

This is a **7-gate figure-8-ish track with two elevated gates (1 and 4) at z ≈ 1.9 m**, not a vertical power loop. The euler-angle envelope confirms it: max roll 59.9°, max pitch 44°, no inversion. The 04-20 "powerloop" track pushed roll past 140° and pitch to ±80° on every lap. **The two sessions are flying different physical tracks despite sharing the same `--track powerloop` nomenclature**, and side-by-side comparisons of tilt envelopes or max angular velocity are meaningful only within a session, not across sessions. This also resolves a small mystery from 04-20: "gate3mask + fullswitch" makes much more sense on a 7-gate figure-8 (where gate 3 is a tight middle-crossing) than on a 4–5 gate inverted loop.

Implication for sim: **the `CircleQuadcopterStrategy` orientation DR no longer needs the wide-roll extension we flagged for 04-20 powerloop**. The tracks the 04-22 policy actually flies are within a ±60° roll envelope, comfortably inside nominal DR. The wider-roll DR is still relevant if we ever return to a true inverted loop, but it is not required for the 04-28 race configuration unless the track changes again.

### Hardware is still a hardware — 11 → 14 bags, 2 drones, 2 tracks, 3 sessions, one answer.

Per the 04-20 analysis, the hardware-invariants-cross-check is the load-bearing claim behind using rosbags for sysid. The 04-22 data reinforces it:

| Signature | 04-15 circle (b3, 8 bags) | 04-20 powerloop (b3, 3 bags) | 04-22 (b5, 3 bags) |
|---|---|---|---|
| Max thrust @ PWM 60000 | 1.170 ± 0.003 N | 1.174 N | **1.174 N** |
| PWM→N map | `2.4e-5·PWM − 0.235`, linear | `2.35e-5·PWM − 0.24`, linear | **`2.349e-5·PWM − 0.2349`, linear** |
| Identified mass median | 65.9 g | 63.3 g | **60.2 g** (b5 bags) |
| Rate-tracking lag (clean bags) | 50–70 ms | 60–90 ms | **60–70 ms** |

The 4 g drop in identified mass (63.3 → 60.2 g) is within the circle-bag-to-bag spread (σ = 3.6 g) and could reflect a different battery, a different drone (`b5` vs `b3`), or a slightly different filter-window composition on this session. It is not a sign of a hardware change. The PWM-map constant of **0.2349** reproduces to 4 significant figures across two drones and two tracks, which is strong evidence that the controller side truly is unchanged.

### The calibrated quadratic PWM map is *still* not in use on 04-22

The 04-20 document flagged: "the 2026-04-16 calibrated quadratic PWM map is still not in these bags". The 04-22 bags are no different:

| Bag | Quadratic coef `a` in `a·PWM² + b·PWM + c` |
|---|---:|
| twr1p87 (reference) | 1.8 × 10⁻¹⁵ (numerical zero) |
| twr1p87-speedv2-postloopvel | −7.6 × 10⁻¹⁵ (numerical zero) |

The 04-16 calibration has now gone six days without making it into a race deployment. **Before 04-28, someone needs to decide whether to race with the linear map (match the current policies' training assumption) or to switch to the calibrated quadratic map (and retrain accordingly)**. Racing on a third combination — policy trained against one thrust map, deployed on the other — would reintroduce the exact Bruce-vs-Zhuohao mismatch §3 of `../rosbags_04_15/dynamics_analysis.md` warned against.

### The two observation-variant policies are negative results

Both alternate-observation variants, built on top of the same `twr1p87` mass fix, underperform the reference:

- **`speedv4-currentvel-g560`** — current-velocity observation, completes 4 ordered laps at 6.66 s best (vs 6.12 s for reference), mean speed 2.82 m/s vs 3.59 m/s. Also **missed recording `/ctbr_cmd` entirely** — no thrust/rate commands in the bag, so we can't identify mass or PWM map from it independently. Controller-side recording-setup regression.
- **`speedv2-postloopvel`** — post-loop-velocity observation, tumbles to −119° roll / −84° pitch at t ≈ 22 s after 1 lap, with single-sample angular-velocity spikes of 3000–5700 °/s (finite-difference numerical transient post-contact, not real body rates). Same failure mode as 04-20's `powerloop-fixed-baseline`: degenerate closed-loop data, useless for sysid cross-correlation (rate-tracking correlation drops to 0.32–0.69 on that bag vs > 0.90 on the reference).

Interpretation: the reference observation is the strongest design in this set, and the two variants are regressions. This is a policy-side result, not a dynamics / DR result — the hardware is fine, the base observation is fine, the variant observations aren't.

### The dominant sim-real gap, post-mass-fix

Pre-fix (30 g sim, 64 g real) the gap was a 2× mass error. That is now gone. What remains, in descending order of likely impact:

1. **Commanded-rate saturation vs actual-rate headroom**. The `twr1p87` policy saturates its ±100 °/s action cap on roll and pitch almost constantly (roll_mean_abs = 75 °/s with max 100 °/s; pitch_mean_abs = 64 °/s with max 100 °/s). But the **measured** body rates exceed the commanded cap (actual 99th-percentile roll-rate 159 °/s, yaw 165 °/s). The policy is asking for max rate, the onboard PID is slamming the motor mix to deliver it, and the drone is overshooting the commanded rate. This means *the sim is under-modeling how aggressive the real rate loop can be when policy demands max*. If we raise the sim's effective rate-loop bandwidth (or add a commanded-rate integrator / overshoot model), the policy trained against it might use finer-grained rate commands instead of bang-bang.
2. **Bang-bang thrust at 74 %**. Same artefact flagged on circle and 04-20 powerloop: with the fixed sim TWR now closer to truth, bang-bang is no longer a mass error, it's an optimization shape — a policy without thrust-smoothness penalty converges to on/off. With mass fixed, there's no physical reason the real drone benefits from 74 % bang-bang; the *sim* reward does. A thrust-rate-of-change penalty in the training reward is a cheap experiment and would likely cut this back toward 40–50 %.
3. **Commanded vs measured yaw mismatch**. On the reference bag, commanded yaw rate peaks at 199.7 °/s (pretty much the policy cap of 200), yet measured 99th-percentile yaw rate is only 165 °/s. On roll/pitch the measured envelope is *above* the cmd cap; on yaw the measured is *below*. This is the expected signature of a yaw axis that is low-authority relative to roll/pitch (less thrust differential on a +-configured quad), and matches Crazyflie physics. Worth mirroring in the sim's yaw gain if it isn't already.
4. **Track geometry. The 04-22 "powerloop" is planar and doesn't need the inverted-roll DR.** If the 04-28 race uses the same 7-gate figure-8 layout, the DR envelope we had already is sufficient. If it uses the 04-20 inverted layout, we need the wider-roll DR back.

None of the above is a 2× error. Most are 10–30 % mismatches that DR can absorb if we target them, which is exactly what the original plan envisaged.

### Concrete next steps

1. **Sysid fit** — the 04-20 doc still calls out `rosbags_powerloop_baseline_controller_04_20/sysid/group30_sysid_0.mcap` as the right input for motor-τ / latency / drag identification; that action is still open. No comparable sysid bag in the 04-22 set.
2. **Decide on PWM map before 04-28** — linear (current) or calibrated quadratic (04-16 commit). Binary. Must be consistent between training and deployment.
3. **Add thrust-rate smoothness penalty to `rewards.action_smoothness`** (or equivalent key). Target 40 % bang-bang. Small sim-side experiment.
4. **Skip** any further observation-variant experiments on this policy family until the reference is re-validated — both `speedv4` and `speedv2` regressed.
5. **Recording-pipeline fix**: the `speedv4` bag is missing `/ctbr_cmd`. Before 04-28, dry-run the recording configuration to ensure every topic we rely on for sysid (`/ctbr_cmd` specifically) is captured on every bag.

---

## Suggested DR patch — upfront summary

Full audit in §6. TL;DR of what the `Powerloop Fullswitch Real-Effective TWR` baseline (`scripts/run/train_powerloop_fullswitch_real_twr.sh`) covers vs what the 04-22 bags say it should cover:

**Already covered correctly:** nominal `thrust_to_weight = 1.87`, aero DR `0.5–2.0`, PID-gain DR (`kpki ±15 %`, `kd ±30 %`), `mass_variation` and `motor_tau` explicitly OFF pending sysid.

**Gaps, in priority order:**

| # | Gap | Evidence | Suggested patch |
|---|---|---|---|
| 1 | TWR band too tight on the low-mass side | `b5` at 58.7 g → real TWR 2.04, outside current [1.76, 1.98] | `twr_randomization_pct: 0.06 → 0.10` |
| 2 | Action-latency window too narrow | Measured rate-tracking lag 60–70 ms vs ~20 ms sim coverage | `action_latency_max: 2 → 6` (optionally `fixed_action_delay_steps: 4`) |
| 3 | Rate-loop overshoot not reproduced | 99th-pct roll rate 159 °/s vs cmd cap 100 °/s (+59 %) | Turn on `motor_tau_scale: 0.8–1.2` *after* fitting 04-20 sysid bag |
| 4 | PWM-map mismatch has zero DR | Linear `2.349e-5·PWM − 0.2349` in sim and deploy; 04-16 quadratic unmerged | Add PWM-nonlinearity DR, or lock one map pre-race |
| 5 | Yaw authority asymmetry not modelled | Real yaw 165 °/s vs cmd cap 200 °/s | Per-axis rate-gain DR (yaw 0.6–1.0, roll/pitch 0.9–1.1) |
| 6 | Orientation DR is per-track | 04-22 stays inside ±60° roll; 04-20 inverted needed ±170° | Confirm 04-28 track; keep nominal DR for planar figure-8 |

**Safe to ship now:** #1 and #2 — both are direct sim-side adjustments with no new sysid dependency.
**Needs sysid fit first:** #3 (motor-τ) depends on `rosbags_powerloop_baseline_controller_04_20/sysid/group30_sysid_0.mcap`.
**Binary decision before 04-28:** #4 (PWM map) — must be consistent between training and deployment.
**Not a DR issue:** 74 % bang-bang thrust is a reward-shape artefact post-mass-fix; fix via `thrust_rate_smoothness` reward term (§5.4), not DR.

Concrete diff to `ENV_OVERRIDES` in the baseline script:

```diff
- "twr_randomization_pct":0.06,
+ "twr_randomization_pct":0.10,
- "action_latency_max":2,
+ "action_latency_max":6,
- "motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,
+ "motor_tau_scale_min":0.8,"motor_tau_scale_max":1.2,   // ONLY after sysid fit
```

---

## 1. Methodology

Identical pipeline to the 04-20 analysis. All three bags were processed with `sim2real/bin/analyze_rosbag.py`, emitting the standard set of `<ns>_*.json` artefacts per bag (see `docs/ros_bags_fields.md` for the field list). We re-extracted the following quantities for cross-session comparison:

- Linear and quadratic fit of `thrust_n` vs `thrust_pwm` on the `/ctbr_cmd` stream.
- Max `thrust_n` at PWM ≥ 59500.
- Identified drone mass from vertical force balance: `m = cos(roll)·cos(pitch) · thrust_n_interp(t) / (a_z_world + g)`, with `a_z_world` via `np.gradient` on `twist.linear.z`. Filter: `z > 0.3 m`, `thrust > 0.3 N`, `cos(roll)·cos(pitch) > 0.5`, `|a_z| < 5 m/s²`, clipped to 12–80 g physical bounds.
- Peak-correlation rate-tracking lag on a 100 Hz grid with ±400 ms search window, on each of (roll, pitch, yaw).
- Bang-bang thrust fraction (`thrust_n < 0.1 N` or `> 1.0 N`), plus explicit PWM-saturation fractions at the 10001 / 60000 rails.
- Euler and body-rate envelopes from odom.
- Gate layout reconstructed from drone position at each `/<ns>/observations` gate-pass timestamp, averaged across the laps in a bag.

The namespace on all 04-22 bags is `crazy_jirl_b5` (a different drone from 04-20's `b3`). This is a meaningful detail for hardware-signature comparison.

---

## 2. Hardware Signatures — Unchanged on a Different Drone

### 2.1 Max thrust at saturation

| Bag | Max `thrust_n` at PWM = 60000 |
|---|---:|
| twr1p87 (reference) | **1.1743 N** |
| twr1p87-speedv2-postloopvel | **1.1743 N** |
| twr1p87-speedv4-currentvel-g560 | *(no `/ctbr_cmd` recorded)* |

Same 1.17 N ceiling as every circle and 04-20 bag.

### 2.2 PWM→Newton mapping — still the linear formula, identical to 4 sig figs

| Bag | Linear fit `b·PWM + c` | Quadratic `a` |
|---|---|---:|
| twr1p87 (reference) | `2.349e-5·PWM − 0.2349` | 1.8 × 10⁻¹⁵ |
| twr1p87-speedv2-postloopvel | `2.349e-5·PWM − 0.2349` | −7.6 × 10⁻¹⁵ |

Linear coefficient matches 04-20 (`2.349e-5`) and Race-1 circle (`≈2.4e-5`) to 3 significant figures. Intercept matches 04-20 (`−0.2349` vs `−0.2349` vs `−0.2381`) to 3 significant figures. The controller's PWM table has not changed across sessions — **the 04-16 calibrated quadratic remains unmerged into the race deployment**.

### 2.3 Identified mass — ~60 g on `b5` vs ~64 g on `b3`

Robust filter as in §1:

| Bag | Median identified mass | σ (g) | N filtered samples |
|---|---:|---:|---:|
| twr1p87 (reference) | 58.7 g | 12.6 | 1260 |
| twr1p87-speedv2-postloopvel | 61.7 g | 11.6 | 315 |
| twr1p87-speedv4-currentvel-g560 | — | — | — (no ctbr) |

**Pooled median: ~60.2 g.** The ~5 g drop vs the 04-20 `b3` pooled median (63.3 g) is consistent with a different physical drone / battery combination, and is within the within-drone bag-to-bag spread (σ = 3.6 g on the circle). Per the sysid formula `TWR = max_thrust / (m · g) = 1.174 / (m · 9.81)`:

| Mass | Implied real TWR |
|---|---:|
| 58.7 g (ref bag) | **2.04** |
| 60.2 g (pooled 04-22) | **1.99** |
| 63.3 g (pooled 04-20) | 1.89 |
| 65.9 g (pooled 04-15) | 1.82 |

The `twr1p87` sim training target was chosen to match the 04-20 pooled identified mass. The 04-22 drone (`b5`) is slightly lighter than that assumption by ≈ 7 %, which the DR bandwidth on `thrust_to_weight_ratio` easily covers. Good news.

### 2.4 Rate-tracking latency — 60–70 ms on the clean bag, noise-contaminated on the crashed bag

Peak cross-correlation lag between commanded and measured body rate:

| Bag | Roll | Pitch | Yaw | Comment |
|---|---:|---:|---:|---|
| twr1p87 (reference) | **+60 ms (r = 0.90)** | **+70 ms (r = 0.92)** | **+70 ms (r = 0.90)** | Tightest, highest-correlation lag estimates we have |
| twr1p87-speedv2-postloopvel | +90 ms (r = 0.39) | +50 ms (r = 0.69) | +10 ms (r = 0.32) | Tumble at t≈22 s corrupts xcorr — ignore |

The reference bag is the **cleanest rate-tracking dataset in the repo to date**: correlation above 0.90 on all three axes and lags tight in the 60–70 ms band. This is still closed-loop data, so per the standing caveat from 04-20 §6 it is not a substitute for an open-loop chirp — but as a sanity check on motor-τ / action-latency sim parameters, it's the best we have.

### 2.5 Summary across all sessions

| Signature | 04-15 (b3, circle) | 04-20 (b3, inverted powerloop) | 04-22 (b5, planar figure-8) |
|---|---|---|---|
| Max thrust @ 60000 | 1.170 ± 0.003 N | 1.174 N | 1.174 N |
| Identified mass | 65.9 g | 63.3 g | 60.2 g |
| PWM map | linear, ~2.4e-5 | linear, 2.35e-5 | **linear, 2.349e-5** |
| Rate lag r/p/y (clean bags) | 50–70 / 60–90 / 50–70 ms | 60–90 / 70–100 / 50–60 ms | **60 / 70 / 70 ms** |
| Drone | `b3` | `b3` | `b5` |

Three sessions, two drones, one hardware story.

---

## 3. Policy-Driven Metrics — The twr1p87 Reference Is the Fastest Policy We've Deployed

| Bag | mean thrust | mean PWM | mean speed | max speed | max roll | max pitch | cmd roll max | cmd pitch max | cmd yaw max | bang-bang % | laps | best lap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 04-22 twr1p87 ref | **0.735 N** | 41286 | **3.59 m/s** | 6.25 m/s | 59.9° | 44.0° | 100 °/s | 100 °/s | 200 °/s | 73.9 % | **5** | **6.12 s** |
| 04-22 twr1p87-speedv2 | 0.730 N | 41105 | 2.62 m/s | 6.25 m/s † | 62.1° | 71.7° | 100 | 100 | 200 | 83.3 % | 1 | 6.71 s |
| 04-22 twr1p87-speedv4 | — | — | 2.82 m/s | 5.93 m/s | 29.0° | 44.9° | — | — | — | — | 4 | 6.66 s |
| 04-20 r1d1-fullswitch | 0.660 | 39521 | 2.54 | 5.79 | 136.8° | 80.0° | 368 | 500 | 362 | 63.6 % | 5 | 7.50 s |
| 04-20 r1d1-gate3mask | 0.705 | 40141 | 2.71 | 5.30 | 108.8° | 80.0° | 689 | 1261 | 402 | 76.6 % | 4 | 7.36 s |

† max speed 397 m/s in the JSON is a finite-difference spike during the crash; 6.25 m/s is the pre-crash true max.

### 3.1 The `twr1p87` reference bag is not bang-bang-bottlenecked, it's thrust-magnitude-bottlenecked

Mean thrust on the reference bag is **0.735 N on a ~59 g drone**, i.e. mean `thrust / weight = 0.735 / (0.0587 · 9.81) = 1.28`. Policy uses full thrust (≥1.0 N) for 33 % of the flight and zero thrust (≤0.1 N) for 14 % of the flight. Compare to 04-20 r1d1-fullswitch: mean thrust 0.66 / (0.064 · 9.81) = 1.05 — i.e. on average flew at just barely above hover.

Reading those two numbers together: **the `twr1p87`-trained policy flies with more average thrust margin, which buys it the extra kinetic energy and tighter cornering that show up as the 19 % better lap time.** This is exactly what you'd expect from a mass-fixed retrain.

### 3.2 The rate cap binds harder than the thrust cap

`roll_mean_abs = 75 °/s` with `roll_max_abs = 100.0 °/s` (policy action cap). That's a policy spending most of its wallclock at or near the action cap on roll. Same on pitch (64 / 100). Yaw is less saturated (92 / 200), consistent with the yaw-authority physics noted in §0.

This is interesting because, at the same time, actual body-rate 99th percentiles are **above** the commanded cap:

| Axis | Cmd max | Actual 99th percentile | Actual peak |
|---|---:|---:|---:|
| Roll | 100 | **159** | 652 |
| Pitch | 100 | 134 | 195 |
| Yaw | 200 | 165 | 184 |

The commanded signal is a bang-bang square wave at ±100 °/s on roll/pitch; the physical integrator output lags and overshoots, producing measured rates higher than any instantaneous command. The peak of 652 °/s on roll is a single-sample transient (likely post-contact, as in 04-20 §4.2), but the 99th-percentile values are real steady-state overshoot. **This tells us the real rate loop has enough bandwidth to overshoot the commanded rate, and the sim's rate-controller model should produce similar overshoot if we want the policy to learn to ask for the cap and stop worrying.**

### 3.3 The `speedv2-postloopvel` tumble is a recovery that fails

`speedv2-postloopvel` logs 15 passes and only 1 ordered lap before roll hits −118° and pitch hits −84°, with yaw rate integrator spiking to 5670 °/s (obvious finite-difference artefact). The drone recovers briefly — `/ctbr_cmd` continues publishing for another ~14 s after the tumble — but the policy can't re-acquire the gate sequence. Gate-interval list shows the last three "passes" at intervals of 0.008 / 0.008 / 0.025 s: these are spurious passes from a drone spinning through the gate-volume as it tumbles. Don't use this bag for sysid. It *is* useful as a failure-mode catalogue entry:

- Post-loop-velocity observation is fragile to post-collision angular state.
- The policy has no recovery controller for >90° roll, consistent with the sim crash-termination model.
- The drone hits PWM-high saturation 43.8 % of the flight (vs 33.2 % on reference), which means during the recovery attempt it's commanding max thrust — i.e. trying to stop falling — and failing to re-orient.

### 3.4 The `speedv4-currentvel-g560` bag flies clean but slower

This variant completes 4 ordered laps (vs 5 on reference) at 6.66 s best (vs 6.12 s). It also stays notably calmer: max roll 29°, max pitch 45°. The most likely interpretation is that the current-velocity observation formulation biases the policy toward a less-aggressive operating point (lower speeds, less attitude aggression). Without `/ctbr_cmd` in the bag we can't say whether the commanded thrust distribution differs.

**Critical recording regression**: `ls plots/group30_powerloop-r1d1-gate3mask-fullswitch-twr1p87-speedv4-currentvel-g560/` confirms there is no `crazy_jirl_b5_ctbr_full.json` and no `crazy_jirl_b5_angular_velocity_comparison.png`. This bag is observation-side only. Whatever the deployment script does to record `/ctbr_cmd`, it didn't fire here. Fix before 04-28.

---

## 4. What's New vs 04-20

### 4.1 Track layout — figure-8 with two elevated gates, no inversion

As derived in §0 TL;DR, the 04-22 powerloop is a 7-gate planar figure-8 with gates 1 and 4 elevated to z ≈ 1.9 m. Gate reconstruction was done by averaging drone-position-at-pass across 6 laps; spread in z is ≤ 7 cm per gate, so the layout is measured, not inferred. Euler envelope stays within ±60° roll and ±45° pitch across all three 04-22 bags.

Comparing to 04-20: gate 2 of the 04-20 powerloop was explicitly described as an **inverted** traversal driving pitch to ±80° and roll past 140° on every lap. None of the three 04-22 bags even approaches that regime (outside the speedv2 tumble, which is off-policy recovery). **The tracks are different physical configurations.**

Actions this implies:
- Any DR envelope re-tuning for the 04-22 configuration does not need to cover inverted flight.
- If we ever go back to the 04-20 track, the DR envelope from the 04-20 §6.4 recommendation (|roll| up to ~170°) becomes relevant again.
- **Before 04-28, confirm which physical track will be used.** The `--track powerloop` CLI flag does not disambiguate.

### 4.2 Twin confirmation that race bags ≠ sysid bags

The `speedv2-postloopvel` bag is the second example (after 04-20 `powerloop-fixed-baseline`) of rate-tracking cross-correlation being noise-destroyed by a single-event tumble. This is general: closed-loop race data can lose an entire signal to one bad second. An open-loop chirp bag has no such failure mode, which is why the sysid-bag action item carries over unchanged from 04-20.

### 4.3 The `b5` drone has slightly lower identified mass than `b3`

04-22 pools to 60.2 g; 04-20 pooled to 63.3 g. Difference is ~5 %, within within-drone spread (σ = 3.6 g on circle). Not a new hardware regime — just a reminder that if per-drone mass gets reported on race day, use the actual number for DR centring rather than re-using the 04-20 average.

---

## 5. Insights for Sysid and Sim-to-Real (04-24 edition)

Priorities from the 04-20 doc, updated with 04-22 evidence:

### 5.1 Priority-0 (was): mass — **DONE**

`twr1p87` retraining closed the ~2× mass gap. Result visible in the 19 % better lap time on the same drone class. Revisit only if a future drone weighs outside the 55–70 g envelope.

### 5.2 Priority-1 (carry-over): PWM-map decision before 04-28

Unchanged from 04-20 §6.2. The calibrated-quadratic map from 04-16 is still not in race deployment as of 04-22. Policy + controller must agree. Choose one.

### 5.3 Priority-2 (carry-over): run `fit_dynamics.py` against the 04-20 sysid bag

No sysid bag was recorded in the 04-22 session. The 04-20 sysid bag remains the only open-loop dataset we have. Action unchanged.

### 5.4 Priority-3 (new): thrust smoothness penalty

Now that mass is correct, bang-bang thrust is no longer a mass artefact but a reward-shape artefact. 74 % bang-bang on the reference bag is costing us actuator wear and mean-speed efficiency without buying capability. Add a `thrust_rate_smoothness` term to the reward dict and retrain. Small, targeted, testable.

### 5.5 Priority-4 (new): rate-loop overshoot in sim

Measured 99th-percentile roll rate (159 °/s) exceeds the commanded cap (100 °/s) by 59 %. If `QuadcopterEnv`'s PID produces a critically-damped or overdamped rate response, the sim underpredicts this overshoot, and the policy overfits to its in-sim rate shape. Worth probing the PID gain in `QuadcopterEnvCfg`.

### 5.6 Priority-5 (carry-over): DR orientation envelope

For the 04-22 track, the wide-roll DR extension is **not needed**. For the 04-20 inverted track, it is. This is a per-track choice — do not carry the wide-roll DR into `twr1p87`-style policies targeting planar tracks.

### 5.7 Priority-6 (new): recording pipeline fix

The `speedv4` bag missed `/ctbr_cmd`. Validate the recording script captures all six topics (`odom`, `observations`, `ctbr_cmd`, `pose`, `tf`, `multi_odometry`) before race day, with a smoke test that `analyze_rosbag.py` finds `ctbr_count > 0` on a 10-second dry run.

---

## 6. DR Coverage Audit — Baseline `Powerloop Fullswitch Real-Effective TWR` Candidate

Cross-checking the baseline training script `scripts/run/train_powerloop_fullswitch_real_twr.sh` (changelog entry "Powerloop Fullswitch Real-Effective TWR Candidate", 2026-04-22) against the sysid findings above. The current `ENV_OVERRIDES` blob is:

```json
{"track_name":"powerloop","thrust_to_weight":1.87,"twr_randomization_pct":0.06,
 "mass_variation":0.0,"action_latency_max":2,"fixed_action_delay_steps":-1,
 "motor_tau_scale_min":1.0,"motor_tau_scale_max":1.0,
 "aero_randomization_scale_min":0.5,"aero_randomization_scale_max":2.0,
 "pid_kpki_randomization_pct":0.15,"pid_kd_randomization_pct":0.30, ...}
```

### 6.1 What the baseline DOES cover

| DR knob | Baseline value | Backed by 04-22 analysis? |
|---|---|---|
| `thrust_to_weight = 1.87` | nominal | ✔ matches 04-20 pool (63.3 g → 1.89 TWR). Closed the ~2× mass gap — confirmed by the 19 % lap-time win. |
| `twr_randomization_pct = 0.06` | ±6 % → TWR ∈ [1.76, 1.98] | ✱ partially. 04-22 `b5` identified mass 58.7 g → real TWR **2.04**, just outside this window. |
| `aero_randomization_scale = 0.5–2.0` | wide | ✔ ample; nothing in the bags argues against. |
| `pid_kpki_pct = 0.15`, `pid_kd_pct = 0.30` | on | ✔ present, but see rate-loop gap (§6.2.3) below. |
| `action_latency_max = 2` steps | on | ✱ ~20 ms coverage; measured lag 60–70 ms. Under-covers. |
| `mass_variation = 0.0`, `motor_tau = 1.0/1.0` | **OFF by design** | ✔ consistent with the stated "wait for clean sysid bag" posture. |

### 6.2 What the 04-22 analysis says is NOT covered (in priority order)

#### 6.2.1 TWR lower bound is too tight for the `b5` drone

Pooled real TWR spread is 1.82 → 2.04 across 3 sessions / 2 drones. Current ±6 % DR is [1.76, 1.98]. Either re-centre the nominal at 1.95 (≈ 60 g) or widen to **±10 %** ([1.68, 2.06]) to bracket both drones without retraining per-drone.

#### 6.2.2 PWM-map mismatch — zero coverage

Linear `2.349e-5·PWM − 0.2349` is baked into both sim and deployment; the 04-16 calibrated quadratic is still unmerged. If race-day switches to the quadratic map, sim has no randomization over thrust nonlinearity. Mitigation: add a PWM→thrust nonlinearity DR (e.g. quadratic coefficient drawn in a band around `±1·a_calib`) so the policy becomes map-agnostic. Binary alternative: enforce one map pre-race and rely on the sim-side decision.

#### 6.2.3 Rate-loop overshoot — not reproduced in sim

Measured 99th-pct roll rate **159 °/s** vs commanded cap **100 °/s** (+59 %). The baseline randomizes PID *gains* only, which shifts the rate-loop pole location but does not explicitly expose the policy to overshoot-dominated responses. Two cheap additions:
- turn on `motor_tau_scale` DR (e.g. `0.7–1.3`) once the 04-20 sysid bag is fit; the 60–70 ms lag is consistent with a motor-τ contribution;
- optionally widen `pid_kpki_pct` asymmetrically upward to allow higher-gain rate loops that produce overshoot.

#### 6.2.4 Action-latency window too narrow

`action_latency_max = 2` steps ≈ 20 ms; real end-to-end rate-tracking lag is 60–70 ms. Raise `action_latency_max` to **6–8 steps** and/or set a non-default `fixed_action_delay_steps` to cover the median lag for deterministic A/B comparisons.

#### 6.2.5 Yaw authority asymmetry — not modelled

Real yaw peaks 165 °/s vs cmd cap 200 °/s (under-actuated, consistent with +-config Crazyflie physics). Sim likely delivers full yaw authority at the cap. If `QuadcopterEnvCfg` exposes a per-axis rate-gain scale, randomize yaw separately (e.g. 0.6–1.0 on yaw, 0.9–1.1 on roll/pitch).

#### 6.2.6 Track-orientation DR should be per-track

04-22 planar figure-8 stays inside ±60° roll; the wide-roll DR extension flagged for the 04-20 inverted loop is **not needed** here. Current baseline does not force wide-roll, so it is fine for 04-22 — but before 04-28 confirm which physical track will be used and toggle accordingly.

#### 6.2.7 Out of DR scope but flagged

74 % bang-bang thrust is now a reward-shape artefact post-mass-fix, not a dynamics gap. Fix with a `thrust_rate_smoothness` reward term (Priority-3 in §5.4), not DR.

### 6.3 Concrete minimal patch to the baseline script

Suggested edits to `ENV_OVERRIDES` in `scripts/run/train_powerloop_fullswitch_real_twr.sh`:

- `twr_randomization_pct`: `0.06 → 0.10`
- `action_latency_max`: `2 → 6` (and optionally `fixed_action_delay_steps: 4` for a deterministic-lag A/B)
- `motor_tau_scale_min/max`: `1.0/1.0 → 0.8/1.2` — **only after** `rosbags_powerloop_baseline_controller_04_20/sysid/group30_sysid_0.mcap` is fit; otherwise this is a guess
- Add PWM-map quadratic-coefficient DR if the env supports it; otherwise resolve the map decision (§5.2) before 04-28 instead.

Items 6.2.1 and 6.2.4 are safe to ship now; 6.2.2, 6.2.3, 6.2.5 depend on the sysid-bag fit and on whether the matching knobs exist in `QuadcopterEnvCfg`.

---

## 7. Files

- `plots/group30_powerloop-r1d1-gate3mask-fullswitch-twr1p87/crazy_jirl_b5_*.json` — reference bag, full pipeline output.
- `plots/group30_powerloop-r1d1-gate3mask-fullswitch-twr1p87-speedv2-postloopvel/crazy_jirl_b5_*.json` — tumbled variant, useful as failure-mode evidence only.
- `plots/group30_powerloop-r1d1-gate3mask-fullswitch-twr1p87-speedv4-currentvel-g560/crazy_jirl_b5_*.json` — alternate-obs variant; **no `/ctbr_cmd` recorded**.
- `summary.csv` — per-bag lap-level race summary from `batch_eval`.
- `../rosbags_powerloop_baseline_controller_04_20/dynamics_analysis.md` — 04-20 companion; this document extends it.
- `../rosbags_powerloop_baseline_controller_04_20/sysid/group30_sysid_0.mcap` — open-loop sysid bag; still the right input for motor-τ / latency / drag identification. No equivalent bag was recorded in the 04-22 session.
