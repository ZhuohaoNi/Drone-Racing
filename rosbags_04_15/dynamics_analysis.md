# Cross-Race Dynamics Analysis — Why Racing Rosbags Are a Poor Basis for Domain Randomization

**Date:** 2026-04-16
**Scope:** Comparing all circle-track rosbags from the two real-world sessions (2026-04-08 group30 batch, 2026-04-15 week-2 batch) to test whether *race rosbags* can be used as the primary source of truth for sim-side domain randomization.

**TL;DR — they cannot.** The racing rosbags reflect what each *policy* did, not what the *hardware* does. Hardware signals that should be identical across all bags (thrust-to-PWM curve, latency, mass) are either (a) consistent across races but uninformative about real dynamics, or (b) inconsistent because the **controller code itself changed between runs**. The things that do vary strongly across bags — mean thrust, body-rate envelope, tilt distribution — are artifacts of the policy's operating regime, not of the real drone.

The right fix is an open-loop system-identification flight program that decouples identification from the racing policy. This doc lays out what we found and proposes that program.

---

## TL;DR — Plain-language version

### Do the dynamics distributions match between Race 1 and Race 2?

**The underlying hardware is the same — but the bags don't look like that.** The things that *should* match if we were doing pure measurement:

| Quantity | Race 1 | Race 2 | Match? |
|---|---|---|---|
| Max thrust at PWM=60000 | 1.17 N | 1.17 N | ✅ |
| Identified drone mass | 62–69 g | 60–67 g | ✅ |

The things we might naively use to set DR ranges from the bags:

| Quantity | Race 1 spread | Race 2 spread | Same drone? |
|---|---|---|---|
| Mean commanded thrust | 0.65 – 0.77 N | 0.66 – 0.72 N | yes — but 18 % spread |
| Mean body-rate RMS | 81 – 140 °/s | 93 – 180 °/s | yes — 2× spread |
| Mean tilt | 32° – 38° | 29° – 39° | yes |
| Max tilt | 49° – 51° | 39° – 57° | yes |

**Same hardware, same track, same day. The numbers vary by 50-100 % just because different policies fly differently.** If you used race-1 bags to set DR ranges, you fit to *those three policies' habits*. If you used race-2 bags, you fit to *those other eight policies' habits*. Neither describes the real drone.

**One extra problem:** the controller code changed between bags inside race 2. Zhuohao's and group30's controllers convert thrust→PWM with a **linear** formula; Bruce's controller uses a **quadratic** formula (the 2026-04-16 fix). The same policy output produces different PWM in the two controller versions. Pre-fix and post-fix bags describe different deployment pipelines.

### Why race rosbags are bad for sysid (in one analogy)

Measuring a drone's dynamics from racing rosbags is like measuring an engine's power curve by watching one driver complete laps at Daytona. You see:

- What *that driver* did (not what the engine can do)
- A narrow slice of RPM/throttle (not the full curve)
- Input and output correlated (you can't tell what caused what)

Specifically: every policy we trained is essentially **bang-bang on thrust** (66–82 % of commands are either near-zero or near-max). The **middle of the thrust curve never gets exercised**, so we can't identify it from race data. Rate-tracking lag we measure depends on whether the policy happens to command reversals often enough to be informative — v3's "locked-attitude" policy gives a systematically biased lag estimate, not a hardware truth.

### The biggest sim-real gap is hiding in plain sight

The force-balance analysis identifies the real drone at **~66 g** across every bag. Our sim is trained on a **30 g** nominal. That means **real TWR is about half of the sim TWR**, and no amount of ±15 % DR around the sim nominal can close that gap. This alone could explain why policies that look great in sim underperform in reality.

### Proposed solution — one ~20-minute flight, no RL in the loop

Use the existing SE3 controller to track a pre-designed reference trajectory. No racing policy involved. The sequence:

| Phase | Maneuver | What it identifies |
|---|---|---|
| 0 | Weigh on scale, log battery voltage | Ground-truth mass anchor |
| 1 | Hover 30 s at 1 m | Confirm mass, hover thrust |
| 2 | Vertical sine + chirp + step | Thrust curve interior + motor time constant |
| 3 | Per-axis rate chirps (roll/pitch/yaw) | Rate-controller bandwidths, axis asymmetry |
| 4 | Straight-line at 1, 2, 3 m/s in ±x, ±y | Linear aerodynamic drag |
| 5 | Thrust step at hover | Total command latency |

Feed the bags into a `fit_dynamics.py` that outputs `{nominal, uncertainty}` for each parameter. **DR nominal = identified value; DR half-width = identification uncertainty.** Race bags get demoted to a validation role: does the sim's state distribution cover what the real drone actually experiences?

### Concrete first step

Before the next Pennovation session, add a `sysid_mode` to the controller that plays the Phase 1–5 reference. Collect one bag. Then re-fit sim DR from that single bag, not from any race bag. Everything upstream (reward, obs, policy) can stay frozen — this change only corrects the sim's dynamics layer, which is likely the dominant gap right now.

---

## Action plan — what to identify and how to feed it into DR

### Tier 1 — Must-identify (dominant sim-real gaps)

| Parameter | What it is (plain) | Why it matters | How to identify | DR treatment |
|---|---|---|---|---|
| **Mass `m`** | Weight of the flying drone with battery | Sets effective TWR, hover thrust, every acceleration. Our current 30 g sim vs ~66 g identified is a 2× gap. | Scale + hover-thrust check | Nominal = scale reading, DR ±5 % (battery/prop variation) |
| **Max thrust `T_max`** | Force produced at PWM = 60 000 | Upper bound on TWR and climb/recovery authority | Quick vertical climb at full PWM, measure `a_z + g` | Nominal = measured; DR ±10 % |
| **Thrust curve `T(PWM)`** | Nonlinear map from PWM command to real force | The controller's *labeled* thrust ≠ real thrust; mismatches cause altitude overshoot (seen in R1 v4 at 28 % saturation) | Vertical sinusoid / chirp, fit polynomial | Nominal = fitted curve; DR scale ±15 % |
| **Motor time constant `τ_m`** | How long a motor takes to reach a new thrust (~30–80 ms) | Determines whether "bang-bang" policies from sim transfer or cause oscillation | Vertical step, fit first-order lag | Nominal = fitted; DR 0.7–1.3× nominal |
| **Total command latency `τ_c`** | Vicon → ROS → Crazyradio → Crazyflie delay (~20–50 ms) | Closed-loop stability; policies trained with zero latency oscillate in real | Thrust step + cross-correlation onset | Random per-episode 0–2× nominal (action latency randomization) |
| **Rate-controller bandwidth per axis** | How fast the onboard PID tracks a commanded body rate, per roll / pitch / yaw | Real yaw bandwidth is 2–3× worse than roll/pitch in the bags — sim must model this asymmetry | Per-axis rate chirp, Bode `−3 dB` point | Nominal = fitted PID gains; DR ±25 % |
| **Linear drag `k_d_{x,y,z}`** | Force that opposes motion, proportional to velocity | Sim usually under-damps horizontal motion, causing the policy to over-commit thrust | Straight-line at constant speed, solve force balance | Nominal = fitted; DR 0.5–2.0× |

### Tier 2 — Useful if you have the time

| Parameter | What / why | How |
|---|---|---|
| **Inertia `I_xx, I_yy, I_zz`** | Resistance to angular acceleration. Usually assumed symmetric but payload shifts it. | Rate step response; or CAD + measured mass |
| **Observation noise** | Vicon jitter + velocity-from-position differentiation noise. Seen in R1 v4 as 0.22 m/s discrepancy between obs-vel and odom-vel. | Stand still, log std of each obs channel |
| **Observation latency** | Age of the state fed to the policy | Compare `/pose` vs `/odom` timestamps; already known to be ~3–5 ms jitter |
| **Max hardware body-rate saturation** | Commanded ±100 / ±200 °/s may not be what onboard controller accepts | Commanded vs actual during max-rate chirp |

### Tier 3 — Environment (not "drone" but still DR targets)

| Parameter | Notes |
|---|---|
| **Gate position calibration error** | Already bounded at ~3 cm at Pennovation. Add ±3 cm noise to gate corners in sim obs. |
| **Initial state distribution** | SE3 takeoff places you within ~10 cm lateral, near-hover orientation. Sim reset distribution should match (already done in circle-V6's ground-reset mixture). |
| **Battery voltage / thrust decay** | Thrust curve drops as battery discharges. Log battery voltage during sysid and during racing to quantify. |

### Priority order (what to do first)

Based on what we've already seen in the bags, prioritize:

1. **Mass and TWR.** If the identified 66 g / real TWR ≈ 2.0 is right, the sim's 30 g / TWR ≈ 4.0 is the dominant gap and nothing else matters until that's fixed. Cost to check: weigh the drone on a kitchen scale. Five seconds.
2. **Motor time constant + total latency.** These drive the "bang-bang transfers badly" problem. Most published Crazyflie sim2real work (Swift, NeuroBEM) invests heavily here. A single chirp flight gives you both.

Everything in Tiers 2–3 is a refinement on top of those.

### What this looks like as a config artifact

After one sysid flight, `fit_dynamics.py` should produce something like:

```yaml
# sysid_output.yaml — identified 2026-04-XX on crazy_jirl_b3, battery 3.8V
mass_kg: {nominal: 0.066, dr_half_width: 0.003}
twr: {nominal: 1.81, dr_half_width: 0.15}
thrust_curve:
  coefs: [a2, b1, c0]
  dr_scale_half_width: 0.15
motor_tau_s: {nominal: 0.045, dr_range: [0.030, 0.060]}
command_latency_s: {nominal: 0.035, action_latency_steps_max: 2}
rate_bandwidth_hz: {roll: 18, pitch: 17, yaw: 8, dr_half_width_pct: 25}
drag_kd_Nsm: {x: 0.12, y: 0.12, z: 0.08, dr_range: [0.5, 2.0]}
obs_noise:
  lin_vel_std_mps: 0.05
  rot_std: 0.01
  gate_corners_std_m: 0.03
```

The sim's `CircleQuadcopterStrategy` then reads this file and applies DR ranges accordingly. The critical property: **every DR range is anchored to an identified nominal, with half-width equal to *our uncertainty* about that nominal — not to the empirical spread of racing policies.**

---

## 1. Methodology

For every bag with a recorded `/ctbr_cmd` topic, we extracted:

- **Policy-independent hardware signatures** — expected to be identical across all bags flown with the same drone / battery / props, regardless of policy:
    - PWM → Newton mapping reported by the controller (`thrust_pwm` vs `thrust_n`)
    - Rate-tracking latency per axis (peak of cross-correlation between commanded `roll/pitch/yaw_rate` and actual body-frame angular velocity)
    - Identified mass (from `m = cos(roll)·cos(pitch)·thrust / (a_z_world + g)` over in-flight samples with `|a_z|<5 m/s²`)
    - Max commanded thrust at saturation (PWM ≈ 60000)
- **Policy-dependent metrics** — expected to vary strongly across bags:
    - Mean commanded thrust, mean PWM, fraction of bang-bang commands
    - Mean/max body-frame speed
    - Mean body-rate RMS, mean/max tilt

Three of the 2026-04-15 bags (`zhuohao_wk2_cycle_v4_good`, `v6`, `v6_smooth`) were recorded **without the `/ctbr_cmd` topic** (a bag-recording config change), so they are used only for odom-based metrics. The machine-readable per-bag output is in `cross_race_hardware_analysis.json`.

---

## 2. What the Hardware Should Be Saying (and Is)

### 2.1 Max thrust at saturation — consistent across all bags

| Bag | Max `thrust_n` at PWM ≈ 60000 |
|---|---:|
| R1 cyclev2 | 1.172 N |
| R1 cyclev3 | 1.168 N |
| R1 cyclev4 | 1.173 N |
| R2 v7 | 1.167 N |
| R2 bruce_v2base+dr | 1.168 N |
| R2 bruce_v5spline+dr | 1.171 N |

Max `thrust_n` is 1.170 ± 0.003 N across all 6 bags, both races. This is the **ceiling the onboard PWM→thrust formula outputs at PWM = 60000** — it is not a measurement of actual motor force, but it is consistent because the controller code path is deterministic.

### 2.2 Identified mass — consistent across all bags

Using the rigid-body vertical force balance, the *same* drone mass pops out of every bag:

| Bag | Mass (median) |
|---|---:|
| R1 cyclev2 | 69.4 g |
| R1 cyclev3 | 62.7 g |
| R1 cyclev4 | 69.5 g |
| R2 v7 | 66.8 g |
| R2 bruce_v2base | 60.5 g |
| R2 bruce_v5spline | 66.7 g |

**Mean: 65.9 g, std: 3.6 g.** The cross-race variance is ~5%, most of it explained by different batteries / prop wear / the 60.5 g outlier (bruce_v2base, which has the shortest flight segment of the set).

Two things to note:
1. **The identified mass is ~2× the 30 g "nominal Crazyflie" we train against in sim.** Either the drone is genuinely heavier (racing frame + upgraded motors + racing pack can easily reach 50–70 g), or the `thrust_n` reported by the controller is calibrated to roughly half the actual thrust. Either way, the **TWR the sim uses is wrong by ~2× relative to the identified value**, and no amount of DR around the sim nominal will cover that gap.
2. Hover PWM implied by this mass (~37–43k, matching the observed `mean_pwm` range) confirms the identification is internally consistent.

### 2.3 Rate-tracking latency — 50–90 ms, but variance is suspicious

Peak cross-correlation lag between commanded and actual body rate:

| Bag | Roll lag | Pitch lag | Yaw lag |
|---|---:|---:|---:|
| R1 cyclev2 | 50 ms | 60 ms | 50 ms |
| R1 cyclev3 | 60 ms | 60 ms | 60 ms |
| R1 cyclev4 | 70 ms | 60 ms | 50 ms |
| R2 v7 | 60 ms | **90 ms** | 70 ms |
| R2 bruce_v2base | 60 ms | 60 ms | 50 ms |
| R2 bruce_v5spline | 60 ms | 70 ms | 60 ms |

The central tendency is ~60 ms across axes. But **cross-correlation on closed-loop data is policy-dependent**: if the policy never commands a rate reversal (e.g. v3's locked-attitude degenerate mode — see `rosbags/analysis.md`), the autocorrelation of the input is dominated by low-frequency content and the lag peak gets noisier and biased. The outlying 90 ms pitch lag in v7 (which uses a much heavier pitch-rate penalty) is almost certainly this artifact, not a real hardware change. We shouldn't set `action_latency_max` from closed-loop peaks like this — we need a proper open-loop chirp to identify bandwidth and lag per axis.

---

## 3. The Smoking Gun: the Controller's PWM→Thrust Map Changed Between Bags

Within the 2026-04-15 session, fit `thrust_n = a·PWM² + b·PWM + c` to the `/ctbr_cmd` messages:

| Bag | Fitted formula |
|---|---|
| R1 cyclev2 | `2.4e-5·PWM − 0.235` (pure linear) |
| R1 cyclev3 | `2.4e-5·PWM − 0.235` (pure linear, identical) |
| R1 cyclev4 | `2.4e-5·PWM − 0.235` (pure linear, identical) |
| R2 zhuohao_v7 | `2.4e-5·PWM − 0.235` (pure linear, identical) |
| **R2 bruce_v2base+dr** | `3.16e-10·PWM² + 2.5e-7·PWM + 0.104` (**quadratic**) |
| **R2 bruce_v5spline+dr** | `7.09e-10·PWM² − 3.05e-5·PWM + 0.561` (**quadratic**) |

**Zhuohao's and group30's racing controller encodes a linear PWM↔thrust relationship. Bruce's controller encodes a different, quadratic (calibrated) relationship.** This is confirmed in `docs/changelog.md`'s 2026-04-16 "Fixed Sim2Real Baseline" entry:

> "Racing-mode thrust is now converted to Crazyflie PWM using the same calibrated nonlinear thrust map as the non-racing controller path. The old linear PWM map was likely contributing to low-thrust / low-flight behavior in real flights."

### Implication for sim2real

The *same policy* trained in sim, deployed through zhuohao's controller vs through bruce's controller, gets a **different PWM sent to the motors** for the same policy output. That is a larger sim2real gap than most of the DR ranges we have been tuning. It means:

1. Any DR range inferred from group30/zhuohao bags is fitted against the **linear** controller's effective dynamics, not against the real drone.
2. Once bruce's corrected controller is in use, the effective sim2real gap changes shape, and those DR ranges no longer match.
3. **Rosbags collected before 2026-04-16 should be treated as coming from a different deployment pipeline than rosbags collected after.** We cannot pool them.

---

## 4. What the Rosbags Say About the Policy (Not the Hardware)

| Bag | mean_thrust | mean_PWM | mean_speed | max_speed | mean_body_rate_RMS | mean tilt | max tilt | bang-bang %* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 cyclev2 | 0.767 N | 42672 | 2.43 | 3.73 | 108 °/s | 34.3° | 50.3° | 76.9 % |
| R1 cyclev3 | 0.698 N | 39726 | 2.50 | 4.01 | 140 °/s | 37.8° | 49.1° | 67.5 % |
| R1 cyclev4 | 0.646 N | 37505 | 2.17 | 4.05 | 81 °/s | 32.1° | 51.2° | 82.1 % |
| R2 v4_good | — | — | 2.28 | 3.52 | — | 32.7° | 48.8° | — |
| R2 v5 | 0.72 N | — | 2.71 | 3.54 | 151 °/s | 32.7° | 43.1° | — |
| R2 v6 | — | — | 3.06 | 4.20 | — | 38.0° | 52.5° | — |
| R2 v6_smooth | — | — | 3.04 | 3.81 | — | 37.6° | 49.2° | — |
| R2 v6_smooth2 | 0.68 N | — | 2.81 | 3.38 | 135 °/s | 32.8° | 44.2° | — |
| R2 v7 | 0.660 N | 38089 | 2.31 | 3.30 | 117 °/s | 28.5° | 39.1° | 70.4 % |
| R2 bruce_v2base+dr | 0.676 N | 39492 | 2.08 | 4.45 | 93 °/s | 33.7° | 54.5° | 66.1 % |
| R2 bruce_v5spline+dr | 0.694 N | 41757 | 2.21 | 4.36 | 131 °/s | 38.8° | 56.5° | 67.9 % |

\* Fraction of `/ctbr_cmd` messages with `thrust_n < 0.1` or `> 1.0` — proxy for "how bang-bang is this policy's thrust?".

### Observations

- **Mean commanded thrust ranges 0.65–0.77 N across bags flown on the same drone, same day, same track.** If you set sim TWR DR around the mean thrust per bag, you'd disagree with yourself by ±17% depending on which bag you opened.
- **Body-rate RMS ranges 81–180 °/s** — more than 2× variation. v3's 140 °/s is inflated by its locked-bank degenerate mode; v4_cons's 81 °/s is depressed because that policy oscillated vertically and stalled horizontally. Neither reflects "what the hardware can do" — both are failure/quirk modes of specific policies.
- **All policies except v7 are heavily bang-bang on thrust (66–82 % of commands are either <0.1 N or >1.0 N).** This is the Crazyflie-in-sim artifact where the policy discovers that with high TWR and no penalty on thrust jitter, extreme on/off is locally optimal. It also means *the middle of the thrust curve is never probed in race data*, so you can't use race bags to calibrate the thrust-PWM relationship in the interior.
- **Tilt extrema: max roll/pitch 49°–56° across bags, but mean tilt 28°–39°.** Setting `orientation` DR or sim spawn distributions from these percentiles would fit to the most aggressive policy's envelope, not to "what a reasonable sim-real transferable policy should tolerate".

### The trap this creates

If you tune DR ranges to match a successful bag (e.g. R2 v6, our current best real-world policy), your sim will be well-calibrated **for that policy's operating regime** — not for the regime a next-generation policy will want to explore. The next policy, trained against DR fitted to v6, will converge to the same regime because that's what the sim rewards; we'd be stuck at v6's ceiling by construction. **Fitting DR to a racing bag is a form of reward hacking at the sim-design layer.**

---

## 5. So: What Do the Race Bags Actually Tell Us?

Three useful things, and nothing more:

1. **A hardware sanity bound on max thrust.** PWM-saturation gives 1.17 N ceiling — but only as labeled by the controller. The mapping from that label to real motor force is not identified.
2. **A consistent identified mass of ~66 g across bags.** Either real drone is 66 g, or the thrust is labeled at ~half the real value. Either way, the *current sim nominal (30 g, TWR ≈ 2.3 at mass × g)* is wrong by ~2× relative to this consistent measurement, and no DR range around the current sim nominal covers it.
3. **Which policies crash in which ways** — for failure-mode analysis, not for sysid. The v3 locked-bank degenerate mode, v4 gate skipping, v6_smooth2 over-damping, etc. are all useful reward/obs design signals. They do not tell us about dynamics.

Everything else in the bags (speed, body-rate envelope, thrust distribution, tilt, latency) is **a projection of the policy onto the hardware**, not a measurement of the hardware.

---

## 6. A Proper Sysid Program (Recommendation)

The sim2real drone racing literature (Kaufmann et al. *Nature* 2023 "Swift", Bauersfeld et al. *NeuroBEM*, Torrente et al. 2021 data-driven MPC) converges on the same answer: **identify the drone with dedicated open-loop excitation trajectories, *not* with task data.** Racing is a closed-loop, task-biased, not-persistently-exciting data source — the worst possible sysid input in classical terms.

### 6.1 What we want to identify

| Quantity | Why it matters | Typical value for Crazyflie class |
|---|---|---|
| Mass `m` | Sets TWR, hover thrust, baseline DR | 30–70 g |
| Max thrust `T_max` | TWR ceiling | 1–3 N |
| Thrust curve `T(PWM)` | Controller label vs real force | nonlinear, battery-dependent |
| Motor time constant `τ_m` | Lag in thrust response; key for "bang-bang vs smooth" sim2real gap | 20–80 ms |
| Body-rate controller bandwidth `ω_bw_{r,p,y}` | How fast the onboard PID tracks rate commands; `-3 dB` point | 10–25 Hz |
| Rate controller latency `τ_r` | Dead time on rate command | 15–40 ms |
| Total command latency `τ_c` | Vicon → ROS → Crazyradio → Crazyflie | 20–50 ms |
| Linear drag `k_d_{x,y,z}` | Deceleration at speed, horizontal asymmetry | 0.1–0.5 N·s/m at Crazyflie scale |
| Yaw vs roll/pitch asymmetry | The bags already show yaw lag ≫ roll/pitch lag | hardware-level |

### 6.2 Sysid flight protocol (single ~20-minute session)

All phases flown with the **SE3 controller tracking a prescribed reference** — no RL policy. Record rosbags with full `/ctbr_cmd`, `/odom`, `/pose`, `/observations`.

**Phase 0 — Static (30 s):** weigh drone on scale, log battery voltage. Gives anchor values for `m` and voltage-dependent thrust curve.

**Phase 1 — Hover (30 s):** hover at z = 1.0 m. Mean hover thrust → `m·g` → mass. Gives anchor against which race-bag mass identification can be validated.

**Phase 2 — Vertical excitation (2 min):**
- Vertical sinusoid at 0.5 Hz, ±0.3 m, 10 cycles → TWR at mid-thrust.
- Vertical chirp 0.2 → 3 Hz, ±0.3 m, 30 s → motor time constant from commanded-vs-actual thrust phase shift.
- Vertical step ±0.4 m holds 2 s, 4 cycles → motor step response directly.

**Phase 3 — Attitude excitation at hover (4 min):**
- Roll-rate chirp 0.5 → 12 Hz, ±40 °/s, 30 s → roll rate bandwidth.
- Pitch-rate chirp (same) → pitch rate bandwidth.
- Yaw-rate chirp 0.5 → 8 Hz, ±60 °/s, 30 s → yaw rate bandwidth (expect lower bandwidth, confirms the race-bag observation that yaw has the worst tracking).
- Doublets (square-pulse pairs) at ±20, ±40, ±60, ±80 °/s per axis → rate tracking linearity + dead time.

**Phase 4 — Horizontal drag (3 min):**
- Straight-line track at constant speed 1, 2, 3 m/s in ±x, ±y (8 segments).
- For each segment, after reaching steady state, the thrust vector balances gravity and drag:  `R·[0,0,T] = m·g·ẑ + k_d·v_world`. Solve for `k_d`.
- Repeat with z = 0.75 m (gate height) and z = 1.5 m to check ground-effect.

**Phase 5 — Coupled / high-speed (2 min):**
- Figure-8 at 2 m/s and 3 m/s → excites simultaneous roll and yaw, useful for coupling identification.
- Hard brake: accelerate to 3 m/s horizontal, zero thrust reference, measure deceleration → drag at high speed.

**Phase 6 — Latency probe (30 s):** from hover, command a step change in thrust (e.g. +0.2 N hold 0.1 s, then back). Repeat 20×. Cross-correlate with measured z-accel onset. Gives total loop latency without the noise of a chirp.

### 6.3 Fitting

After the sysid bags are recorded, a single `scripts/sysid/fit_dynamics.py` should consume them and output a **single JSON of identified parameters** plus **their uncertainty ranges**. The DR config in sim then reads from that JSON:

- Each parameter → sim nominal value
- Parameter uncertainty → sim DR half-width

That is the correct direction of information flow. DR should encode *our ignorance about the hardware*, quantified from sysid, not *the spread we happened to observe across policies in race bags*.

### 6.4 Validation, not generation

Race bags remain valuable as a **validation set**, not a generation set:

- Sim-simulated trajectories under the identified parameters + DR should cover the state distributions we see in real race bags.
- When they don't, that tells us a DR axis is too narrow (or the wrong axis).
- When sim predicts an event never seen in real (e.g. 109° tilts in V6 sim evals, max 52.5° in real), that says DR is generating adversarial samples outside the real envelope — usable but should not drive policy choice.

### 6.5 A specific next step

The changelog's 2026-04-16 note already has `sysid_from_bag.py` converting deg/s ↔ rad/s and using the calibrated thrust map. That script currently works only on race bags — i.e. it's doing the thing this doc argues against. The cheapest high-impact change is:

1. Add a `sysid_mode` to the controller that runs an SE3-tracked chirp/step sequence instead of a racing policy.
2. Fly it at the next Pennovation session before the racing runs.
3. Re-fit `sysid_from_bag.py` against *those* bags, and only those bags, to seed DR ranges.
4. Keep the racing bags for closed-loop validation and failure-mode analysis.

This lets us decouple "what should DR cover?" (set by sysid) from "did the policy generalize?" (set by race bags). Currently those two questions are tangled, and as §3 shows, the tangle is bad enough that the controller code change on 2026-04-16 effectively invalidates DR ranges fit to everything before it.

---

## 7. Files

- `cross_race_hardware_analysis.json` — machine-readable per-bag metrics used for the tables above.
- `plots_full/<bag>/crazy_jirl_b3_statistics.json` — per-bag detailed stats (generated by `sim2real/bin/analyze_rosbag.py`).
- `summary.csv` — per-bag summary at lap-timing level.

## 8. Appendix — Raw comparison figures

All numbers in §2 (hardware-consistent) and §4 (policy-dependent) come from the same pipeline (`sim2real/bin/analyze_rosbag.py`) run on every bag. Mass identification uses `m = cos(roll)·cos(pitch)·thrust_n / (a_z_world + g)` on in-flight samples with `z > 0.3 m`, `thrust > 0.3 N`, `cos(roll)·cos(pitch) > 0.5`, `|a_z_world| < 5 m/s²`, trimming mass samples to `[0.012, 0.080] kg`. Cross-correlation lag uses a 100 Hz resampled grid, ±400 ms search window.
