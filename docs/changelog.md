# Changelog

All notable changes to this project will be documented in this file.

---

## [Ground-Only Checkpoint Selection Results] - 2026-04-27

Ran the final real-start selector with `REAL_GROUND_ONLY=1` on the three main
5000-iteration policy families.

All results below are from `1000/1000` ground starts. This is different from the
older mixed-reset batch evals, which had only about `210/1000` ground starts.

### `twr1p87-5000`
seed-42: 19.3s
seed-123 (only 1000 iterations): crash


Run:

```text
2026-04-24_10-16-47_powerloop-r1d1-gate3mask-fullswitch-twr1p87-5000-seed42
```

Selected checkpoint:

| Rank | Checkpoint | SR | Ground SR | Takeoff | Mean | Std | Best | Worst | Rule |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `best_model.pt` | 100.0% | 100.0% | 100.0% | **18.10 s** | 0.08 s | 17.92 s | 18.36 s | yes |
| 2 | `model_750_5415.pt` | 100.0% | 100.0% | 100.0% | 18.15 s | 0.07 s | 17.96 s | 18.36 s | yes |
| 3 | `model_700_5360.pt` | 100.0% | 100.0% | 100.0% | 18.18 s | 0.08 s | 17.98 s | 18.40 s | yes |
| 4 | `model_600_5433.pt` | 100.0% | 100.0% | 100.0% | 18.31 s | 0.12 s | 18.06 s | 18.62 s | yes |
| 5 | `model_2350_5416.pt` | 100.0% | 100.0% | 100.0% | 18.52 s | 0.08 s | 18.32 s | 18.74 s | yes |

Summary files:

```text
logs/rsl_rl/quadcopter_direct/2026-04-24_10-16-47_powerloop-r1d1-gate3mask-fullswitch-twr1p87-5000-seed42/batch_eval/checkpoint_selection_summary_ground_only.csv
logs/rsl_rl/quadcopter_direct/2026-04-24_10-16-47_powerloop-r1d1-gate3mask-fullswitch-twr1p87-5000-seed42/batch_eval/checkpoint_selection_summary_ground_only.json
```

Interpretation: this is the cleanest ground-start result so far. The original
`best_model.pt` remains the best checkpoint after the corrected selector.

### `gate0p8-5000`
19.55s

Run:

```text
2026-04-24_21-09-38_powerloop-r1d1-gate3mask-fullswitch-twr1p87-gate0p8-5000-seed42
```

Selected checkpoint:

| Rank | Checkpoint | SR | Ground SR | Takeoff | Mean | Std | Best | Worst | Rule |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `model_600_5477.pt` | 100.0% | 100.0% | 100.0% | **18.26 s** | 0.05 s | 18.12 s | 18.40 s | yes |
| 2 | `model_550_5464.pt` | 100.0% | 100.0% | 100.0% | 18.55 s | 0.06 s | 18.38 s | 18.72 s | yes |
| 3 | `best_model.pt` | 100.0% | 100.0% | 100.0% | 18.59 s | 0.09 s | 18.38 s | 20.08 s | yes |
| 4 | `model_2900_5515.pt` | 100.0% | 100.0% | 100.0% | 18.60 s | 0.17 s | 18.34 s | 22.16 s | yes |
| 5 | `model_500_5327.pt` | 100.0% | 100.0% | 100.0% | 19.09 s | 0.07 s | 18.94 s | 19.28 s | no |

Summary files:

```text
logs/rsl_rl/quadcopter_direct/2026-04-24_21-09-38_powerloop-r1d1-gate3mask-fullswitch-twr1p87-gate0p8-5000-seed42/batch_eval/checkpoint_selection_summary_ground_only.csv
logs/rsl_rl/quadcopter_direct/2026-04-24_21-09-38_powerloop-r1d1-gate3mask-fullswitch-twr1p87-gate0p8-5000-seed42/batch_eval/checkpoint_selection_summary_ground_only.json
```

Interpretation: `gate0p8` is clean under ground-only eval, but it does **not**
beat `twr1p87-5000 best_model.pt` on mean time. Deploy only if the real
controller observation gate is matched to `0.8 m`; otherwise do not fly it.

### `ground20-5000`

Run:

```text
2026-04-24_15-43-22_powerloop-r1d1-gate3mask-fullswitch-twr1p87-ground20-5000-seed42
```

Selected checkpoint:

| Rank | Checkpoint | SR | Ground SR | Takeoff | Mean | Std | Best | Worst | Rule |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `model_4700_5664.pt` | 100.0% | 100.0% | 100.0% | **18.59 s** | 0.16 s | 18.36 s | 20.24 s | yes |
| 2 | `best_model.pt` | 100.0% | 100.0% | 100.0% | 18.62 s | 0.10 s | 18.42 s | 19.30 s | yes |
| 3 | `model_4200_5625.pt` | 100.0% | 100.0% | 100.0% | 18.64 s | 0.13 s | 18.34 s | 18.92 s | yes |
| 4 | `model_4900_5696.pt` | 100.0% | 100.0% | 100.0% | 18.66 s | 0.16 s | 18.42 s | 21.94 s | no |
| 5 | `model_3800_5643.pt` | 100.0% | 100.0% | 100.0% | 18.66 s | 0.10 s | 18.44 s | 18.92 s | no |

Summary files:

```text
logs/rsl_rl/quadcopter_direct/2026-04-24_15-43-22_powerloop-r1d1-gate3mask-fullswitch-twr1p87-ground20-5000-seed42/batch_eval/checkpoint_selection_summary_ground_only.csv
logs/rsl_rl/quadcopter_direct/2026-04-24_15-43-22_powerloop-r1d1-gate3mask-fullswitch-twr1p87-ground20-5000-seed42/batch_eval/checkpoint_selection_summary_ground_only.json
```

Interpretation: `ground20` is clean, but slower than the base. It remains a
backup/safety candidate, not the primary speed policy.

Current deployment read:

- Primary sim-selected checkpoint: `twr1p87-5000 best_model.pt`.
- Real-validated fallback: `2026-04-22_00-53-52...twr1p87 best_model.pt`.
- Optional 0.8 observation candidate: `gate0p8-5000 model_600_5477.pt`, only if
  the real controller observation gate is explicitly set to `0.8 m`.
- Safety/backup candidate: `ground20-5000 model_4700_5664.pt` or
  `ground20-5000 best_model.pt`.

---

## [Ground-Only Final Eval Fix] - 2026-04-27

Finding:

- `batch_eval_race.py` runs with `env_cfg.is_train=True` to keep the reset and
  dynamics hooks active.
- Therefore reset sampling follows the training-style reset mixture unless
  overridden.
- With default `ground_reset_ratio=0.2`, a `1000`-env eval only has about
  `200` ground launches, e.g. `210/1000`. The rest are mid-course spline/linear
  resets.
- Real official testing starts from the ground, so the mixed-reset eval is a
  robustness diagnostic, not the final real-start selector.

Change:

- `batch_eval_race.py` now reports `Ground 3-Lap SR` and stores
  `ground_success_rate_pct` in the result JSON.
- `eval_powerloop_checkpoint_candidates.sh` now defaults to:

```bash
REAL_GROUND_ONLY=1
```

which merges these env overrides:

```json
{"ground_reset_ratio":1.0,"replay_reset_ratio":0.0,"real_start_reset_ratio":1.0}
```

- `train_powerloop_fullswitch_real_twr_final_attempt.sh` now defaults final
  candidate eval to `REAL_GROUND_ONLY=1`.
- `eval_powerloop_real_twr.sh` also now defaults to `REAL_GROUND_ONLY=1`, so
  direct calls such as `./scripts/run/eval_powerloop_real_twr.sh <run> <ckpt>
  1000` use `1000/1000` real-like ground starts by default.
- Ground-only checkpoint evals are saved under separate
  `batch_eval/<checkpoint>_ground_only/` directories so old mixed-reset JSONs
  are not accidentally reused.
- To intentionally run the old mixed-reset robustness eval, call the selector
  with `REAL_GROUND_ONLY=0`.
- Candidate selection now defaults to `CANDIDATE_STRATEGY=hybrid`: always
  include `best_model.pt`, use most slots for highest training-reward
  `model_*.pt` checkpoints across the full run, and reserve a few slots for
  latest/late/window anchors. This avoids both failure modes: missing
  mid-training policies such as `model_1900_5476.pt`, and letting one reward
  spike window monopolize all eval slots.
- To intentionally use only `best_model.pt` plus global reward top checkpoints,
  call the selector with `CANDIDATE_STRATEGY=global_top`.
- To use the older anchor-heavy behavior, call the selector with
  `CANDIDATE_STRATEGY=stratified`.

Final real-flight selection command for a `gate0p8` checkpoint family:

```bash
REAL_GROUND_ONLY=1 \
EXTRA_ENV_OVERRIDES='{"gate_side":0.8}' \
./scripts/run/eval_powerloop_checkpoint_candidates.sh \
  <run_dir> 1000 20 3000
```

This should print `Ground starts: 1000/1000`; otherwise it is not the final
real-start eval.

---

## [Checkpoint Selection Fix] - 2026-04-27

Finding:

- `best_model.pt` is selected by RSL-RL training reward only:
  `statistics.mean(rewbuffer)` in `on_policy_runner.py`.
- This does **not** match the real-flight selection criterion. It does not
  directly optimize 3-lap success rate, gate-hit robustness, mean lap time,
  standard deviation, or worst-case tail.
- Current example:
  `2026-04-27_01-43-18_powerloop-r1d1-gate3mask-fullswitch-twr1p87-final-tightdr-gate0p8-5000-seed42`
  saved `best_model.pt` around iteration `513` at
  `Train/mean_reward = 5746.86`. Its batch eval is poor:
  `84.7% SR`, `19.09 s` mean, `1.26 s` std, `28.44 s` worst.

Interpretation:

- The training reward is useful for optimization but too noisy for final
  checkpoint selection. A fast/aggressive checkpoint can win training reward
  while having bad gate-clearance tail behavior.
- For real-flight decisions, `best_model.pt` should be treated as one candidate,
  not the answer.

Change:

- Added:

```bash
scripts/run/eval_powerloop_checkpoint_candidates.sh
```

- The selector batch-evals `best_model.pt`, the latest saved checkpoint, and
  high-reward/later `model_*.pt` candidates, then ranks them by:
  success rate first, mean 3-lap time second, then std and worst-case tail.
- Updated `scripts/run/train_powerloop_fullswitch_real_twr_final_attempt.sh` so
  future automatic evals use checkpoint-candidate selection instead of only
  evaluating `best_model.pt`.

For the currently running `gate0p8` process, the already-started bash script may
still run the old single-`best_model.pt` eval after training finishes. Manually
run the selector after completion:

```bash
EXTRA_ENV_OVERRIDES='{"gate_side":0.8}' \
./scripts/run/eval_powerloop_checkpoint_candidates.sh \
  2026-04-27_01-43-18_powerloop-r1d1-gate3mask-fullswitch-twr1p87-final-tightdr-gate0p8-5000-seed42 \
  1000 8 2000
```

---

## [Final Tight-DR Overnight Partial Results] - 2026-04-27
20.88s

Status while this note was written:

- `final-tightdr-g20-linearvel0p6` completed training and batch eval.
- `final-tightdr-g20-twr1p99` was still training.
- `final-tightdr-gate0p8` and `final-tightdr-gate0p8-linearvel0p6` had not
  started yet.

Completed result:

| Candidate | Main change | SR | Takeoff SR | Mean 3-lap | Std | Best | Worst | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `final-tightdr-g20-linearvel0p6` | tight DR, `ground20`, linear reset velocity `0.2-0.6 m/s` | 98.3% | 98.5% | 19.46 s | 0.45 s | 18.50 s | 22.72 s | Reject as speed candidate. |

Comparison to the relevant references:

| Candidate | SR | Mean 3-lap | Std | Best | Interpretation |
|---|---:|---:|---:|---:|---|
| `twr1p87` base | 98.0% | 18.34 s | 0.48 s | 17.50 s | Still the real-validated speed baseline. |
| `twr1p87-5000` | 98.2% | 18.32 s | 0.41 s | 17.52 s | Longer base training did not hurt and is marginally cleaner in sim. |
| `ground20` | 99.0% | 18.70 s | 0.32 s | 17.74 s | Robustness/tail improvement, slower mean. |
| `gate0p8` | 98.9% | 18.58 s | 0.42 s | 17.74 s | Best clean speed/robustness sim ablation so far, if real observation gate is matched. |
| `ground20-linearvel1p2` | 97.8% | 18.68 s | 0.43 s | 17.84 s | Some speed signal, but SR/takeoff regression. |
| `final-tightdr-g20-linearvel0p6` | 98.3% | 19.46 s | 0.45 s | 18.50 s | Lower reset velocity recovered SR but removed the speed signal. |

Interpretation:

- The smaller linear reset velocity did not solve the tradeoff. Reducing
  `linear_reset_vel_max` from `1.2` to `0.6` recovered some reliability, but it
  also made the policy much slower. This suggests the `linearvel1p2` speed was
  not a clean distribution-match improvement; it was partly training a more
  aggressive state distribution that hurt reliability.
- Tightening dynamics DR did not make this candidate faster. For this policy
  family, the remaining speed gap is more likely path/geometry/raceline than
  "too much DR".
- Do not deploy `final-tightdr-g20-linearvel0p6`. If the TWR1.99 run also fails
  the selection rule, the only promising new runs left are the pure `gate0p8`
  tight-DR candidates.

Current practical decision:

- Keep existing `twr1p87` as the safest validated real policy.
- Keep existing `gate0p8` as the best sim-side speed candidate if the real
  controller observation gate can be changed to `0.8`.
- Do not spend more attempts on linear reset velocity unless there is time for
  a separate study; it is not a final-flight improvement.

---

## [Final Real-Flight Attempt Plan] - 2026-04-26

Added the final overnight training entry point:

```bash
scripts/run/train_powerloop_fullswitch_real_twr_final_attempt.sh
```

Default command:

```bash
./scripts/run/train_powerloop_fullswitch_real_twr_final_attempt.sh
```

This is the last-attempt script before the next real flight. It deliberately
does **not** add another Gate-3 detector, does **not** change the simple
real-world gate switch semantics, does **not** reintroduce dense velocity
reward shaping, and does **not** make DR larger. The final script uses tighter
vehicle DR than the earlier base scripts while keeping reset diversity.

Literature basis:

- Pasumarti, Bianchi, and Loquercio, *Agile Flight Emerges from Multi-Agent
  Competitive Racing*, report that sparse high-level racing objectives can
  produce aggressive flight and transfer better than manually prescribing
  progress/raceline behavior in the same sim/hardware setup:
  https://arxiv.org/abs/2512.11781
- The released UPenn codebase for that work trains the competitive policy in
  Isaac Lab/MAPPO, but reproducing that stack is too large for the final
  overnight attempt: https://github.com/Jirl-upenn/AgileFlight_MultiAgent
- Kaufmann, Bauersfeld, Loquercio et al., *Swift*, combine sim RL with
  real-world data-driven residual/noise models for transfer. We do not have
  enough clean open-loop sysid for a full residual model before the last flight,
  but this supports keeping the real-effective TWR/latency/noise calibration
  rather than guessing broad new DR:
  https://www.nature.com/articles/s41586-023-06419-4
- Ferede et al., *One Net to Rule Them All*, show the DR tradeoff directly:
  no randomization can fail transfer, while more randomization improves
  robustness but costs speed. For this project, the right move is still
  moderate, targeted DR around the identified vehicle, not huge randomization:
  https://arxiv.org/abs/2504.21586
- Loquercio et al., *Deep Drone Racing*, used domain-randomized simulation for
  zero-shot agile flight transfer, but in a modular perception/planning stack.
  The lesson relevant here is not to add a fancy gate detector; it is to keep
  the training/deployment interface consistent:
  https://arxiv.org/abs/1905.09727

Local evidence being acted on:

- `twr1p87` remains the only policy with strong real evidence: Apr22 official
  3-lap eval `19.004 s`, 5 ordered laps, mean lap `6.236 s`.
- Gate-3 approach-gating ablations failed to beat the base.
- Speed rewards (`speedv1` through `speedv5`) either reduced SR, slowed mean
  time, or failed in real bags. The best-case rollout time was not predictive
  of real success.
- `ground20` improved/tightened robustness but was slightly slower.
- `gate0p8` was already run twice. The complete run
  `2026-04-24_21-09-38_powerloop-r1d1-gate3mask-fullswitch-twr1p87-gate0p8-5000-seed42`
  trained with `gate_side=0.8`, `ground_reset_ratio=0.10`,
  `twr_randomization_pct=0.06`, aero DR `0.5-2.0`, and PID DR `0.15/0.30`.
  Batch result: `98.9%` SR, `18.58 s` mean, `0.42 s` std, `17.74 s` best.
  It was the fastest clean return-to-base sim ablation, but it requires
  matching real observation gate geometry.
- `ground20-linearvel` showed a real speed signal, but `linear_reset_vel_max=1.2`
  hurt SR/tail. The final script tests a smaller `0.6 m/s` cap.
- Apr22 b5 sysid/rosbag analysis estimates real TWR around `1.99`, while the
  current base is centered at `1.87`; a TWR-1.99 candidate is worth one run
  because it does not require controller changes.
- DR should stay targeted for the last flight. The final candidates tighten
  dynamics DR to `twr_randomization_pct=0.04`, aero `0.8-1.2`, PID `0.08/0.15`.
  Action-delay randomness stays on because fixed nominal eval profiles were too
  easy in the Apr22 backtest.

Final candidates trained by the script:

| Candidate | Change | Controller change? | Expected use |
|---|---|---|---|
| `final-tightdr-g20-linearvel0p6` | tight DR, `ground_reset_ratio=0.20`, linear reset velocity `0.2-0.6 m/s` | No | Lowest-risk new speed candidate if it recovers the `linearvel` speed without the 1.2 m/s tail regression. |
| `final-tightdr-g20` at `twr1p99` | tight DR, `ground_reset_ratio=0.20`, nominal `thrust_to_weight=1.99` | No | Tests whether centering training on Apr22 b5's real TWR improves speed without changing observations/reward. |
| `final-tightdr-gate0p8` | tight DR, pure `gate_side=0.8`, `ground_reset_ratio=0.10` | **Yes, observation gate must be 0.8** | Reruns the already-good 0.8 direction without forcing the slower `ground20` stitch. |
| `final-tightdr-gate0p8-linearvel0p6` | tight DR, pure `gate0p8 + linearvel0p6` | **Yes, observation gate must be 0.8** | Highest-upside candidate; only fly if batch eval is very clean. |

Selection rule before real flight:

- Existing `2026-04-22_00-53-52_powerloop-r1d1-gate3mask-fullswitch-twr1p87`
  remains the primary validated real policy.
- Existing `2026-04-24_21-09-38_powerloop-r1d1-gate3mask-fullswitch-twr1p87-gate0p8-5000-seed42`
  remains the reference 0.8 checkpoint; the final 0.8 runs must beat or match
  it before replacing it.
- Only promote a new final candidate if `eval_powerloop_real_twr.sh` gives
  roughly `SR >= 98.5%`, mean `<= 18.65 s`, std `<= 0.5 s`, and no ugly
  worst-case tail.
- A faster best-case alone is not sufficient; this already failed on speedv2.
- If the real controller stays at `SAFETY_MARGIN=0.15` / observation
  `gate_side=0.7`, do **not** deploy the `gate0p8` candidates.
- If deploying `gate0p8`, set the real controller observation virtual opening
  to `0.8 m` (for the current physical `gate_side=1.0`, that means
  `SAFETY_MARGIN=0.10` or equivalent). This does not change the simple physical
  plane-crossing gate switch; it only matches the policy observation geometry.

Recommended real-flight order if there is time for multiple policies:

1. Fly the existing `twr1p87` base first to re-establish the validated baseline.
2. Fly the best no-controller-change final candidate only if its batch eval
   passes the selection rule.
3. Fly `gate0p8` only if the controller observation gate was explicitly matched
   to `0.8`.
4. Treat the combined `gate0p8-linearvel0p6` as an experimental/high-upside
   run, not as the first race policy.

---

## [Real-Calibrated Eval Backtest Against Apr22 Bags] - 2026-04-26

Backtested the new `eval_powerloop_real_calibrated.sh` profile against the
three policies that were actually flown in `rosbags_apr22`.

Goal: check whether the rosbag/sysid-calibrated profile reproduces real-world
ranking before trusting it for deployment decisions.

Real Apr22 ranking:

| Policy | Real result |
|---|---|
| `twr1p87` base | Best: official 3-lap eval `19.004 s`, 5 ordered laps, mean lap `6.236 s`. |
| `speedv4-currentvel-g560` | Middle: 4 ordered laps, best lap `6.659 s`, no `/ctbr_cmd` so no official eval time. |
| `speedv2-postloopvel` | Worst: only 1 ordered lap, negative clearance at Gate 4, not a deployment candidate. |

Backtest: `b5_apr22` profile, 512 envs, ground-start real-profile eval.

| Policy | SR | Mean 3-lap | Std | Result |
|---|---:|---:|---:|---|
| `twr1p87` base | 100.0% | 18.24 s | 0.05 s | Stable, close-ish to real official time. |
| `speedv2-postloopvel` | 100.0% | **17.96 s** | 0.07 s | Incorrectly ranked as best, despite real failure. |
| `speedv4-currentvel-g560` | 100.0% | 18.57 s | 0.06 s | Incorrectly ranked behind speedv2. |

Backtest: `latency4` profile, 512 envs, same as `b5_apr22` but fixed 4-step
action delay.

| Policy | SR | Mean successful 3-lap | Result |
|---|---:|---:|---|
| `twr1p87` base | 100.0% | 19.19 s | Still robust. |
| `speedv2-postloopvel` | 45.7% | 18.39 s | Stress exposes fragility, but still not the real ranking. |
| `speedv4-currentvel-g560` | 15.2% | 21.67 s | Over-penalized relative to real, where it completed 4 ordered laps. |

Comparison to the older reset-diverse `eval_powerloop_real_twr.sh` artifacts:

| Policy | Old eval SR | Old eval mean | Old eval std | Match to real |
|---|---:|---:|---:|---|
| `twr1p87` base | 98.0% | 18.34 s | 0.48 s | Correctly keeps base as strongest real candidate. |
| `speedv2-postloopvel` | 96.2% | 18.68 s | 1.21 s | Flags higher variance / lower SR, but still underestimates real failure. |
| `speedv4-currentvel-g560` | 98.0% | 18.80 s | 0.43 s | More consistent with real than `b5_apr22`, though still not perfect. |

Conclusion:

- The `b5_apr22` real-calibrated profile is **too easy** and should not be used
  as a deployment-ranking metric. It removes too much reset/observation
  diversity and fails to expose the real speedv2 failure mode.
- The `latency4` profile is useful as a stress test, but it is not calibrated
  either; it over-penalizes `speedv4` relative to real.
- The older reset-diverse batch eval is still the better sim-side guardrail for
  ranking policies. It did not perfectly predict speedv2's real crash, but it
  at least showed speedv2 had worse SR and much worse tail variance.
- The gap is likely not just TWR/start/latency. The remaining real failure mode
  is probably tied to trajectory clearance / gate geometry / unmodeled
  rate-loop overshoot or contact sensitivity. A fixed nominal profile cannot
  expose that by itself.

Decision:

- Keep `eval_powerloop_real_calibrated.sh` as a diagnostic slice only.
- Do not use `b5_apr22` profile to select the race policy.
- For deployment ranking, require agreement between:
  - reset-diverse `eval_powerloop_real_twr.sh`,
  - targeted stress checks such as latency / gate bias / velocity bias,
  - and real bag evidence when available.

---

## [Real-Calibrated Eval Profile] - 2026-04-26

Added a separate eval entry point for checking policies under a more realistic
single-drone profile instead of using the existing reset-diverse batch metric
as the only decision signal:

```bash
scripts/run/eval_powerloop_real_calibrated.sh
```

Why this is needed:

- `eval_powerloop_real_twr.sh` is not broad dynamics DR: it already uses
  `--num_params_per_env 0` and `twr_randomization_pct=0.0`.
- However, because `batch_eval_race.py` runs with `is_train=True`, the old
  batch metric still includes training-style observation noise and random
  reset distribution. It is a useful robustness/screening metric, but it is not
  the same as the real race start.
- The real system is one fixed vehicle on one day. Rosbags should therefore
  define a fixed nominal profile plus a few targeted stress checks, not a huge
  per-env dynamics randomization.

Default profile: `b5_apr22`.

| Field | Value | Source / reason |
|---|---:|---|
| `thrust_to_weight` | `1.99` | Apr22 b5 bags: max thrust `1.174 N`, pooled mass `~60.2 g`, so `1.174 / (0.0602*9.81) ~= 1.99`. |
| `fixed_action_delay_steps` | `3` | Clean Apr22 rate-lag estimate `60-70 ms`; policy rate is 50 Hz, so 3 steps is about `60 ms`. |
| `ground_reset_ratio` | `1.0` | Real race starts from the ground, not random mid-track states. |
| `real_start_reset_ratio` | `1.0` | Use the narrow launch-pose distribution instead of the full training ground-start spread. |
| `obs_noise_std_scale` | `0.0` | Disable training-scale observation noise for real-profile eval. |
| `obs_lin_vel_noise_std` | `0.005` | Sysid stationary observation velocity noise is about `0.005 m/s`, not `0.05 m/s`. |
| `obs_lin_vel_bias` | `[0.001, -0.0006, 0.0]` | Sysid stationary observation velocity bias mean. |
| Gate switching | simple plane crossing | `approach_x_threshold=0.0`, `backtrack_check_enabled=false`, matching the deployed controller semantics. |

Available profiles:

| Profile | Purpose |
|---|---|
| `b5_apr22` | Default profile for the latest successful real powerloop session. |
| `b3_apr20` | Older b3 profile with `thrust_to_weight=1.87`, but the same fixed 3-step delay and real-start eval shape. |
| `latency4` | Stress check for the high side of the measured `60-70 ms` lag. |
| `legacy` | Approximate the previous `eval_powerloop_real_twr.sh` profile for A/B comparison. |

Recommended use:

```bash
./scripts/run/eval_powerloop_real_calibrated.sh \
  2026-04-22_00-53-52_powerloop-r1d1-gate3mask-fullswitch-twr1p87 \
  best_model.pt 1000 b5_apr22
```

Interpretation rule:

- Use `eval_powerloop_real_twr.sh` as a reset-diverse robustness screen.
- Use `eval_powerloop_real_calibrated.sh b5_apr22` only as a diagnostic
  fixed-profile slice. The backtest above shows it is not reliable for
  deployment ranking by itself.
- If these disagree, do not automatically trust the real-calibrated profile;
  inspect the failure mode and prefer the reset-diverse robustness metric plus
  real bag evidence.
- This still does not replace real bags: race rosbags are closed-loop and
  policy-biased, so they can calibrate TWR/start/latency/noise envelopes, but a
  clean open-loop sysid bag is still needed for motor tau, aero, and rate-loop
  dynamics.

---

## [Post-Gate3 Reset-Speed Overnight Results] - 2026-04-26

Completed the four reset-distribution speed ablations from
`train_powerloop_fullswitch_real_twr_overnight_realstart_speed.sh`.

Evaluation setup:

- Checkpoint: `best_model.pt`
- Eval script: `scripts/run/eval_powerloop_real_twr.sh`
- Eval size: `1000` envs for the four new runs; reference rows below use the
  prior matched real-TWR eval artifacts.
- Shared base: `fullswitch + real-effective TWR=1.87`, same base reward,
  `replay_reset_ratio=0.25`, `ground_reset_ratio=0.20`, staged replay.
- These runs did not change the real controller and did not add velocity reward.

Batch eval results:

| Candidate | Success | Takeoff SR | Mean 3-lap | Std | Best | Worst | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| current `twr1p87` base reference | 98.0% | 96.7% | **18.34 s** | 0.48 s | **17.50 s** | 24.24 s | Still the speed reference and real-validated base. |
| `ground20-5000` reference | **99.0%** | **99.0%** | 18.70 s | **0.32 s** | 17.74 s | **20.94 s** | Still the clean robustness reference. |
| `ground20-realstart60-5000` | 98.7% | 98.6% | 19.45 s | 0.41 s | 18.68 s | 22.86 s | Reject for speed; maybe useful only as a deployment-start idea. |
| `ground20-postloopfocus-5000` | 98.9% | 98.6% | 19.10 s | 0.41 s | 18.22 s | 22.36 s | Stable, but slower than both base and `ground20`. |
| `ground20-linearvel-5000` | 97.8% | 96.6% | 18.68 s | 0.43 s | 17.84 s | 22.78 s | Fastest of this batch, but SR/tail regress. |
| `ground20-splinevel-5000` | 67.5% | 97.5% | 20.50 s | 1.61 s | 18.50 s | 29.58 s | Reject; completion robustness collapsed. |

Interpretation:

- None of the four reset-speed ablations beats the current real base. The
  original `twr1p87` policy remains the speed reference and is still the only
  policy already validated by the Apr22 real bag.
- `ground20-linearvel` partially supports the hypothesis that zero-velocity
  mid-track resets are suboptimal: it is the fastest new candidate and roughly
  matches `ground20` mean time. But the cost is real: `97.8%` SR, worse takeoff
  SR, worse std, and worse tail. It is not a replacement candidate as-is.
- `ground20-realstart60` did not translate real-start matching into speed. It
  keeps high SR but slows the mean by about `+0.75 s` vs `ground20` and
  `+1.13 s` vs the current base. Narrowing ground starts likely reduces useful
  reset diversity and teaches a more conservative launch/first-gate behavior.
- `ground20-postloopfocus` is safer than `linearvel` but still too slow. More
  resets on target gates `[5,6,0]` improve neither mean time nor tail enough to
  justify replacing `ground20`.
- `ground20-splinevel` is a clear negative result. Moving spline resets with
  `1.0-3.0 m/s` tangent velocity create a reset distribution the current reward
  and policy optimization do not handle; first-gate/takeoff is fine, but many
  episodes fail before completing 3 laps.

Decision:

- Keep `2026-04-22_00-53-52_powerloop-r1d1-gate3mask-fullswitch-twr1p87` as
  the real-flight base.
- Keep `ground20-5000` as the robustness/takeoff reference, not a speed
  replacement.
- Do not deploy `realstart60`, `postloopfocus`, or `splinevel`.
- Do not deploy `linearvel` unless a follow-up run recovers SR to roughly
  `98.5-99%` while preserving the `18.6-18.7 s` mean.

Next direction:

- Stop adding more reset-complexity around Gate 3 or spline states for now.
- If continuing speed work, use the validated base and test smaller,
  lower-risk changes: slightly wider virtual gate (`gate0p8`) with matching
  real observation gate, mild time pressure, or a lower `linear_reset_vel_max`
  such as `0.6-0.8 m/s` to see whether `linearvel` can keep its speed without
  the SR/tail regression.

---

## [Post-Gate3 Reset-Speed Overnight Plan] - 2026-04-25

The Gate 3 approach-gating ablations did not beat the restored
`fullswitch + real-effective TWR=1.87` base. The real controller already uses
the simple physical rule that matters onsite: after the drone crosses the gate
plane inside the gate window, the gate is counted. Training should therefore
move away from more Gate 3 switch tricks and back to reset distributions that
match real deployment and the slow real segments.

Latest sim batch-eval summary:

| Candidate | Success | Mean 3-lap | Interpretation |
|---|---:|---:|---|
| current `twr1p87` base | 98.2% | **18.32 s** | Still the speed reference. |
| `ground20-5000` | **99.0%** | 18.70 s | More robust/tighter tail, not faster. |
| `g3approach0p1-5000` | 86.1% | 19.26 s | Too much SR loss. |
| `g3approach0p0-5000` | 98.3% | 20.18 s | SR recovers, but clearly slower. |

Added:

```bash
scripts/run/train_powerloop_fullswitch_real_twr_overnight_realstart_speed.sh
```

Default command:

```bash
./scripts/run/train_powerloop_fullswitch_real_twr_overnight_realstart_speed.sh
```

This sequentially trains four 5000-iteration candidates, all preserving the
same base reward and real-effective dynamics:

| Candidate | Change | Intended effect |
|---|---|---|
| `ground20-realstart60` | `ground_reset_ratio=0.20`; 60% of ground resets use a narrow real launch-pose distribution | Better takeoff/start behavior and potentially faster official start-to-Gate0 time. |
| `ground20-postloopfocus` | `segment_focus_reset_ratio=0.50`; focus target gates `[5,6,0]` | More practice on the slow/fragile post-loop real segments without adding a velocity reward. |
| `ground20-linearvel` | linear fallback resets start with `0.3-1.2 m/s` velocity toward the target gate | Reduce train/test mismatch from zero-velocity mid-track resets; should be the cleanest speed-oriented reset ablation. |
| `ground20-splinevel` | enable spline resets with `1.0-3.0 m/s` tangent velocity | More realistic moving-state coverage, but higher risk of hurting success rate. |

Selection rule: only consider a candidate better than the current base if it
keeps sim success around the current `98-99%` range and improves mean time or
tail time. `ground20-realstart60` is mainly a real deployment/start candidate;
`postloopfocus`, `linearvel`, and `splinevel` are the speed candidates.

Implementation details:

- Added reset knobs for real-start ground rows, segment-focused linear resets,
  and optional target-directed velocity on linear fallback resets.
- Added reset-sampled observation bias ranges for future robustness ablations,
  but the overnight speed script leaves these off.
- Added TensorBoard logging for the new reset/bias config values.

---

## [Gate0p8 vs Ground20 and New Gate3 Direction] - 2026-04-25

Ran the two clean overnight ablations on top of the restored
`fullswitch + real-effective TWR=1.87` base.

Batch results:

| Candidate | Takeoff SR | Overall SR | Mean 3-lap | Std | Best | Worst | Max tilt mean / p95 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `twr1p87-gate0p8-5000` | 98.8% overall | 98.9% | **18.58 s** | 0.42 s | 17.74 s | 24.08 s | 59° / 63° | Slightly faster than `ground20`, still very high SR. |
| `twr1p87-ground20-5000` | **99.0% overall** | **99.0%** | 18.70 s | **0.32 s** | 17.74 s | **20.94 s** | 59° / 62° | Slightly slower mean time, but tighter tail and slightly higher robustness. |

Shared observation:

- Both runs still show `Peak body-rate: mean=1.00 of max`, i.e. the controller
  is still saturating rate authority in essentially all successful episodes.
- The two changes help different axes:
  - `gate0p8` buys a small speed gain;
  - `ground20` buys a slightly safer and tighter completion distribution.

Interpretation:

- Neither result is a clear enough win to replace the current real base on its
  own.
- `gate0p8` looks like the better pure speed candidate.
- `ground20` looks like the better takeoff / deployment-robustness candidate.
- The next more meaningful direction is not another small gate/reset tweak, but
  reducing the training-only Gate 3 constraints that do not exist in the real
  controller and may push training toward an unnecessarily conservative loop
  line.

New direction:

- `eval_powerloop_real_twr.sh` was updated so eval behaves closer to the real
  controller's simple physical gate-switch semantics at Gate 3:
  - `approach_x_threshold=0.0`
  - `backtrack_check_enabled=false`
- New planned training ablation:
  - first reduce `approach_x_threshold` from `0.3 -> 0.1`
  - then disable it entirely with `approach_x_threshold=0.0`
- This isolates whether the current speed/robustness ceiling is being limited
  by training-only Gate 3 constraints rather than by TWR, gate size, or reset
  distribution.

Added:

```bash
scripts/run/train_powerloop_fullswitch_real_twr_gate3switch_ablation.sh
```

This sequentially trains two clean candidates from the same `twr1p87` base:

| Candidate | Change | Purpose |
|---|---|---|
| `twr1p87-g3approach0p1-5000` | `approach_x_threshold: 0.3 -> 0.1` | Reduce Gate 3 switch strictness without fully removing it. |
| `twr1p87-g3approach0p0-5000` | `approach_x_threshold: 0.3 -> 0.0` | Remove the Gate 3 sim-only approach gate and align training more closely with the real controller's simple plane-crossing switch semantics. |

---

## [Powerloop Return-to-Base Overnight Plan] - 2026-04-23

Decision after Apr22 real bag evaluation: return to the plain
`fullswitch + real-effective TWR=1.87` policy as the base. The speed reward
variants did not beat it in real flight.

Added:

```bash
scripts/run/train_powerloop_fullswitch_real_twr_overnight_return_base.sh
```

Default command:

```bash
./scripts/run/train_powerloop_fullswitch_real_twr_overnight_return_base.sh
```

This sequentially trains three 5000-iteration candidates:

| Candidate | Change from current best base | Purpose |
|---|---|---|
| `twr1p87-5000` | Same `twr1p87` base, train for 5000 iterations | Check whether longer training improves the already best real policy. |
| `twr1p87-ground20-5000` | `ground_reset_ratio: 0.10 -> 0.20` | More ground-start coverage for real takeoff/deployment robustness. |
| `twr1p87-gate0p8-5000` | `gate_side: 0.70 -> 0.80` | Test a slightly less conservative virtual gate / reward window. |

The `ground20` and `gate0p8` runs are separate ablations, not cumulative.

Implementation note: `gate_side` lives at `env_cfg.gate_model.gate_side`, not
as a top-level env config field. Added dotted/nested ENV override support to
`train_race.py`, `eval_race.py`, `play_race.py`, and `batch_eval_race.py`, with
`"gate_side": 0.8` as a convenience alias for
`"gate_model.gate_side": 0.8`.

`eval_powerloop_real_twr.sh` now accepts `EXTRA_ENV_OVERRIDES` and merges it
into the fixed real-TWR eval config. The overnight script uses this so the
`gate0p8` checkpoint is evaluated with `gate_side=0.8`, matching its training
observation/reward gate.

If the `gate0p8` checkpoint is used in real flight, the real controller's
observation gate should also use a `0.8 m` virtual opening; otherwise the
policy was trained with a different gate observation than it receives on the
drone.

---

## [Apr22 Real Powerloop Bag Eval] - 2026-04-23

Reprocessed the `rosbags_apr22` real powerloop bags after two analysis fixes:

- `scripts/run/batch_test_bags.sh` and `scripts/run/test_bag.sh` now default to
  `namespace=auto`, because the Apr22 bags use `/crazy_jirl_b5/*` rather than
  the old `/crazy_jirl_b3/*`.
- The real powerloop gate coordinates used by the bag-analysis scripts were
  corrected to match the controller target centers recovered from
  `/observations`. The previous hard-coded powerloop coordinates were shifted
  by about `0.25 m` in x, which made the strong base run look like missed gates.

Final batch command:

```bash
bash scripts/run/batch_test_bags.sh rosbags_apr22
```

Final result:

| Real bag | Official eval | Ordered laps | Ordered passes | Valid passes | Best lap | Mean lap | Std lap | Min clearance | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `twr1p87` base | **19.004 s** | **5** | **41** | **52** | **6.121 s** | **6.236 s** | 0.144 s | 0.032 m @ Gate 3 | Only run with valid race-start command and >=3 completed laps. |
| `speedv2-postloopvel` | n/a | 1 | 12 | 15 | 6.707 s | 6.707 s | 0.000 s | -0.040 m @ Gate 4 | Did not complete 3 ordered laps; not a deployment candidate. |
| `speedv4-currentvel-g560` | n/a | 4 | 30 | 38 | 6.659 s | 6.791 s | 0.128 s | **0.110 m @ Gate 4** | Completed laps, but `/ctbr_cmd` was not recorded, so official race-start timing is unavailable. |

Official eval definition remains:

```text
first /ctbr_cmd race command -> 3rd lap's last gate (Gate 6)
```

For the `twr1p87` base bag:

```text
race_start_cmd_t = 7.186 s
3rd-lap Gate 6  = 26.190 s
official eval   = 19.004 s
```

Segment-time comparison from ordered passes:

| Real bag | S->0 | 0->1 | 1->2 | 2->3 | 3->4 | 4->5 | 5->6 | 6->0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `twr1p87` base | 0.858 | 0.902 | 0.856 | **1.341** | **0.708** | 1.002 | **0.677** | **0.737** |
| `speedv2-postloopvel` | 0.831 | 1.054 | 1.544 | 1.601 | 0.816 | 1.028 | 0.773 | 0.708 |
| `speedv4-currentvel-g560` | n/a | 0.972 | 0.897 | 1.516 | 0.805 | **0.950** | 0.783 | 0.859 |

Interpretation:

- The real result agrees with the sim batch-eval conclusion: the plain
  `twr1p87` fullswitch base remains the best current real-flight policy.
- `speedv4` has larger gate margins and completed 4 ordered laps, but it is
  slower on mean lap time and lacks `/ctbr_cmd`, so it cannot replace the base
  as the official timed result.
- `speedv2` is clearly worse: only 1 ordered lap and one negative-clearance
  pass. This matches its weaker sim robustness.
- The base's tightest part is Gate 3 / Gate 4 / Gate 5 clearance, especially
  Gate 3 minimum clearance `0.032 m`. It is fast and successful, but still
  close to the frame in the loop section.

Current recommendation: keep
`2026-04-22_00-53-52_powerloop-r1d1-gate3mask-fullswitch-twr1p87` as the real
powerloop base. Future speed work should not use the current `speedv2` or
`speedv4` checkpoints as base; instead, preserve the `twr1p87` reward/dynamics
and make smaller, isolated changes around line radius or reset distribution.

Plots for the base bag were generated by:

```bash
bash scripts/run/test_bag.sh \
  rosbags_apr22/group30_powerloop-r1d1-gate3mask-fullswitch-twr1p87 \
  auto 3
```

Output directory:

```text
rosbags_apr22/plots/group30_powerloop-r1d1-gate3mask-fullswitch-twr1p87/
```

---

## [SysID Controller Safety Guard] - 2026-04-23

Updated `docs/controller_simple_policy.py`, the sysid drop-in controller, after
the real sysid attempt where the drone reportedly took off but dropped/dragged
when sysid began.

Controller-side fixes:

- Added an altitude gate after the takeoff ramp: if the drone has not reached
  the hover target within `takeoff_ready_z_tol_m=0.20`, sysid holds hover and
  does not enter lateral/roll/pitch/yaw/drag probes.
- Removed the dangerous zero-thrust fallback path. If the policy update or SE3
  call fails, the controller now commands a hover fallback instead of
  `cmd_thrust=0.0`.
- The hover fallback now prefers the sysid center once sysid has started, so an
  exception tries to recover to the safe hover altitude instead of holding a
  low/falling current pose.
- Added an open-loop hover-thrust fallback based on vehicle mass and the same
  controller thrust scaling (`0.038 * 9.81 * 3.15`) in case SE3 itself fails.

Interpretation: this change does not prove whether the previous failed sysid
bag was caused by insufficient thrust. That still requires checking PWM /
`cmd_thrust_n` during the drop. It does fix a controller-side failure mode where
an exception could make the drone fall even though the vehicle has enough thrust
to hover.

---

## [Powerloop Eval Gate-Switch Detector Fix] - 2026-04-22

Investigated a visual/eval gate-switch issue where the drone could visibly pass
the third/powerloop gate but the red target marker did not advance, forcing the
policy to fly back and cross again.

Root cause: the switch detector checked the sampled pose after crossing the
gate plane:

```text
prev_x > 0 and curr_x <= 0 and |curr_y| < 0.5 and |curr_z| < 0.5
```

At racing speed, the actual segment can cross the gate opening, but the next
sample can already be slightly outside the window. This is a sampling artifact,
not necessarily a bad physical pass.

Implemented change:

- Added `env._prev_pose_drone_wrt_gate` so the detector has the full previous
  gate-frame pose, not only previous `x`.
- Target switching now uses a line-segment interpolation detector that checks
  the `prev -> curr` crossing point at the `x = 0` gate plane.
- The Gate-3 approach-zone anti-exploit check is training-only. Eval/play uses
  physical gate-switch semantics so the red target marker advances on a valid
  physical pass.

Important correction: the poor `86.7%` / `20.39 s` batch result reported after
this investigation should be attributed to the speed/reward candidate being
tested, not to this detector fix. The detector fix is retained.

## [Powerloop Fullswitch Real-Effective TWR Candidate] - 2026-04-22

Added `scripts/run/train_powerloop_fullswitch_real_twr.sh` as a conservative
A/B training entry for the next powerloop candidate. Added matching
`scripts/run/eval_powerloop_real_twr.sh` so the candidate can be evaluated under
the same nominal dynamics.

This script keeps the current successful `R1+D1 + Gate-3 mask + fullswitch`
logic and changes only the dynamics parameter with the strongest repeated
evidence from real bags: effective thrust authority.

- Nominal `thrust_to_weight`: `3.15 -> 1.87`
- TWR randomization: narrow `±6%`
- `mass_variation`: kept off, because the current implementation changes the
  commanded thrust scale rather than the actual rigid-body mass/inertia.
- Motor-tau and drag changes: kept off until there is a clean airborne sysid
  bag.

Rationale: real deployment maps policy thrust percentage to about `1.174 N`
maximum command, while the real powerloop/circle bags consistently indicate an
effective vehicle mass near `64 g`. That corresponds to an effective TWR of
`1.174 / (0.064 * 9.81) ~= 1.87`. This is a cleaner test than widening DR around
the old `3.15` nominal.

Run:

```bash
./scripts/run/train_powerloop_fullswitch_real_twr.sh 3000 8192
./scripts/run/eval_powerloop_real_twr.sh <run_dir> best_model.pt 768
```

### First sanity eval on existing fullswitch checkpoint

User-reported result from running the existing `fullswitch` checkpoint under
`eval_powerloop_real_twr.sh`:

| Eval config | Time | Gates | Laps |
|---|---:|---:|---:|
| `powerloop`, `thrust_to_weight=1.87` | `21.16 s` | 22 | `3/3` |

This makes the TWR change look directionally correct: the old checkpoint still
finishes the 3-lap task under the lower real-effective thrust authority, and the
behavior is reported as qualitatively closer to real flight. This is not yet a
trained-policy conclusion, but it is enough evidence to justify starting a new
training run with `thrust_to_weight=1.87` as the nominal dynamics.

### First eval on newly trained TWR-1.87 checkpoint

Run:

```bash
./scripts/run/eval_powerloop_real_twr.sh \
  2026-04-22_00-53-52_powerloop-r1d1-gate3mask-fullswitch-twr1p87 \
  best_model.pt 1
```

User-reported result:

| Eval config | Time | Gates | Laps |
|---|---:|---:|---:|
| new `twr1p87` policy, `thrust_to_weight=1.87` | `18.76 s` | 22 | `3/3` |

The same policy was reported to perform poorly under the old nominal
`eval_powerloop.sh` setup (`thrust_to_weight=3.15`), with example partial runs
ending around `7` gates / `1` lap and `3` gates / `0` laps. This is acceptable
for this candidate: the purpose of the run is to specialize the policy around
the real-effective thrust authority, not to remain optimal under the old
over-powered sim nominal. The old-nominal failure should be tracked as reduced
cross-TWR generality, but it is not a blocker for real deployment if the real
controller remains on the same thrust scale used in the 04-20 bags.

Batch eval under the matched real-effective TWR config:

```bash
./scripts/run/eval_powerloop_real_twr.sh \
  2026-04-22_00-53-52_powerloop-r1d1-gate3mask-fullswitch-twr1p87 \
  best_model.pt 768
```

User-reported result:

| Metric | Value |
|---|---:|
| Trials | 1 |
| Envs/trial | 768 |
| Takeoff SR, ground starts | 100.0% |
| Takeoff SR, overall | 96.7% |
| First-gate SR, ground starts | 100.0% |
| First-gate SR, overall | 100.0% |
| Overall 3-lap SR | 98.0% (`753/768`) |
| Mean 3-lap time | 18.34 s |
| Std 3-lap time | 0.48 s |
| Best / worst 3-lap time | 17.50 s / 24.24 s |

Interpretation: this is no longer a single-rollout result. The TWR-1.87
fullswitch policy is strong under the dynamics it was trained for. Further
experiments should keep `thrust_to_weight=1.87` as the nominal; speed work
should change path/reward shaping, while robustness work should change
observation/control mismatch tests rather than broad TWR again.

### Speed candidate v1 — not recommended

Added `scripts/run/train_powerloop_fullswitch_real_twr_speed.sh` for the next
racing-speed ablation.

This keeps the validated `fullswitch + thrust_to_weight=1.87` dynamics and does
not change `gate_side`, Gate-3 progress masking, or switching semantics. The
first speed attempt only changes reward pressure:

| Parameter | Base TWR-1.87 | Speed candidate |
|---|---:|---:|
| `progress_goal_reward_scale` | 20.0 | 30.0 |
| `lap_complete_reward_scale` | 0.5 | 0.75 |
| `lap_incomplete_penalty_scale` | -0.05 | -0.08 |
| `cmd_reg_rp_scale` | -1.0 | -0.7 |
| `cmd_reg_yaw_scale` | -0.5 | -0.35 |
| `cmd_smoothness_scale` | -0.1 | -0.05 |

Rationale: the real bag comparison showed the safer `fullswitch` line mostly
lost time after the loop (`4->5`, `5->6`, `6->0`), while Gate 3 / Gate 4
clearance was better. Therefore the first speed attempt should not make Gate 3
more aggressive or enlarge the virtual reward gate. It should increase overall
time pressure and allow slightly stronger body-rate/action changes, then be
screened with the same `eval_powerloop_real_twr.sh` batch metric.

Run:

```bash
./scripts/run/train_powerloop_fullswitch_real_twr_speed.sh 3000 8192
./scripts/run/eval_powerloop_real_twr.sh <speed_run_dir> best_model.pt 768
```

User-reported batch result:

| Metric | Base TWR-1.87 | Speed v1 |
|---|---:|---:|
| Envs | 768 | 1000 |
| Overall 3-lap SR | **98.0%** | 86.7% |
| Mean 3-lap time | **18.34 s** | 20.39 s |
| Std 3-lap time | **0.48 s** | 2.12 s |
| Best 3-lap time | 17.50 s | 17.56 s |
| Worst 3-lap time | 24.24 s | 29.88 s |

Conclusion: this is a reward-design failure, not a gate-switch detector failure.
The broad pressure change made the policy less reliable and did not improve
the successful-time distribution. Do not use this script as the next base.

### Speed candidate v2 — targeted post-loop velocity shaping

Added optional `vel_toward_gate` shaping to the current
`CircleQuadcopterStrategy` with default scale `0.0`, so existing base/eval
behavior is unchanged unless a training script explicitly enables it.

Added `scripts/run/train_powerloop_fullswitch_real_twr_speed_v2.sh` as the next
speed candidate. This keeps the validated `fullswitch + TWR=1.87` base reward
unchanged and only adds a small racing-line velocity term:

| Parameter | Value |
|---|---:|
| `vel_toward_gate_reward_scale` | 0.75 |
| `vel_reward_gate_indices` | `[4, 5, 6, 0]` |
| `vel_reward_min_gates_passed` | 1 |
| `vel_reward_next_weight` | 0.375 |
| `vel_reward_clamp_min/max` | `-2.0 / 8.0` |

Rationale from the old `changelog_powerloop.md`: the stable fast powerloop
line used a current/next-gate velocity blend, while more aggressive global
velocity/heading reward caused speed-obsessed crashes. The current real bag
analysis shows `fullswitch` mostly loses time after the loop (`4->5`, `5->6`,
`6->0`), not inside the powerloop itself. Therefore v2 applies the velocity
term only after Gate 3 and after the first gate has already been passed; it
does not touch Gate 3 progress masking, target switching, TWR, time pressure,
or control regularization.

Run:

```bash
./scripts/run/train_powerloop_fullswitch_real_twr_speed_v2.sh 3000 8192
./scripts/run/eval_powerloop_real_twr.sh <speed_v2_run_dir> best_model.pt 768
```

User-reported batch result:

| Metric | Base TWR-1.87 | Speed v1 | Speed v2 |
|---|---:|---:|---:|
| Envs | 768 | 1000 | 1000 |
| Overall 3-lap SR | **98.0%** | 86.7% | 96.2% |
| Mean 3-lap time | **18.34 s** | 20.39 s | 18.68 s |
| Std 3-lap time | **0.48 s** | 2.12 s | 1.21 s |
| Best 3-lap time | 17.50 s | 17.56 s | **17.44 s** |
| Worst 3-lap time | 24.24 s | 29.88 s | 27.88 s |

Interpretation: v2 is much healthier than v1, but it is still not better than
the TWR-1.87 fullswitch base. The slightly faster best-case (`17.44 s`) shows
the post-loop velocity idea can create a faster tail, but the lower SR, slower
mean, and larger std mean the reward is still pulling some rollouts into less
reliable trajectories. Do not use v2 as the real-flight base yet.

Next reward direction: keep the TWR-1.87 fullswitch base and make the speed
reward narrower/weaker. Because `idx_wp` is the current target gate, the slow
real segments `4->5 / 5->6 / 6->0` correspond to target indices `[5, 6, 0]`.
A conservative v3 should start with `vel_toward_gate_reward_scale=0.5` on
`[5, 6]` only, then reintroduce target `0` only if Gate-0 clearance remains
safe.

### Overnight speed ablations

Added `scripts/run/train_powerloop_fullswitch_real_twr_overnight_speed.sh` to
train three sequential candidates and automatically run matched real-TWR batch
eval after each model.

All three keep the same deployment-relevant dynamics as the current base:
`fullswitch + thrust_to_weight=1.87 + narrow TWR DR + staged replay reset`.

| Candidate | Reward change | Hypothesis |
|---|---|---|
| `speedv3-vel05-g56` | `vel_toward_gate=0.5`, target indices `[5,6]`, next weight `3/8` | v2 was too broad/strong; only speed up real-slow `4->5` and `5->6`. |
| `speedv4-currentvel-g560` | `vel_toward_gate=0.5`, target indices `[5,6,0]`, next weight `0` | keep post-loop speed pressure, but remove next-gate corner-cutting that may increase variance. |
| `speedv5-time06` | no velocity reward, `lap_incomplete_penalty=-0.06` | isolate mild time pressure while keeping control regularization unchanged. |

Batch eval results under the matched real-effective TWR eval:

| Candidate | Envs | 3-lap SR | Mean | Std | Best | Worst | 3-lap successes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `twr1p87` base | 768 | **98.0%** | **18.34 s** | 0.48 s | 17.50 s | 24.24 s | 753 |
| `speedv1` broad time/control pressure | 1000 | 86.7% | 20.39 s | 2.12 s | 17.56 s | 29.88 s | 867 |
| `speedv2-postloopvel` | 1000 | 96.2% | 18.68 s | 1.21 s | **17.44 s** | 27.88 s | 962 |
| `speedv3-vel05-g56` | 1000 | **98.8%** | 19.05 s | **0.42 s** | 18.16 s | 23.26 s | 988 |
| `speedv4-currentvel-g560` | 1000 | **98.0%** | 18.80 s | **0.43 s** | 18.04 s | 22.92 s | 980 |
| `speedv5-time06` | 1000 | 94.2% | 21.73 s | 2.55 s | 17.90 s | 27.84 s | 942 |

Interpretation: none of the speed candidates beat the `twr1p87` base on the
primary racing metric, mean 3-lap time at high SR. `speedv3` and `speedv4` are
stable, but they are slower. `speedv2` found the fastest single rollout, but
the lower SR and larger std make it a poor real-flight candidate. The current
deployment/racing base should remain
`2026-04-22_00-53-52_powerloop-r1d1-gate3mask-fullswitch-twr1p87`.

Decision rule: keep `TWR-1.87 fullswitch base` unless a candidate improves mean
3-lap time while keeping SR near the base (`>=97%`) and std close to base
(`~0.5 s`). A faster best-case alone is not enough for real deployment.

### Reset ablation — linear fallback velocity

Added configurable non-spline linear-reset velocity:

| Parameter | Default | Reset-vel candidate |
|---|---:|---:|
| `linear_reset_vel_min` | 0.0 | 0.3 |
| `linear_reset_vel_max` | 0.0 | 1.0 |

The current baseline's linear fallback reset spawns the drone mid-air between
two gates with zero velocity. That is convenient for coverage, but unlike real
race flight. The new option gives only the non-spline linear fallback a small
world-frame velocity toward the target gate. Replay resets and ground resets
are unchanged, and spline reset remains off.

Added `scripts/run/train_powerloop_fullswitch_real_twr_resetvel.sh`:

```bash
./scripts/run/train_powerloop_fullswitch_real_twr_resetvel.sh 3000 8192 77
./scripts/run/eval_powerloop_real_twr.sh \
  <resetvel_run_dir> best_model.pt 1000
```

This intentionally does not add segment-focused replay weighting yet. That
would change which gates are sampled more often and would confound the result.
Test the cleaner reset-velocity ablation first; if it helps, segment-focused
replay can be tested as a separate candidate.

## [Real Powerloop Bag Evaluation: Official 3-Lap Timing] - 2026-04-21

### Purpose

After the first real-world powerloop test, update the bag analysis to match the
evaluation timing used for race comparison:

**first 3 laps = first race-start command → crossing the last gate of lap 3**

For the 7-gate powerloop track, the endpoint is the ordered Gate 6 crossing of
lap 3, not the next Gate 0 crossing, and not takeoff time.

### Tooling changes

- `scripts/run/test_bag.sh`
  - Defaults to real powerloop bag coordinates.
  - Runs `lap_time.py` with `--track powerloop`.
  - Keeps full plot analysis available, but skips desired trajectory and
    animation by default.
- `scripts/run/batch_test_bags.sh`
  - Runs every ROS2 bag under a given folder.
  - Does not generate plots.
  - Writes `summary.csv`.
  - Prints a compact comparison table sorted by official eval time.
- `bin/lap_time.py` in the sim2real repo
  - Adds machine-readable `--summary-json`.
  - Adds official `eval_first_n_laps_time`.
  - Uses real powerloop mocap coordinates (`sim_y + 4.5 m`).
  - Handles the powerloop shared physical gate where Gate 3 and Gate 6 can
    trigger at nearly the same timestamp.

### Command

```bash
cd /home/peterni/Documents/ese6510/ese651_project
./scripts/run/batch_test_bags.sh rosbags_powerloop_baseline_controller_04_20 crazy_jirl_b3 3
```

### Results

Source:

```text
rosbags_powerloop_baseline_controller_04_20/summary.csv
```

| Bag | Official 3-lap eval (s) | Complete ordered laps | Ordered passes | Valid passes | Best lap (s) | Mean lap (s) | Lap std (s) | Min clearance (m) | Mean clearance (m) | Max speed (m/s) | Max body-rate cmd (deg/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `powerloop-r1d1-gate3mask` | **22.435** | 4 | 33 | 42 | **7.355** | **7.451** | **0.074** | -0.048 | 0.188 | 5.302 | 244.784 |
| `powerloop-r1d1-gate3mask-fullswitch` | 22.943 | 5 | 38 | 48 | 7.495 | 7.729 | 0.184 | -0.017 | **0.201** | **5.572** | 244.862 |
| `powerloop-fixed-baseline` | n/a | 0 | 4 | 9 | n/a | n/a | n/a | -0.009 | 0.211 | n/a | n/a |
| `sysid` | n/a | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Ordered-pass clearance by gate:

| Bag | Gate | Ordered clearance (m) |
|---|---:|---|
| `powerloop-r1d1-gate3mask` | 0 | `0.016 -0.025 -0.011 -0.048 -0.029` |
|  | 1 | `0.297 0.337 0.313 0.313 0.286` |
|  | 2 | `0.283 0.271 0.310 0.282 0.295` |
|  | 3 | `0.313 0.210 0.265 0.332 0.307` |
|  | 4 | `0.086 0.119 0.113 0.078 0.112` |
|  | 5 | `0.025 -0.015 0.015 0.025` |
|  | 6 | `0.393 0.242 0.322 0.378` |
| `powerloop-r1d1-gate3mask-fullswitch` | 0 | `0.009 -0.017 0.021 -0.010 0.001 0.043` |
|  | 1 | `0.329 0.304 0.235 0.278 0.235 0.229` |
|  | 2 | `0.338 0.339 0.334 0.341 0.314 -0.003` |
|  | 3 | `0.300 0.330 0.338 0.272 0.316` |
|  | 4 | `0.243 0.244 0.223 0.181 0.233` |
|  | 5 | `0.023 0.011 0.019 0.030 0.045` |
|  | 6 | `0.332 0.376 0.272 0.287 0.254` |

Mean segment times:

| Bag | start->0 | 0->1 | 1->2 | 2->3 | 3->4 | 4->5 | 5->6 | 6->0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `powerloop-r1d1-gate3mask` | 0.994 | 1.122 | **0.958** | **1.502** | **1.054** | **1.039** | **0.933** | **0.880** |
| `powerloop-r1d1-gate3mask-fullswitch` | **0.937** | **1.117** | 1.028 | 1.510 | 1.058 | 1.157 | 0.961 | 0.961 |
| `powerloop-fixed-baseline` | 0.925 | 1.045 | 1.027 | 6.771 | n/a | n/a | n/a | n/a |

Definitions:

- `valid pass`: the drone crosses a gate's plane and, at the interpolated
  crossing point, is inside the physical gate window. The current detector uses
  a `1.0 m x 1.0 m` window with `0.05 m` slack, so the threshold is
  `|lateral| < 0.55 m` and `|vertical| < 0.55 m`.
- `clearance`: physical gate-edge margin at an ordered pass,
  `0.5 - max(|lateral|, |vertical|)`. Positive means the drone center is inside
  the nominal `1.0 m` opening; slightly negative means it was counted valid
  only because of the `0.05 m` detector slack.
- `ordered pass`: a `valid pass` that matches the expected gate sequence
  `0 -> 1 -> ... -> 6`. Official timing is computed from these ordered passes.
- `near-miss` is no longer used as a reporting metric for this result. In this
  detector it only means "crossed a gate's infinite plane outside the gate
  window", which is noisy on powerloop and not useful as a success/failure
  metric.

### Interpretation

The fixed baseline is not a viable real powerloop policy in this test. It never
formed a complete ordered lap and has only sparse valid gate passes.

Both R1D1 gate3mask policies successfully executed real ordered powerloop laps.
This is the important result: the powerloop-specific training logic transferred
to real flight well enough to repeatedly complete the 7-gate sequence.

`powerloop-r1d1-gate3mask` is faster on the official 3-lap metric:

- `22.435 s` official eval
- `0.508 s` faster than `fullswitch`
- lower lap mean (`7.451 s` vs `7.729 s`)
- lower lap std (`0.074 s` vs `0.184 s`)
- faster through most post-powerloop segments, especially `4->5`, `5->6`, and
  `6->0`

`powerloop-r1d1-gate3mask-fullswitch` has cleaner geometry in several places:

- higher mean ordered-pass clearance (`0.201 m` vs `0.188 m`)
- better Gate 3 clearance (`min 0.272 m` vs `0.210 m`)
- much better Gate 4 clearance (`min 0.181 m` vs `0.078 m`)

This suggests `fullswitch` is not slower because it fails the powerloop. The
actual powerloop segments are nearly identical:

- `2->3`: `1.510 s` vs `1.502 s`
- `3->4`: `1.058 s` vs `1.054 s`

The speed loss is mostly after the loop:

- `4->5`: `1.157 s` vs `1.039 s`
- `5->6`: `0.961 s` vs `0.933 s`
- `6->0`: `0.961 s` vs `0.880 s`

So the best current interpretation is: `fullswitch` is slightly more
conservative and has better gate-edge clearance, especially at Gates 3 and 4,
but gives up time in the post-loop straight/chicane and return-to-Gate-0
segments.

The one `fullswitch` Gate 2 clearance value of `-0.003 m` happened during the
final stopping/outlier portion of the recording and should not drive the policy
choice. Gate 0 and Gate 5 remain tight for both policies.

The extra recorded passes/laps in the `fullswitch` bag should not be interpreted
as better durability because the bags were not necessarily recorded for the
same duration. The fair comparison metric here is the official first-3-lap eval
time.

### Conclusion

For official first-3-lap timing, `powerloop-r1d1-gate3mask` was the faster real
policy in this batch. For the next training base, `fullswitch` is still a
reasonable choice because its target-switch semantics are cleaner and its Gate
3 / Gate 4 clearances look better; the likely speed loss is in the post-loop
straight/chicane segments rather than in the powerloop itself. The baseline
should not be used as the main powerloop candidate.

Next experiment direction: keep the `fullswitch` base and avoid making the
Gate 3 powerloop more aggressive for now. Instead, target speed/racing-line
improvements on `4->5`, `5->6`, and `6->0`, where the real bag shows most of
the time loss.

---

## [Powerloop Split Gate Semantics: Full Switch, Virtual Reward, Gate-3-Only Approach Gate] - 2026-04-20

### Purpose

The previous setup mixed two different goals into one gate detector:

- runtime target switching / lap counting
- conservative center-through shaping

This caused two problems:

- `circle` visual evals could under-count visibly valid passes
- shrinking the gate for conservative shaping also shrank the target-switch
  semantics, which is not what we want for the next powerloop real-transfer
  experiment

### New semantics

Training-side gate logic is now split into three parts:

1. `target switch` / `lap counting`: use the **full physical gate**
   (`1.0 m`)
2. `gate_pass` reward / replay shaping: use the **virtual inner gate**
   (`gate_side = 0.7`)
3. `approach-zone` Fix A: keep it enabled **only for `powerloop` Gate 3**

This gives the policy a clear incentive to fly conservatively through the
center while avoiding the bad side effect of "physically passed, but target did
not switch".

### Implementation

In `CircleQuadcopterStrategy.get_rewards()`:

- `within_switch_bounds` uses a fixed physical half-side of `0.5`
- `within_reward_bounds` uses `gate_side / 2`
- `gate_switched` drives:
  - `idx_wp`
  - `n_gates_passed`
  - `desired_pos_w`
  - lap completion logic
- `gate_rewarded` drives:
  - `gate_pass` reward
  - replay buffer insertion
- `approach_valid` remains active only when:
  - `track_name == "powerloop"`
  - `idx_wp == 3`

### Why this is the next version

This is the cleanest way to separate:

- **runtime correctness**: the system should recognize a physically valid pass
- **training conservatism**: the policy should still prefer a safer inner line

It also preserves the existing Gate-3 anti-exploit logic without reapplying it
to `circle`.

### Training command

Use the canonical powerloop script:

```bash
cd /home/peterni/Documents/ese6510/ese651_project
./scripts/run/train_powerloop.sh 3000 8192
```

This now trains the split-semantics version and writes a new run name:

- `powerloop-r1d1-gate3mask-fullswitch`

---

## [Powerloop Robustness Sweep: Observation Bias vs Control Mismatch] - 2026-04-19

### Purpose

Before the next real-world sysid day, run a minimal but clean robustness sweep
for the current powerloop checkpoint to answer one question:

Is this policy more likely to fail from moderate control mismatch, or from
systematic observation mismatch?

This was intentionally a deterministic eval-only sweep, not training DR:

- Track: `powerloop`
- Checkpoint: `2026-04-18_14-55-19_powerloop-r1d1-gate3mask`
- Reset mode: ground-start only (`ground_reset_ratio=1.0`)
- Per scenario: 768 ground starts
- Metrics: takeoff success, first-gate success, 3-lap success, successful
  3-lap time

### Figure

![Powerloop robustness sweep summary](figures/powerloop_robustness_sweep_2026-04-18.png)

### Key result

All 13 scenarios reached:

- Takeoff success = **100%**
- First-gate success = **100%**

So the separating signal in this sweep is **not** takeoff failure or immediate
Gate-1 failure. The main differences appear in the later trajectory execution.

The strongest pattern is:

- Moderate control mismatch did **not** collapse the policy.
- Zero-mean observation noise also did **not** collapse the policy.
- **Systematic observation bias** was the dominant failure mode.

Most severe cases:

- `obs_vel_bias_0p20`: 3-lap success dropped from **96.6%** to **2.9%**
- `obs_gate_bias_5cm`: 3-lap success dropped to **92.4%**, mean successful
  time slowed from **18.65 s** to **20.76 s**

### Full sweep table

| Scenario | Group | 3-lap SR (%) | Mean 3-lap (s) | Delta SR vs nominal (pct-pts) |
|---|---:|---:|---:|---:|
| `control_nominal` | control | 96.6 | 18.65 | +0.0 |
| `control_thrust_0p90` | control | 100.0 | 18.33 | +3.4 |
| `control_thrust_1p10` | control | 96.2 | 18.64 | -0.4 |
| `control_latency_20ms` | control | 96.9 | 18.40 | +0.3 |
| `control_latency_40ms` | control | 99.9 | 18.41 | +3.3 |
| `control_rate_gain_0p85` | control | 95.6 | 18.65 | -1.0 |
| `control_rate_gain_1p15` | control | 99.5 | 18.38 | +2.9 |
| `obs_vel_noise_0p05` | observation | 96.5 | 18.81 | -0.1 |
| `obs_vel_bias_0p20` | observation | 2.9 | 28.15 | -93.8 |
| `obs_yaw_bias_5deg` | observation | 99.9 | 18.30 | +3.3 |
| `obs_gate_bias_5cm` | observation | 92.4 | 20.76 | -4.2 |
| `obs_delay_20ms` | observation | 98.2 | 18.40 | +1.6 |
| `obs_delay_40ms` | observation | 100.0 | 18.39 | +3.4 |

Raw outputs are stored under:

- `logs/rsl_rl/quadcopter_direct/2026-04-18_14-55-19_powerloop-r1d1-gate3mask/robustness_sweep/best_model/robustness_sweep_summary.csv`
- `logs/rsl_rl/quadcopter_direct/2026-04-18_14-55-19_powerloop-r1d1-gate3mask/robustness_sweep/best_model/robustness_sweep_summary.json`

### Interpretation

This result makes sense and is worth recording, with one important boundary:
it is a **sim-side robustness diagnosis**, not a proof of real deployment
readiness.

Why it makes sense:

- The project has already shown that the 40-dim observation pipeline can fly
  in real on circle tasks, so the observation design itself is not obviously
  broken.
- The project has also already hit a real deployment bug where wrong
  `gate_side` handling changed the gate-corner observation geometry. That is
  qualitatively consistent with the measured sensitivity to gate bias here.
- It is plausible that a policy tolerates zero-mean noise better than a
  fixed, systematic bias. The sweep matches that expectation.

What is safe to conclude from this sweep:

- For this checkpoint, **moderate actuator mismatch is not the first-order
  failure mode** among the tested perturbations.
- For this checkpoint, **systematic observation bias is the highest-priority
  risk**, especially body-frame x-velocity bias.
- Gate-geometry / gate-corner bias is the second clearest vulnerability in
  this sweep.

What is **not** safe to conclude:

- That the real system definitely has a `+0.20 m/s` body-frame velocity bias
- That the policy is ready for unrestricted real deployment
- That the exact sign of every small improvement in the control sweeps should
  be over-interpreted; some scenarios were slightly better than nominal, which
  is informative but not enough to claim a monotonic trend

### Next real-world sysid priority

Based on this sweep, the next real test day should prioritize:

1. Body-frame velocity sanity and bias checks
2. Gate geometry / gate-corner alignment checks
3. Low-level response softness checks (effective thrust and body-rate tracking)

Recommended log targets:

- Hover: body-frame velocity near zero when physically stationary
- Straight-line accel/decel: body-frame velocity sign, scale, and lag
- Roll/pitch/yaw step response: latency, rise time, overshoot, rate tracking
- Gate observation pipeline: transformed gate corners or equivalent geometry
  fed to the policy

Use this changelog entry as a prioritization result for real sysid, not as a
deployment-readiness claim.

---

## [Previous-Gate Backtrack Detection] - 2026-04-18c

### Purpose

After Fix A (approach-zone gating) cleaned up the Gate-3 micro-U-turn, a new
exploit surfaced: on the powerloop track the drone would pass Gate 2 cleanly,
then reverse back through Gate 2 (re-cross its plane into the +x approach
zone), and proceed to Gate 3. Gates 2 and 3 sit on the same gate structure
with the same yaw, so re-entering Gate 2 backwards is a cheap way to
"restart" the approach toward Gate 3 without taking a proper detour.

The existing `wrong_side_crossed` check only monitors the **current target**
gate. Once Gate 2 was passed and the target advanced to Gate 3, nothing in
the reward function cared about Gate 2's plane any longer.

### Fix

Added a generic previous-gate backtrack detector. Every step, the drone's x
is computed in the frame of `prev_gate_idx = (idx_wp - 1) % num_gates`. If
the drone crosses that plane from `-x` to `+x` while inside the gate's y/z
bounds, the episode is killed (`_crashed = 200`, same treatment as
`wrong_side_crossed`).

Guards:

- Skipped while `n_gates_passed == 0` — at spawn, prev_gate is spurious.
- Skipped on the step a gate is passed (prev_gate just changed; comparing
  last step's x in the old prev_gate frame against this step's x in the new
  prev_gate frame is meaningless).
- Inside y/z gate bounds only, so flying around the outside of a passed gate
  never triggers.

### Code changes

- `CircleQuadcopterStrategy.__init__`: added `_prev_x_drone_wrt_prev_gate`
  buffer.
- `CircleQuadcopterStrategy.get_rewards`: computes drone pose wrt prev gate
  via `subtract_frame_transforms`, flags backtrack crossings, updates buffer
  each step with a seed override on the just-passed-gate transition.
- `CircleQuadcopterStrategy.reset_idx`: zero-initialized; the
  `n_gates_passed > 0` guard prevents false positives until the first pass.

Transfer risk: zero. Training-only termination; `play_race.py` / `eval_race.py`
do not run through `get_rewards`.

---

## [Approach-Zone Gate-Pass Gating] - 2026-04-18b

### Purpose

Even after the Gate-3 progress mask, the policy was still exhibiting a
"quick-pass-and-come-back-out" at Gate 3 — a micro-U-turn at the gate plane
that briefly puts `prev_x > 0`, then immediately crosses back to `curr_x <= 0`
to fire gate_pass for the +200 reward. This is not a real powerloop nor a
real side detour; it exploits the fact that the existing detector only
requires a single 20 ms step of being on the +x approach side.

The project description explicitly defines Gate 3's semantics as
"entering the right opening of the same structure from the same side
(forward/away from camera) as it did for Gate 2" — the policy's micro-
U-turn does not satisfy this.

### Fix

Added an approach-zone gate to `gate_passed`: the drone must have reached
`x_wrt_gate >= approach_x_threshold` (default 0.3 m, gate_half_side=0.35)
at some point since the current target gate was activated. A per-env
`_max_x_since_gate_change` buffer tracks this and is reset on:

- Episode reset (initialized from actual post-reset `_pose_drone_wrt_gate[:, 0]`
  so replay resets that land on either side of a gate are handled correctly)
- Gate-pass transition (initialized from drone's x in the new gate's frame)

After the gate, the next step's running `max()` takes over.

All legal passes (direct approach, side detour, classic powerloop) reach
`max_x` values well above 0.3 m by geometry, so the threshold has ample
margin. The micro-U-turn at the gate plane cannot reach it.

### Code changes

- `CircleQuadcopterStrategy.__init__`: added `_max_x_since_gate_change`
  buffer and `_approach_x_threshold` config field.
- `CircleQuadcopterStrategy.get_rewards`: update buffer each step, add
  `approach_valid` condition to `gate_passed`, reset buffer on gate pass
  using the new-frame x from `subtract_frame_transforms`.
- `CircleQuadcopterStrategy.reset_idx`: initialize buffer from the
  post-reset `_pose_drone_wrt_gate[:, 0]`.

Transfer risk: zero. Detector is training-only; `play_race.py` /
`eval_race.py` don't run through `get_rewards`.

---

## [Powerloop R1+D1 + Gate-3 Progress Mask] - 2026-04-18

### Purpose

R1+D1 (dense gate-progress reward + staged replay/DR) produced unclean Gate-3
behavior on the powerloop track: small poke from the exit side of the gate
plane, then a bounce back to cross correctly, or loitering near the plane.
Side-detour behavior from the 2026-04-16 fixed baseline was lost.

### Root cause

`progress_goal` uses 3D Euclidean distance to the gate center
(`quadcopter_strategies.py:1100-1105`). For the powerloop loop gate (idx_wp=3),
the correct trajectory — whether over-the-top or side-detour — **must first
increase distance** before re-entering from the correct side. The dense
progress reward instead pulls the policy toward distance-minimizing shortcuts,
which only remain legal by poking close to the plane from the wrong side (the
wrong-side crash only fires inside the tight y/z gate bounds, so grazing the
plane just outside the opening costs nothing).

### Fix

Added optional `progress_skip_gate_indices` reward config. When set, dense
progress is zeroed on those gate indices; `gate_pass`, `lap_incomplete`,
`cmd_reg`, `cmd_smoothness`, `crash`, and `death_cost` all continue to apply.
For powerloop training this is set to `[3]`.

Effect per gate:
- Non-loop gates (0,1,2,4,5,6): R1 progress intact — retains the speed benefit
- Gate 3: signal reduces to the 2026-04-16 baseline (pure sparse) — side-detour
  is the learned optimum, as already demonstrated by that baseline

Transfer impact: zero. Obs/action/network unchanged; the masked training signal
on Gate 3 is identical to the already-deployed baseline for that gate.

### Code changes

- `CircleQuadcopterStrategy.get_rewards()`: read
  `progress_skip_gate_indices` from the reward dict and mask `r_progress`
  per-env by `_idx_wp`. Default behavior (empty list) unchanged.
- New canonical training script `scripts/run/train_powerloop.sh`
  (replaces ad-hoc `train_powerloop_r1d1.sh`) with
  `progress_skip_gate_indices=[3]` baked into `REWARD_OVERRIDES`.

### Training command

```bash
cd /home/peterni/Documents/ese6510/ese651_project
./scripts/run/train_powerloop.sh 5000 8192
```

---

## [Powerloop R1 / D1 / R1+D1 Sequence] - 2026-04-17

### Purpose

Prepare the next three **single-agent powerloop** candidates for real testing without disturbing the fixed post-fix baseline.

- `R1`: reward-only fast candidate
- `D1`: reset/DR-only robust candidate
- `R1+D1`: combined candidate

This sequence is intentionally aligned with the strongest **real-world single-agent racing** ideas from:

- Song et al. 2021 / 2023: dense gate-progress reward plus successful-state replay resets
- Kaufmann et al. 2023 (Swift): controller-aligned observation, previous-action input, smooth-action regularization, real-driven sim2real refinement
- Ferede et al. 2025: moderate domain randomization improves transfer, but excessive DR costs speed

### Code changes

- Added optional `progress_goal_reward_scale` and `lap_complete_reward_scale` to the current `CircleQuadcopterStrategy`.
- Added configurable replay-reset and dynamics-DR knobs to the environment config so experiments can change reset/DR without forking the strategy.
- Added optional **staged replay reset**:
  replay can start at zero early in training and ramp to the configured ratio later, similar in spirit to Song-style reset buffers.
- Rewrote `train_overnight.sh` to run a sequential **powerloop** experiment triplet instead of the old circle smoothness ablations.

### Experiment definitions

#### `R1` — reward-only

Goal: increase speed with the least invasive change to the current baseline reward.

- Keep the current sparse/event-based baseline intact
- Add a **dense gate-progress term**:
  `progress_goal_reward_scale = 20.0`
- Add a small **lap completion bonus**:
  `lap_complete_reward_scale = 0.5` which maps to `+50` on lap completion
- Keep:
  `gate_pass_reward_scale = 200.0`
  `lap_incomplete_penalty_scale = -0.05`
  `cmd_reg_rp/yaw`
  `cmd_smoothness = -0.1`
  `use_spline_reset = False`

Rationale:
- Matches the successful single-agent real-racing pattern from Song 2023 more closely than the pure sparse baseline
- Still keeps gate-pass events dominant, avoiding a full return to raceline-heavy dense shaping

#### `D1` — reset + DR only

Goal: improve real-world robustness without changing the reward.

- Reward remains the fixed sparse baseline
- Track switches to `powerloop`
- Enable **staged replay reset**:
  - `replay_reset_ratio = 0.25`
  - `ground_reset_ratio = 0.10`
  - `staged_replay_reset = True`
  - `replay_warmup_iterations = 500`
  - `replay_full_iterations = 2000`
- Use **moderate DR** instead of the current wider baseline DR:
  - `twr_randomization_pct = 0.10`
  - `aero_randomization_scale_min/max = 0.5 / 2.0`
  - `pid_kpki_randomization_pct = 0.15`
  - `pid_kd_randomization_pct = 0.30`
- Keep:
  - `action_latency_max = 2`
  - `obs_latency_prob = 0.0`
  - `mass_variation = 0.0`
  - `motor_tau_scale_min/max = 1.0 / 1.0`
  - `use_spline_reset = False`

Rationale:
- Replay warmup is closer to Song 2021 / 2023 than immediately replaying high-probability cached states from iteration 0
- Moderate DR follows the real-world trend from Ferede 2025 that robustness improves before large DR begins to slow the controller down

#### `R1+D1` — combined

Combines the `R1` reward changes with the `D1` reset/DR changes for the final candidate in the overnight sequence.

### Training command

Run the sequence with:

```bash
cd /home/peterni/Documents/ese6510/ese651_project
./scripts/run/train_overnight.sh 5000 8192
```

This now launches, in order:

1. `powerloop-r1-gateprogress`
2. `powerloop-d1-stagedreset-dr`
3. `powerloop-r1d1-gateprogress-stagedreset`

## [Fixed Sim2Real Baseline] - 2026-04-16

### Purpose

Freeze a single **post-fix baseline** for all future experiments. This baseline is not claimed to be the fastest historical circle policy; it is the most defensible starting point after fixing the controller-analysis mismatches that were inflating sim-real uncertainty.

### Controller / analysis fixes included

- Real controller gate switching now uses the same **plane-crossing + gate-bounds** logic as training/evaluation. The old center-distance switch was removed.
- Racing-mode thrust is now converted to Crazyflie PWM using the same **calibrated nonlinear thrust map** as the non-racing controller path. The old linear PWM map was likely contributing to low-thrust / low-flight behavior in real flights.
- `sysid_from_bag.py` now uses the current nominal mass/TWR, converts `/ctbr_cmd` body-rate commands from **deg/s to rad/s**, and bounds PID lag estimation to a physically meaningful range.
- `obs_latency_prob` is now actually applied during training instead of being a dead config field.

### Baseline configuration

This is now the default training baseline in code:

| Parameter | Baseline |
|---|---:|
| `gate_side` | **0.7** |
| `gate_pass_reward_scale` | **200.0** |
| `lap_incomplete_penalty_scale` | **-0.05** |
| `cmd_reg_rp_scale` | **-1.0** |
| `cmd_reg_yaw_scale` | **-0.5** |
| `cmd_smoothness_scale` | **-0.1** |
| `action_latency_max` | **2** |
| `fixed_action_delay_steps` | **-1** |
| `obs_latency_prob` | **0.0** |
| `mass_variation` | **0.0** |
| `motor_tau_scale_min/max` | **1.0 / 1.0** |
| `use_spline_reset` | **False** |

### Why this is the baseline

1. Keeps the sparse V3 reward core that transferred best.
2. Keeps V6's `gate_side=0.7` safety margin.
3. Keeps V6-Smooth's light `cmd_smoothness=-0.1`, which gave the best real consistency without the V6-Smooth2 slowdown.
4. Removes spline reset and extra mass/tau DR, which consistently failed to show real-world benefit.
5. Uses conservative action-latency randomization: `fixed_action_delay_steps=-1`
   means each training/eval environment samples `0/1/2` policy-step delay from
   `action_latency_max=2` rather than using a deterministic 2-step delay.
   Observation latency is left off for the baseline because it only became
   active after the bug fix and needs a clean ablation later.

### Status

This is the **recommended launch baseline** for the next round of testing. It is appropriate as a baseline because it minimizes known mismatches and removes the least trusted additions, but it still needs fresh real validation after the thrust-map and gate-switch fixes.

### Next experiment

The next planned experiment is the **powerloop track**, using this same fixed baseline as the starting point. Only the track/task-specific logic should change first; reward, reset, and DR should stay frozen until the first post-fix powerloop result is in.

Todo after the first clean post-fix powerloop runs:
- Add a **minimal powerloop-guide ablation (`G1`)** on top of the clean baseline or `D1`, not on top of the old dense powerloop stack.
- `G1` should restore only the geometric guide pieces that were likely genuinely useful:
  - Gate 2 `apex` redirect
  - Gate 3 `2-phase` guide: `apex -> offset_center`
- `G1` should explicitly *not* restore the older high-risk pieces:
  - `pre_entry`
  - `vel_toward_gate`
  - heading reward
  - old monolithic `DefaultQuadcopterStrategy` powerloop stack
- Preferred first comparison: `baseline`, `D1`, and then `G1` or `G1+D1` as the next task-specific candidate.

## [Real-World Evaluation Summary — All Versions] - 2026-04-16 (v3 strict-order lap timer)

All rosbags re-analyzed with the **v3 strict-order lap timer** (enforces gate 0→1→2→3→0 sequence; laps with skipped or misordered gates are rejected as sequence breaks).

**All policies completed laps with zero crashes** — the sim2real pipeline is production-ready.

### Cross-cutting comparison (sorted by Takeoff → 3 laps)

| Version (bag) | gate_side | Laps | Brk | Best | Mean | Median | Std | Takeoff→3 | 3-lap best | Path/lap | v_max | tilt_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **V3** (`group30_cyclev3`) | 1.0 | **16** | 0 | 2.409 | **2.435** | 2.431 | 0.020 | **7.927** | **7.273** | 7.08 | 3.97 | 49.1 |
| **V6** (`zhuohao_wk2_cycle_v6`) | 0.7 | 10 | 0 | **2.369** | 2.497 | 2.503 | 0.050 | 7.866 | 7.375 | 7.65 | **4.20** | 52.5 |
| bruce V6-v5spline+dr | 0.7 | 5 | **2** | **2.317** | 2.462 | 2.508 | 0.114 | 7.873 | **7.243** | 9.91 | 4.36 | 56.5 |
| V2 (`group30_cyclev2`) | 1.0 | 15 | 0 | 2.525 | 2.697 | 2.596 | 0.297 | 8.151 | 7.637 | 7.58 | 3.73 | 50.3 |
| bruce V6-v2base+dr | 0.7 | 6 | 0 | 2.502 | 2.607 | 2.624 | 0.049 | 8.258 | 7.766 | 7.23 | 4.18 | 54.5 |
| V5 (`zhuohao_wk2_cycle_v5`) | 1.0 | 12 | 0 | 2.531 | 2.610 | 2.616 | 0.030 | 8.332 | 7.746 | 7.07 | 3.54 | 43.1 |
| **V6-Smooth** (`zhuohao_wk2_cycle_v6_smooth`) | 0.7 | 10 | 0 | 2.505 | 2.517 | 2.518 | **0.007** | 8.341 | 7.539 | 7.65 | 3.81 | 49.2 |
| V4-good (`zhuohao_wk2_cycle_v4_good`) | 0.7 | 9 | 0 | 2.532 | 2.620 | 2.631 | 0.035 | 8.455 | 7.791 | 7.08 | 3.36 | 48.8 |
| V6-Smooth2 (`zhuohao_wk2_cycle_v6_smooth2`) | 0.7 | 9 | 0 | 2.662 | 2.728 | 2.739 | 0.024 | 8.837 | 8.133 | 7.68 | 3.38 | 44.2 |
| V7 (`zhuohao_wk2_cycle_v7`) | 0.7 | 11 | 0 | 2.878 | 3.135 | 3.132 | 0.110 | 10.070 | 9.320 | 8.25 | 3.30 | **39.1** |
| V4-Cons (`group30_cyclev4`) | 1.0 | 3 | **4** | 2.602 | 2.856 | 2.949 | 0.182 | **19.068** | 8.568 | 17.04 | 4.05 | 51.2 |

### Observations

1. **V3 is the overall best for race day (Takeoff→3).** 7.927 s, 16 laps, zero breaks, std 0.020 s. Tighter DR + obs noise + action latency produced the most reliable and fastest policy.
2. **V6 is a close second** (Takeoff→3 = 7.866 s, only 0.06 s behind V3) with the fastest single lap (2.369 s). gate_side=0.7 adds safety margin at almost no speed cost.
3. **bruce V6-v5spline+dr has raw speed** (best lap 2.317 s, fastest 3-lap 7.243 s) but 2 sequence breaks and high tilt (56.5°) make it unreliable.
4. **V2 had a hiccup** (laps 9–10), inflating std to 0.297 s. V3 never hiccupped across 16 laps — obs noise + action latency were the key additions.
5. **V6-Smooth is the consistency champion** (std 0.007 s) — useful if scoring penalizes variance, but Takeoff→3 is 8.341 s (0.4 s behind V3).
6. **V4-Conservative is broken.** Strict-order analysis reveals only 3 valid laps with 4 sequence breaks. Takeoff→3 = 19.068 s. Reduced `lap_incomplete_penalty` caused gate-skipping.
7. **V7-Rebalance is smoothest but slowest.** Lowest tilt (39.1°) — useful only as safety fallback.
8. **Spline reset + extra DR (mass, motor tau) consistently hurt.** V4/V5 are slower than V3 which has neither.
9. **Sim→real transfer well-calibrated.** Relative rankings held across versions.

### Recommended deployment order for race day

1. **V3** — fastest Takeoff→3 (7.927 s), most proven (16 laps, 0 breaks), excellent consistency.
2. **V6** — 0.06 s behind V3 but with 0.15 m gate safety margin; use if gate calibration uncertain.
3. **V6-Smooth** — ultra-consistent (std 0.007 s); use if scoring rewards consistency.
4. **V7** — safety fallback for rough venue conditions.

---

## [Circle-V7-Rebalance] - 2026-04-15

### Ablation — Pasumarti-aligned cmd_reg ratio

Same as V6 (gate_side=0.7, no smoothness penalty), but re-weights the command regularization to match Pasumarti 2026 Eq. 5: `r_cmd = 2.0·(ω_roll² + ω_pitch²) + 0.05·ω_yaw²`.

**Why:** V2/V3/V4 real-world rosbags show mean body rates of 98–160 rad/s, with peaks driven by roll/pitch during gate passage. Yaw rate stays low throughout. The current split (`rp=-1.0, yaw=-0.5`) under-penalizes roll/pitch (the actual source of jerkiness) and over-penalizes yaw (a non-issue), which likely crowds out exploration in the axes that matter.

**Run name:** `circle-v7-rebalance`
**Log folder:** `logs/rsl_rl/quadcopter_direct/<timestamp>_circle-v7-rebalance/`

### Changed (relative to V6)

| Parameter | V6 | V7-Rebalance |
|-----------|----|----|
| `cmd_reg_rp_scale` | -1.0 | **-2.0** |
| `cmd_reg_yaw_scale` | -0.5 | **-0.05** |

### Deployment
Shares the V6 controller: `controller_simple_policy.py` now hardcodes `self.gate_side = 0.7` (see V6 deployment robustness fix). No config.yaml dependency. V2/V3/V4 rollback uses the locally-saved pre-V6 controller (`gate_side=1.0`).

### Expected outcome
- Lower real-world mean body rate (target < 120 rad/s, i.e. beating V2).
- Similar or slightly slower lap time than V2 (stronger rp penalty may cost ~0.05s/lap).
- If body rate *does not* drop, the bottleneck isn't the reward — likely dynamics mismatch, and the next step should be V7-Residual using real rosbags.

### Experiment Results — Sim batch eval (5000 envs, 3-param DR) 2026-04-15

| Metric | Value |
|--------|-------|
| 3-lap success rate | **100.0 %** (5000/5000) |
| Mean 3-lap time | 8.88 s |
| Best 3-lap time | 8.22 s |
| Worst 3-lap time | 9.58 s |
| 3-lap time std | 0.27 s |
| Mean extra gates | 0.00 (clean) |
| Max tilt (mean / p95 / worst) | 39° / 41° / 49° |
| Peak body-rate (mean frac / envs pegged >0.9) | 1.00 / 100 % |

**Interpretation:** The Pasumarti-style cmd_reg (roll/pitch heavily penalised, yaw almost free) produced the **smoothest and most consistent** policy: worst-case tilt 49° (vs 67° on V6-Smooth and 109° on V6), std 0.27s (≈5× tighter than V6-Smooth). Speed paid for it — mean 8.88s is ~0.8s slower than V6-Smooth. Strongest real-world candidate if V6-Smooth's real deployment shows aggressive behaviour.

### Experiment Results — Real-world @ Pennovation 2026-04-15 (bag: `zhuohao_wk2_cycle_v7`)

| Metric | Value |
|--------|-------|
| Complete laps logged | 11 |
| Total race time (lap 1 → last) | 34.484 s |
| Takeoff → 3 laps done | 10.070 s |
| Fastest 3-lap window | 9.320 s (laps 7–9) |
| Best lap | 2.878 s |
| Mean lap | 3.135 s |
| Median lap | 3.132 s |
| Lap std (consistency) | 0.110 s |
| Path length / lap | ~8.25 m |
| Mean / max speed | 2.63 / 3.30 m/s |
| Mean / max tilt | 28.5 / 39.1 deg |
| Mean / max body rate | 133.94 / 240.89 rad/s |
| Mean / max thrust | 0.66 / 1.17 N |

**Interpretation (real):** Predictions held exactly. V7 produced the **lowest mean tilt and mean body rate** of the real-world batch (28.5°, 134 rad/s) — the roll/pitch-heavy cmd_reg caps attitude aggressively. Cost is real: mean lap 3.135 s (~0.6 s slower than V6), path/lap ~8.25 m (longest of the batch), std 0.11 s (loosest). Strong safety fallback, not the speed choice.

---

## [Circle-V6-Smooth2] - 2026-04-15

### Ablation — V6 + stronger delta action penalty

Identical to V6 except `cmd_smoothness_scale = -0.2`. Controls how aggressively consecutive action changes are penalized.

**Run name:** `circle-v6-smooth2`
**Log folder:** `logs/rsl_rl/quadcopter_direct/<timestamp>_circle-v6-smooth2/`

### Changed (relative to V6)

| Parameter | V6 | V6-Smooth2 |
|-----------|----|----|
| `cmd_smoothness_scale` | 0.0 | **-0.2** |

### Experiment Results — Sim batch eval (5000 envs, 3-param DR) 2026-04-15

| Metric | Value |
|--------|-------|
| 3-lap success rate | 97.6 % (4882/5000) |
| Mean 3-lap time | 7.56 s |
| Best 3-lap time | 6.80 s |
| Worst 3-lap time | 27.30 s |
| 3-lap time std | 1.06 s |
| Mean extra gates | 0.00 (clean) |
| Max tilt (mean / p95 / worst) | 52° / 55° / 93° |
| Peak body-rate (mean frac / envs pegged >0.9) | 0.96 / 100 % |

**Interpretation:** Doubling `cmd_smoothness_scale` (−0.1 → −0.2) did **not** clearly improve over V6-Smooth: SR dropped slightly (99.1% → 97.6%), worst-case tilt worsened (67° → 93°), std grew (0.44s → 1.06s). Stronger delta penalty appears to destabilise the policy rather than smooth it further. V6-Smooth (-0.1) remains the better tuning.

### Experiment Results — Real-world @ Pennovation 2026-04-15 (bag: `zhuohao_wk2_cycle_v6_smooth2`)

| Metric | Value |
|--------|-------|
| Complete laps logged | 9 |
| Total race time (lap 1 → last) | 24.555 s |
| Takeoff → 3 laps done | 8.837 s |
| Fastest 3-lap window | 8.133 s (laps 1–3) |
| Best lap | 2.662 s |
| Mean lap | 2.728 s |
| Median lap | 2.739 s |
| Lap std (consistency) | 0.024 s |
| Path length / lap | ~7.68 m |
| Mean / max speed | 2.81 / 3.38 m/s |
| Mean / max tilt | 32.8 / 44.2 deg |
| Mean / max body rate | 135.38 / 191.50 rad/s |
| Mean / max thrust | 0.68 / 1.11 N |

**Interpretation (real):** Confirmed the sim conclusion — V6-Smooth2 is over-damped. Mean lap 2.728 s is 0.2 s slower than V6-Smooth (2.517 s) with no improvement in consistency. Body rate (135 rad/s) dropped vs V5 (151 rad/s) but is only marginally better than V7 (134 rad/s) while being much faster. The −0.2 sweet spot doesn't exist; −0.1 (V6-Smooth) remains optimal.

---

## [Circle-V6-Smooth] - 2026-04-15

### Ablation — V6 + delta action smoothness penalty

Identical to V6 but adds a per-step penalty on the squared change between consecutive actions:

```
r_smooth = |a_t - a_{t-1}|² × cmd_smoothness_scale
```

**Why:** `cmd_reg` penalizes large absolute body rates but not high-frequency jitter. A policy can have moderate average body rate but still be jerky frame-to-frame. In real hardware, high-frequency commands are attenuated by motor bandwidth — the drone physically can't execute 50Hz control changes. The smoothness penalty explicitly discourages this jitter without slowing the drone down.

**Coefficient choice:** -0.1 is small relative to gate_pass (+200). At typical delta_action ≈ 0.05, per-step penalty ≈ 0.00025 — nudges the policy without dominating.

**Run name:** `circle-v6-smooth`
**Log folder:** `logs/rsl_rl/quadcopter_direct/<timestamp>_circle-v6-smooth/`

### Changed (relative to V6)

| Parameter | V6 | V6-Smooth |
|-----------|----|----|
| `cmd_smoothness_scale` | 0.0 | **-0.1** |

### Experiment Results — Sim batch eval (5000 envs, 3-param DR) 2026-04-15

| Metric | Value |
|--------|-------|
| 3-lap success rate | 99.1 % (4953/5000) |
| Mean 3-lap time | 8.06 s |
| Best 3-lap time | 6.30 s |
| Worst 3-lap time | 18.50 s |
| 3-lap time std | 1.74 s |
| Mean extra gates | 0.00 (clean) |
| Max tilt (mean / p95 / worst) | 57° / 60° / 67° |
| Peak body-rate (mean frac / envs pegged >0.9) | 1.00 / 100 % |

**Interpretation:** Adding the delta-action penalty dramatically tightened worst-case tilt (109° → 67°) while barely hurting SR (98.4% → 99.1%) or best lap (6.52s → 6.30s). Mean time +0.6s vs V6 is a reasonable cost for the stability gain. Best overall candidate for real-world deployment so far. However, 3-lap std remains high (1.74s) — a small fraction of envs still degrade under DR.

### Experiment Results — Real-world @ Pennovation 2026-04-15 (bag: `zhuohao_wk2_cycle_v6_smooth`)

| Metric | Value |
|--------|-------|
| Complete laps logged | 10 |
| Total race time (lap 1 → last) | 25.171 s |
| Takeoff → 3 laps done | 8.341 s |
| Fastest 3-lap window | 7.539 s (laps 2–4) |
| Best lap | 2.505 s |
| Mean lap | 2.517 s |
| Median lap | 2.518 s |
| Lap std (consistency) | **0.007 s** |
| Path length / lap | ~7.65 m |
| Mean / max speed | 3.04 / 3.81 m/s |
| Mean / max tilt | 37.6 / 49.2 deg |

**Interpretation (real):** **Standout of the batch.** Lap std 0.007 s is extraordinary — 4–15× tighter than any other policy. Mean lap 2.517 s is only ~0.02 s slower than V6, so the delta-action penalty costs nearly nothing in speed while producing a near-metronomic flight pattern. The sim prediction (best reliability/speed tradeoff) was validated in reality. **Top choice for race day.**

---

## [Circle-V6] - 2026-04-15

### Stage 2 — V5 + reduced gate_side for safer gate passage

Small tweak to V5: reduce `gate_side` from 1.0 to 0.7 so the policy learns to aim for the center of the gate rather than anywhere within the full opening.

**Reasoning:**

The gate inner opening is 1.0m × 1.0m (from USD asset: inner vertices at ±0.5m). From real-world data (V2), mean gate clearance was 0.443m from gate center, leaving only 0.057m (5.7cm) from the inner edge — tighter than the Crazyflie's half-diagonal (4.6cm). This is risky.

By setting `gate_side = 0.7` (±0.35m virtual corners), the policy learns to aim for a 0.7m × 0.7m virtual opening centered in the real 1.0m × 1.0m gate. Each side has 0.15m physical margin:

```
0.15m margin - 0.046m (Crazyflie half-diagonal) - 0.03m (gate calibration error) ≈ 0.074m real safety buffer
```

`gate_side = 0.8` was considered but only gives 5.4cm real buffer after accounting for drone body — too tight. `gate_side = 0.7` gives ~7.4cm.

**Expected cost:** ~0.1-0.2s per lap slower (policy flies more conservatively through gate center). Acceptable tradeoff for robustness.

**Important:** V6's `gate_side=0.7` must be set in both training (`quadcopter_env.py`) AND real controller (`config.yaml`). Previous versions (V2/V3/V4/V5) used `gate_side=1.0` — revert config.yaml when deploying those checkpoints.

**Deployment robustness fix (2026-04-15):** Relying on `config.yaml` is fragile — at Pennovation the course-provided runtime config may override ours, feeding the policy the wrong gate-corner positions (±0.5m instead of ±0.35m). Fix: treat `params["gate_side"]` as the *physical* gate opening (1.0m) and apply a per-policy `SAFETY_MARGIN` in `controller_simple_policy.py`:

```python
SAFETY_MARGIN = 0.15  # 0 for V2/V3/V4, 0.15 for V6/V7
self.gate_side = params["gate_side"] - 2 * SAFETY_MARGIN
```

`config.yaml` is restored to `gate_side: 1.0` to match the physical-size convention. Works correctly whether TA's runtime config supplies 1.0 or ours does. For V2/V3/V4 rollback, either set `SAFETY_MARGIN = 0.0` or swap in the locally-saved pre-V6 controller.

### Changed (relative to V5)

| Parameter | V5 | V6 | Reason |
|-----------|----|----|--------|
| `gate_side` (train + real) | 1.0 | **0.7** | Push policy to aim for gate center; 0.15m per side accounts for drone body (4.6cm) + calibration error |
| `spline_vel_min` | 0.5 m/s | **1.0 m/s** | (carried from V5) |
| `spline_vel_max` | 1.5 m/s | **3.0 m/s** | (carried from V5) |

### Training
- `num_envs=8192`, `max_iterations=3000`, `seed=42`
- **Run name:** `circle-v6`
- **Log folder:** `logs/rsl_rl/quadcopter_direct/<timestamp>_circle-v6/`

### Experiment Results — Sim batch eval (5000 envs, 3-param DR) 2026-04-15

| Metric | Value |
|--------|-------|
| 3-lap success rate | 98.4 % (4921/5000) |
| Mean 3-lap time | 7.45 s |
| Best 3-lap time | 6.52 s |
| Worst 3-lap time | 19.90 s |
| 3-lap time std | 1.22 s |
| Mean extra gates | 0.00 (clean) |
| Max tilt (mean / p95 / worst) | 56° / 58° / **109°** |
| Peak body-rate (mean frac / envs pegged >0.9) | 1.00 / 100 % |

**Interpretation:** Baseline V6 — fastest mean time of the family (7.45s) but **worst-case tilt spikes to 109°** (near-inverted). Under adversarial DR the policy occasionally loses attitude control while still completing laps. Motivated the V6-Smooth variants which add delta-action / cmd_reg penalties to cap the worst-case tilt.

### Experiment Results — Real-world @ Pennovation 2026-04-15 (bag: `zhuohao_wk2_cycle_v6`)

| Metric | Value |
|--------|-------|
| Complete laps logged | 10 |
| Total race time (lap 1 → last) | 24.971 s |
| Takeoff → 3 laps done | 7.866 s |
| Fastest 3-lap window | **7.375 s** (laps 1–3) |
| Best lap | **2.369 s** |
| Mean lap | **2.497 s** |
| Median lap | 2.503 s |
| Lap std (consistency) | 0.050 s |
| Path length / lap | ~7.65 m |
| Mean / max speed | 3.06 / 4.20 m/s |
| Mean / max tilt | 38.0 / 52.5 deg |

**Interpretation (real):** **Fastest real-world policy.** Best lap 2.369 s, fastest 3-lap 7.375 s, mean speed 3.06 m/s. Sim's 109° worst-case tilt never materialized in 10 laps (real max 52.5°) — the adversarial DR tail that triggers the instability doesn't exist at Pennovation. However, std 0.050 s is 7× worse than V6-Smooth, so there's lap-to-lap variance. Use for speed runs; V6-Smooth for consistency.

---

## [Circle-V5] - 2026-04-15

### Stage 2 — V4 minor variant: higher spline reset velocity

Small tweak to V4: increase spline reset velocity range to better match real flight speed (~2.9 m/s avg observed in V2/V3 real runs).

### Changed (relative to V4)

| Parameter | V4 | V5 | Reason |
|-----------|----|----|--------|
| `spline_vel_min` | 0.5 m/s | **1.0 m/s** | Remove near-hover starts from spline resets (ground resets already cover 0 m/s) |
| `spline_vel_max` | 1.5 m/s | **3.0 m/s** | Match real flight speed; policy needs to learn high-speed gate entry |

All other settings identical to V4 (spline reset, V2 DR ranges, mass ±5%, motor tau 0.7-1.3x, obs noise, gate_side=1.0).

### Training
- `num_envs=8192`, `max_iterations=3000`, `seed=42`

### Experiment Results — Real-world @ Pennovation 2026-04-15 (bag: `zhuohao_wk2_cycle_v5`)

See V5 results in the [Week-2 Real-World Evaluation Summary](#week-2-real-world-evaluation-summary---2026-04-15) and in the consolidated V5 entry above V4-Conservative.

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

### Experiment Results — Real-world @ Pennovation 2026-04-15 (bag: `zhuohao_wk2_cycle_v5`)

| Metric | Value |
|--------|-------|
| Complete laps logged | 12 |
| Total race time (lap 1 → last) | 31.323 s |
| Takeoff → 3 laps done | 8.332 s |
| Fastest 3-lap window | 7.746 s (laps 1–3) |
| Best lap | 2.531 s |
| Mean lap | 2.610 s |
| Median lap | 2.616 s |
| Lap std (consistency) | 0.030 s |
| Path length / lap | ~7.07 m |
| Mean / max speed | 2.71 / 3.54 m/s |
| Mean / max tilt | 32.7 / 43.1 deg |
| Mean / max body rate | 150.87 / 241.71 rad/s |
| Mean / max thrust | 0.72 / 1.17 N |

**Interpretation (real):** V5 ran the most laps in a single session (12). Performance is nearly identical to V4 (mean lap 2.610 vs 2.620 s) — the higher spline reset velocity (1–3 m/s) had negligible real-world impact. Body rate (151 rad/s mean) is the highest of any policy where measured, confirming this baseline has no smoothness penalty.

---

## [Circle-V4] - 2026-04-13

### Experiment Results — Real-world @ Pennovation 2026-04-15 (bag: `zhuohao_wk2_cycle_v4_good`)

| Metric | Value |
|--------|-------|
| Complete laps logged | 9 |
| Total race time (lap 1 → last) | 23.583 s |
| Takeoff → 3 laps done | 8.455 s |
| Fastest 3-lap window | 7.791 s (laps 1–3) |
| Best lap | 2.532 s |
| Mean lap | 2.620 s |
| Median lap | 2.631 s |
| Lap std (consistency) | 0.035 s |
| Path length / lap | ~7.08 m |
| Mean / max speed | 2.70 / 3.36 m/s |
| Mean / max tilt | 32.7 / 48.8 deg |

**Interpretation (real):** Solid baseline. V4's spline reset + moderate DR produces a reliable, consistent policy (std 0.035 s). Speed similar to V5 but tighter path (~7.08 m/lap). V4 and V5 together form the "V2-class" baseline that V6+ improved upon.

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

> **Note:** Results use v3 strict-order lap timer (gate 0→1→2→3→0 enforced). v2 timer had reported 5 laps; strict ordering reveals only 3 valid laps with 4 sequence breaks (drone skipped/misordered gates).

| Metric | Value |
|--------|-------|
| Complete laps (strict order) | **3** |
| Sequence breaks | **4** |
| Takeoff → 3 laps done | **19.068 s** |
| Fastest 3-lap window | 8.568 s |
| Best lap | 2.602 s |
| Mean lap | 2.856 s |
| Median lap | 2.949 s |
| Lap std (consistency) | 0.182 s |
| Path length / lap | ~17.04 m |
| Mean / max speed | 2.90 / 4.05 m/s |
| Mean / max tilt | 32.1 / 51.2 deg |
| Mean / max body rate | 93.72 / 243.48 rad/s |
| Mean / max thrust | 0.67 / 1.17 N |

**Interpretation:** Worse than previously thought. Strict-order analysis shows V4-Cons only completed 3 valid laps with 4 sequence breaks — the drone repeatedly skipped or misordered gates. Takeoff→3 laps is 19.068 s (>2× slower than V3). The reduced `lap_incomplete_penalty` (-0.02 vs -0.05) caused the policy to wander between gates and miss the required gate order. **Not viable.**

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

> **Note:** Results below use the v2 gate-plane-crossing lap timer (full gate-0→gate-0 laps).

| Metric | Value |
|--------|-------|
| Complete laps logged | 16 |
| Total race time (lap 1 → last) | 38.966 s |
| Takeoff → 3 laps done | 7.927 s |
| Fastest 3-lap window | 7.273 s (laps 9–11) |
| Best lap | 2.409 s |
| Mean lap | 2.435 s |
| Median lap | 2.431 s |
| Lap std (consistency) | 0.020 s |
| Path length / lap | ~7.08 m |
| Mean / max speed | 2.91 / 3.97 m/s |
| Mean / max tilt | 37.8 / 49.1 deg |
| Mean / max body rate | 154.59 / 235.07 rad/s |
| Mean / max thrust | 0.71 / 1.17 N |

**Interpretation:** Most laps completed in a single run (16). Very consistent (std 0.020 s). Higher body rate than V2 (155 vs 127 rad/s) — tighter DR didn't help smoothness in real. Slightly faster than V2 (mean 2.435 vs 2.697 s). Fastest 3-lap window 7.273 s is the best of V2/V3.

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

> **Note:** Results below use the v2 gate-plane-crossing lap timer (full gate-0→gate-0 laps).

| Metric | Value |
|--------|-------|
| Complete laps logged | 15 |
| Total race time (lap 1 → last) | 40.461 s |
| Takeoff → 3 laps done | 8.151 s |
| Fastest 3-lap window | 7.637 s (laps 1–3) |
| Best lap | 2.525 s |
| Mean lap | 2.697 s |
| Median lap | 2.596 s |
| Lap std (consistency) | 0.297 s |
| Path length / lap | ~7.58 m |
| Mean / max speed | 2.81 / 3.73 m/s |
| Mean / max tilt | 34.3 / 50.3 deg |
| Mean / max body rate | 126.54 / 244.61 rad/s |
| Mean / max thrust | 0.77 / 1.17 N |

Per-lap breakdown (selected):

| Lap | Time (s) | Path (m) | v_avg | v_max | br_avg | br_max |
|-----|----------|----------|-------|-------|--------|--------|
| 1 | 2.53 | 7.60 | 3.03 | 3.73 | 128.52 | 243.88 |
| 2 | 2.54 | 7.28 | 2.87 | 3.12 | 124.07 | 243.33 |
| 3 | 2.57 | 7.25 | 2.84 | 3.13 | 120.60 | 244.56 |
| 9 | 3.08 | 7.61 | 2.48 | 3.42 | 154.59 | 241.46 |
| 10 | 3.70 | 9.30 | 2.52 | 3.45 | 129.23 | 224.62 |

**Interpretation:** V2 completed 15 laps, but laps 9–10 had a significant hiccup (3.08 s + 3.70 s), inflating mean to 2.697 s and std to 0.297 s. Excluding the hiccup, steady-state laps are ~2.55 s with good body rate (127 rad/s). Not as consistent as V3 (std 0.020 s). Lowest body rate of the V2/V3 pair.

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
