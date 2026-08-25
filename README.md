# Drone Racing — Sim-to-Real Training (Isaac Lab)

ESE 6510 *Physical Intelligence* (UPenn) drone racing project by **Zhuohao Ni** and **Bruce (Yuqian) Zhang**.

> **You are on `sim2real` — the branch that trains policies meant to fly on real hardware.**
> For the simulation-only racing phase, see the [`main`](../../tree/main) branch.

This branch is still the **training-side** repo: Isaac Lab, PPO, reward/observation/reset
design. What changed is the objective. On `main` the target was the fastest lap under the
course's evaluation randomization. Here the target is a policy that survives contact with a
physical Crazyflie flying over Vicon at Pennovation, with **zero tolerance for crashes**.
Everything else follows from that.

---

## Branch map

| Branch | Phase | What it optimizes | Track | Success criterion |
|---|---|---|---|---|
| [`main`](../../tree/main) | Phase 1 — sim racing | Lap time under evaluation-time domain randomization | `powerloop` (7 gates) | 3 laps, no crash, fastest mean time |
| **`sim2real`** | Phase 2/3 — real deployment | Zero-crash transfer to a physical Crazyflie | `circle` (4 gates) → `powerloop` | 3 real laps, no crash; speed second |

### What actually differs from `main`

| | `main` | `sim2real` |
|---|---|---|
| Observation | 31-dim: body vel, gravity vector, 3-gate lookahead centers, gate normals, gate index, yaw sin/cos, prev action | **40-dim: body linear velocity (3) + rotation matrix (9) + current gate 4 corners in body frame (12) + next gate 4 corners (12) + prev action (4)** — byte-for-byte the vector the real controller builds |
| Actor | `[256, 256]` | `[512, 512, 256, 128]` (matches the deployed network) |
| Obs normalization | `empirical_normalization = True` | **False** — the running normalizer is not exported to the real controller |
| Reward | Dense speed shaping: progress 50, racing-line velocity 8 | Sparse/event core: gate pass 200, light progress 20, lap bonus, command regularization on body rates, action smoothness, crash penalties |
| Reset | Gate-relative spawns, mid-track spawns, velocity init | Gate-relative + **successful-state replay resets** (staged in over training) + ground/real-start resets |
| Domain randomization | Wide, tuned to beat an evaluation script | Tuned to *measured* hardware: nominal TWR from real bags, narrow band around it, plus latency and observation-mismatch models |
| Strategy class | `DefaultQuadcopterStrategy` | `CircleQuadcopterStrategy` (the default; the name is historical, it is **not** tied to the `circle` track) |

---

## Where the code comes from

Same inheritance as `main` — this is a student fork of the course scaffold, not original
infrastructure.

**Inherited (upstream, not ours):**

- **Course starter repo** — [`Jirl-upenn/ese651_project`](https://github.com/Jirl-upenn/ese651_project)
  (also distributed as `vineetpasumarti/ese651_project`): `QuadcopterEnv` physics at 500 Hz,
  PID body-rate control and the motor model, contact-based crash detection, gate USD assets,
  the strategy-pattern hooks, and the `train_race.py` / `play_race.py` entry points.
- **`rsl_rl`** — vendored at `src/third_parties/rsl_rl_local/`, from the
  [Robotic Systems Lab, ETH Zurich & NVIDIA](https://github.com/leggedrobotics/rsl_rl)
  (BSD-3-Clause), via [`Jirl-upenn/rsl_rl`](https://github.com/Jirl-upenn/rsl_rl). The PPO
  update step was a TODO block for students.
- **`rotorpy`** — submodule from [`Jirl-upenn/rotorpy`](https://github.com/Jirl-upenn/rotorpy).
- **Isaac Lab / Isaac Sim** — NVIDIA.

**Ours:** the PPO update and vectorized GAE, `CircleQuadcopterStrategy` in its entirety
(40-dim observation, sparse reward core, replay/ground reset sampling, split gate semantics,
sim2real randomization), the override mechanism in `train_race.py`, everything in
`scripts/run/` and `scripts/analysis/`, `robustness_sweep.py`, the checkpoint-selection
pipeline, and the rosbag analysis under `rosbags*/`.

**Method influences** (papers in `docs/`): Song et al. 2021/2023 (dense gate progress +
successful-state replay resets), Kaufmann et al. 2023 *Swift* (`docs/swift.pdf` —
controller-aligned observation, previous-action input, smooth-action regularization), Ferede
et al. 2025 (moderate DR helps transfer, excessive DR costs speed), and
`docs/Agile_Flight.pdf`.

---

## The sim2real contract

The one constraint that overrides everything else in this branch: **the observation vector and
network architecture must match the real controller exactly.**

```
this branch (Isaac Lab)                  ../sim2real/ (ROS 2)
quadcopter_strategies.py                 controller_simple_policy.py
  get_observations() → 40-dim      ←→      update() → 40-dim   (must match exactly)
  train → best_model.pt            →       load .pt via config.yaml → /ctbr_cmd → drone
```

Gate corners are built with identical math on both sides:
`corners_w = local_square @ R_gate.T + gate_pos`, then `corners_b = (corners_w - drone_pos) @ R_body`.

A local read-only snapshot of the real controller is kept at
[`docs/controller_simple_policy.py`](docs/controller_simple_policy.py) so sim-side changes can
be checked against it without switching repos. **The sim side adapts to the controller, not
the other way around** — the deployment code is tested against hardware and much harder to debug.

---

## How the work progressed

Full detail in [`docs/changelog.md`](docs/changelog.md) (2500+ lines, newest entry first).
The arc:

**1. Circle track, zero-shot transfer (Apr 9–16).** New 36→40-dim controller-matched
observation, safety-first reward (progress 50→20, velocity 8→1, smoothness −0.2→−1.5, death
−10→−50), no velocity-initialized spawns. Versions V1→V7 were flown and each flight was
rosbagged and re-analyzed with a strict-order lap timer (a lap only counts if gates go
0→1→2→3→0).

> **All circle policies completed laps with zero crashes.** Best race-day config: V3 —
> takeoff→3 laps in **7.93 s** over 16 laps, 0 sequence breaks, lap std 0.020 s.
> V6 (`gate_side=0.7`) was 0.06 s behind with a 0.15 m gate safety margin.

**2. Fixing the analysis, not the policy (Apr 16).** Several "sim2real gaps" turned out to be
bugs on the analysis/controller side: the real gate switch used center distance instead of
plane-crossing, racing thrust used a linear PWM map instead of the calibrated nonlinear one,
sysid mixed deg/s with rad/s, and `obs_latency_prob` was a dead config field. Fixing those and
freezing a **post-fix baseline** was the single most valuable step of the phase.

**3. What actually breaks transfer (Apr 19).** A 13-scenario deterministic robustness sweep
(`scripts/rsl_rl/robustness_sweep.py`, 768 ground starts each) asked one question: control
mismatch or observation mismatch?

| Scenario | 3-lap SR | Δ vs nominal |
|---|---:|---:|
| nominal | 96.6% | — |
| thrust ±10%, rate gain ±15%, latency 20/40 ms | 95.6–100% | ≈ 0 |
| zero-mean velocity noise σ=0.05 | 96.5% | −0.1 pp |
| gate corner bias 5 cm | 92.4% | −4.2 pp |
| **body-velocity bias 0.20 m/s** | **2.9%** | **−93.8 pp** |

> **Systematic observation bias is the dominant failure mode — not control mismatch.**
> A constant 0.2 m/s error in estimated body velocity destroys the policy while ±10% thrust
> error does nothing. This redirected the whole effort toward the Vicon→`/odom`→`/observations`
> pipeline (see [`docs/sysid_plan.md`](docs/sysid_plan.md)).

**4. Real dynamics, not guessed dynamics (Apr 20–22).** Real bags consistently identified an
effective vehicle mass near **64 g** against the sim's 30 g. With the real thrust map capping
at ≈1.174 N, effective thrust-to-weight is `1.174 / (0.064 × 9.81) ≈ **1.87**`, not the sim's
3.15. Retraining with `thrust_to_weight = 1.87` and a narrow ±6% band — instead of widening DR
around a wrong nominal — produced the deployment policy: **98.0% 3-lap SR, 18.34 s mean** over
768 envs.

**5. Real powerloop flight (Apr 22).** `twr1p87` base flew the official 3-lap eval
(first `/ctbr_cmd` race command → 3rd lap's Gate 6) in **19.004 s**, 5 ordered laps, best lap
6.121 s, tightest clearance 0.032 m at Gate 3. Sim ranking and real ranking agreed; the two
"speed" variants were worse in both.

**6. Checkpoint selection is its own problem (Apr 27).** `best_model.pt` is chosen by RSL-RL on
mean *training reward*, which does not track 3-lap success, gate clearance, or worst-case tail
— one example checkpoint won on reward and scored 84.7% SR with a 28.4 s worst case. Worse, the
batch eval ran with the training reset mixture, so a "1000-env" eval contained only ~210 ground
starts while the real race always starts from the ground. Both were fixed: `REAL_GROUND_ONLY=1`
forces 1000/1000 ground starts, and `eval_powerloop_checkpoint_candidates.sh` ranks many
checkpoints by deployment metrics.

> Final selection, 1000/1000 ground starts: `twr1p87-5000` seed-42 `best_model.pt` —
> 100% SR, 100% ground SR, **18.10 s** mean, 0.08 s std.

---

## Layout

```
src/isaac_quad_sim2real/tasks/race/config/crazyflie/
  quadcopter_env.py         # physics + QuadcopterEnvCfg (all DR / reset / mismatch knobs, tracks)
  quadcopter_strategies.py  # DefaultQuadcopterStrategy (legacy) + CircleQuadcopterStrategy (current)
  agents/rsl_rl_ppo_cfg.py  # actor [512,512,256,128], critic [512,512,256,256,128,128], ELU
scripts/rsl_rl/             # train_race, play_race, eval_race, batch_eval_race, paired_eval, robustness_sweep
scripts/run/                # experiment launchers — thin wrappers that set JSON override blobs
scripts/analysis/           # bag_ordered_metrics.py — offline rosbag metrics
docs/                       # changelog, sysid plan, metrics guide, controller snapshot, papers
rosbags/                    # Apr 9-13 circle flights (group30_cyclev2/3/4) + analysis.md
rosbags_04_15/              # Apr 15 circle version sweep (V4–V7) + dynamics_analysis.md
rosbags_powerloop_baseline_controller_04_20/  # first real powerloop flights + sysid
rosbags_apr22/              # Apr 22 powerloop flights — the official 19.004 s run
rosbags_sysid/              # dedicated system-identification flights
```

Tracks in `quadcopter_env.py`: `circle` (default here), `powerloop`, `complex`, `lemniscate`.
`track_name` and `strategy_class` are independent — `CircleQuadcopterStrategy` runs the
powerloop track too.

---

## Running it

Conda environment `ese6510`; Isaac Lab at the same directory level as this repo.

Most experiments are configured by **JSON override blobs in environment variables**, not by
editing config classes — `ENV_OVERRIDES`, `REWARD_OVERRIDES`, `PPO_OVERRIDES` are parsed by
`train_race.py` and the eval scripts. When two run scripts behave differently, diff their
override blobs first.

```bash
# Circle track (Phase 2 baseline)
./scripts/run/train_circle.sh
./scripts/run/eval_circle.sh <run_dir>

# Powerloop with real-calibrated thrust authority (the deployment line)
./scripts/run/train_powerloop_fullswitch_real_twr.sh 3000 8192
./scripts/run/eval_powerloop_real_twr.sh <run_dir> best_model.pt 1000   # ground-only by default

# Rank many checkpoints by deployment metrics instead of trusting best_model.pt
REAL_GROUND_ONLY=1 ./scripts/run/eval_powerloop_checkpoint_candidates.sh <run_dir> 1000 20 3000

# Deterministic control-vs-observation mismatch sweep
./scripts/run/robustness_sweep.sh <run_dir> [checkpoint]

# One-off override without a new script
ENV_OVERRIDES='{"track_name":"powerloop","thrust_to_weight":1.87}' \
  python scripts/rsl_rl/train_race.py --task Isaac-Quadcopter-Race-v0 \
  --num_envs 8192 --max_iterations 3000 --headless --logger wandb

# Analyze real flight bags (strict-order lap timing, clearances, segment times)
bash scripts/run/batch_test_bags.sh rosbags_apr22
bash scripts/run/test_bag.sh rosbags_apr22/<bag_dir> auto 3
```

Deeper bag processing (`analyze_rosbag.py`, `process_bags.sh`) lives in the sibling
`../sim2real/bin/`; the field schema is documented in
[`docs/ros_bags_fields.md`](docs/ros_bags_fields.md).

---

## Docs

| File | Contents |
|---|---|
| [`docs/changelog.md`](docs/changelog.md) | Every sim2real experiment, newest first — the primary record |
| [`docs/changelog_powerloop.md`](docs/changelog_powerloop.md) | Archived Phase-1 changelog (V1→V31), carried over from `main` |
| [`docs/metrics.md`](docs/metrics.md) | How to read each wandb/tensorboard metric and what a bad curve means |
| [`docs/sysid_plan.md`](docs/sysid_plan.md) | System-ID flight plan: what to log, phase ordering, and how it feeds DR |
| [`docs/ros_bags_fields.md`](docs/ros_bags_fields.md) | ROS topic/field schema for the recorded bags |
| [`docs/controller_simple_policy.py`](docs/controller_simple_policy.py) | Read-only snapshot of the real controller — the observation contract |
| [`docs/ideas.md`](docs/ideas.md), [`docs/ESE6510_Intro_sim2real.md`](docs/ESE6510_Intro_sim2real.md) | Background and open directions |
| `rosbags*/analysis.md`, `dynamics_analysis.md`, `sysid_analysis.md` | Per-session flight analyses |
| [`docs/archive/`](docs/archive) | Superseded Phase-1 design notes — kept for history, not authoritative |
| `docs/swift.pdf`, `docs/Agile_Flight.pdf` | Reference papers |

The combined technical report covering both phases (`docs/tech_report.tex`) lives on the
[`main`](../../tree/main) branch.

**Experiment discipline:** every change gets a dated `docs/changelog.md` entry with its
motivation and its measured result before the next change starts, and comparisons are run
systematically (sweep scripts, paired evals) rather than one-off. Negative results stay in the
log — spline resets, extra mass/motor-tau DR, and both "speed" powerloop variants are recorded
as failures on purpose.

---

## License

GPL-3.0 (`LICENSE`), inherited from the course project repo. The vendored `rsl_rl` retains its
own BSD-3-Clause license at `src/third_parties/rsl_rl_local/LICENSE`.
