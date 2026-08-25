# Drone Racing — RL Policy Training (Isaac Lab)

ESE 6510 *Physical Intelligence* (UPenn) drone racing project by **Zhuohao Ni** and **Bruce (Yuqian) Zhang**.

We train a quadcopter racing policy with PPO in NVIDIA Isaac Lab and then fly it on a real
Crazyflie at Pennovation. The work splits into two phases, and this repository keeps one
branch per phase.

> **You are on `main` — the simulation-only racing phase.**
> For the real-hardware phase, see the [`sim2real`](../../tree/sim2real) branch.

---

## Branch map

| Branch | Phase | What it optimizes | Track | Success criterion |
|---|---|---|---|---|
| **`main`** | Phase 1 — sim racing | Lap time under the course's evaluation-time domain randomization | `powerloop` (7 gates: vertical loop + chicane) | 3 laps, no crash, fastest mean time |
| **`sim2real`** | Phase 2/3 — real deployment | Zero-crash transfer to a physical Crazyflie over Vicon | `circle` (4 gates) → `powerloop` | 3 real laps, no crash; speed is secondary |

The two branches share the same Isaac Lab task skeleton but diverge in almost every
learning-side decision — observation layout, reward terms, reset sampling, and domain
randomization — because they are solving different problems. Sim racing rewards speed at the
edge of the sim's dynamics; sim2real rewards behavior that survives a physical Crazyflie whose
mass, thrust authority, and state estimate the simulator only approximates. Details of the
divergence are in the `sim2real` branch README.

Deployment code (ROS 2, Vicon, Crazyradio, the real controller node) lives in the **sibling
repository** [`Drone-Racing-sim2real`](https://github.com/brucehacker66/Drone-Racing-sim2real),
expected to be checked out next to this one as `../sim2real/`.

---

## Where the code comes from

This repo is **not** written from scratch. It is a student fork of the course project
scaffold, with our own learning-side implementation layered on top.

**Inherited (upstream, not ours):**

- **Course starter repo** — [`Jirl-upenn/ese651_project`](https://github.com/Jirl-upenn/ese651_project)
  (also distributed as `vineetpasumarti/ese651_project`). Provides the Isaac Lab task
  scaffold: `QuadcopterEnv` physics at 500 Hz, the PID body-rate controller and motor model,
  contact-based crash detection, gate USD assets in `usd/`, the strategy-pattern hooks, and
  the `train_race.py` / `play_race.py` entry points. Reward, observation, and reset methods
  were left as TODO stubs for students.
- **`rsl_rl`** — vendored at `src/third_parties/rsl_rl_local/`, from the
  [Robotic Systems Lab, ETH Zurich & NVIDIA](https://github.com/leggedrobotics/rsl_rl)
  (BSD-3-Clause, see `src/third_parties/rsl_rl_local/LICENSE`), routed through the course fork
  [`Jirl-upenn/rsl_rl`](https://github.com/Jirl-upenn/rsl_rl). The PPO **update step** was
  shipped as a TODO block for students to fill in.
- **`rotorpy`** — submodule from [`Jirl-upenn/rotorpy`](https://github.com/Jirl-upenn/rotorpy).
- **Isaac Lab / Isaac Sim** — NVIDIA. Must be installed at the same directory level as this repo.

**Ours (what we wrote):**

- The PPO update step in `rsl_rl/algorithms/ppo.py` (clipped surrogate, clipped value loss,
  entropy bonus, adaptive-KL learning-rate schedule).
- A GPU-vectorized GAE rewrite in `rsl_rl/storage/rollout_storage.py`.
- The entire task definition in `quadcopter_strategies.py` — observations, reward shaping,
  reset sampling, powerloop guidance, and domain randomization.
- All evaluation tooling under `scripts/rsl_rl/` and `scripts/run/` beyond the two starter
  entry points.

---

## What the `main` branch does

Trains a policy to fly the **powerloop** track — 7 waypoints including a central double-gate
structure (gates 2 & 3 entered from the same side) and a chicane where gate 6 is physically
the same object as gate 3 but passed in the opposite direction. Evaluation is 3 laps under
randomized thrust-to-weight, aerodynamic drag, and PID gains.

**Design summary** (full derivation in [`docs/writeup.tex`](docs/writeup.tex) /
[`docs/writeup.pdf`](docs/writeup.pdf)):

- **Observation (31-dim, ego-centric).** Body-frame linear + angular velocity (6), gravity
  vector in body frame (3) instead of a quaternion, current/next/next-next gate positions in
  body frame (9), current + next gate normals (6), normalized gate index (1), sin/cos of yaw
  error (2), previous action (4).
- **Reward.** Gate pass +200, progress +50, racing-line velocity +8, orientation penalty −1
  (activates past ~30° of roll+pitch), action smoothness −0.2, crash −1, death −10.
- **Corner clipping without waypoints.** The racing-line velocity target blends the current
  gate direction (5/8) with the next gate direction (3/8), so the policy learns to cut the
  inside of corners on its own. The powerloop segment uses the current gate only.
- **Powerloop bypass.** A 2-phase virtual target (apex `[0, −0.3, 1.6]` → offset target
  `[0.425, 0, 0.75]`) that takes the drone around the *side* of the double gate rather than
  over the top — a legal pass and measurably faster.
- **Reset sampling.** Random starting gate, curriculum spawn distance, 40% mid-track spawns
  (the single biggest robustness win: 86.7% → 96.3% success), ~20% ground starts, 50% of air
  spawns with 0.5–3.0 m/s initial velocity.
- **Domain randomization.** Re-sampled every reset during training, deliberately wider than
  the evaluation bounds (TWR ±8%, drag 0.3–2.5×, PID ±25–40%); nominal at eval time.

**Results** (1000–5000 randomized envs, official evaluation bounds):

| Metric | V26 (baseline) | V30 (fine-tuned) |
|---|---:|---:|
| 3-lap success rate | 94.9% | **97.3%** |
| Mean 3-lap time | 15.78 s | **15.68 s** |
| Std dev | 0.54 s | **0.50 s** |

V30 was fine-tuned from V26 for 2000 extra iterations under the wider DR bounds with
`entropy_coef=0.001`. A paired evaluation on identical DR seeds showed V30 rescues 78.5% of
the seeds V26 failed, at the cost of some regression on V26's easiest cases.

---

## Layout

```
src/isaac_quad_sim2real/tasks/race/config/crazyflie/
  quadcopter_env.py         # inherited: physics, PID rate control, motors, crash detection, tracks
  quadcopter_strategies.py  # ours: rewards, observations, resets, powerloop guide, DR
  agents/rsl_rl_ppo_cfg.py  # PPO hyperparameters (actor [256,256], critic [512,256,128,128], ELU)
src/third_parties/rsl_rl_local/   # vendored rsl_rl fork; PPO + GAE are ours
scripts/rsl_rl/             # train_race, play_race, eval_race, batch_eval_race, paired_eval, play_ta_style
scripts/run/                # thin bash launchers for the above
usd/                        # gate + Crazyflie assets (inherited)
docs/                       # writeup, changelog, course project description
logs/rsl_rl/quadcopter_direct/<timestamp>_<run_name>/   # checkpoints, videos, exports
```

Task id: `Isaac-Quadcopter-Race-v0`, registered in
`src/isaac_quad_sim2real/tasks/race/config/crazyflie/__init__.py`.
Tracks defined in `quadcopter_env.py`: `powerloop` (default here), `complex`, `lemniscate`.

---

## Running it

Conda environment: `ese6510` (what every script in `scripts/run/` activates; `.envrc` names a
separate `isaac_quad_sim2real` env). Isaac Lab must sit at the same directory level as this repo.

```bash
# Train
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 16384 --max_iterations 5000 --headless --logger wandb
# or: ./scripts/run/train.sh

# Play back one run and record video
python scripts/rsl_rl/play_race.py \
    --task Isaac-Quadcopter-Race-v0 --num_envs 1 \
    --load_run <YYYY-MM-DD_HH-MM-SS> --checkpoint best_model.pt \
    --headless --video --video_length 1600
# or: ./scripts/run/play.sh <run_dir>

# Batch evaluation under evaluation-style randomization (3-lap SR + time distribution)
./scripts/run/batch_eval.sh <run_dir> [num_trials] [num_envs] [max_steps] [checkpoint]

# Paired A/B of two checkpoints on identical DR draws
./scripts/run/paired_eval.sh
```

`play_race.py` also exports the trained policy as TorchScript and ONNX, which is what the
deployment repo consumes.

---

## Docs

| File | Contents |
|---|---|
| [`docs/writeup.tex`](docs/writeup.tex) / `writeup.pdf` | Final Phase-1 strategy writeup — algorithm, obs, reward, powerloop, resets, DR, results |
| `docs/tech_report.tex` / `tech_report.pdf` | Full technical report covering **both** phases, sim through real flight (currently untracked — commit it if you want it in the repo) |
| [`docs/changelog.md`](docs/changelog.md) | Every experiment V1→V31 with the change, the motivation, and the measured outcome |
| [`docs/project_description.md`](docs/project_description.md) | Course handout: track definition, evaluation protocol, DR bounds |
| [`docs/strategy_design.md`](docs/strategy_design.md), [`docs/how_drone_racing_works.md`](docs/how_drone_racing_works.md) | Working notes |

**Experiment discipline:** every reward/config change gets a numbered entry in
`docs/changelog.md` with its measured result before the next change is made. Reverted
experiments stay in the log — the negative results are the most reusable part of it.

---

## License

GPL-3.0 (`LICENSE`), inherited from the course project repo. The vendored `rsl_rl` retains its
own BSD-3-Clause license at `src/third_parties/rsl_rl_local/LICENSE`.
