# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Created this changelog file to track project modifications.
- Implemented the PPO algorithm update step in `ppo.py`, including surrogate loss, clipped value loss, entropy bonus, and model optimization.
- Created convenient bash scripts (`scripts/run/train.sh` and `scripts/run/play.sh`) to easily run training and evaluation.

### Changed
- Refactored `compute_returns()` in `rollout_storage.py` to use a GPU-optimized vectorized calculation for generalized advantage estimation (GAE). By computing the TD errors (deltas) in parallel across the sequence, PyTorch kernel launch overhead is minimized during loop iteration.
