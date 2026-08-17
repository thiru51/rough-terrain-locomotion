# Status

What actually works, updated per commit, so I don't end up claiming something I haven't run.

**Nothing has been trained. No number in this repo is measured.**

Legend — **[C]** core locomotion system · **[E]** engineering extension around it

## Completed

Written, reviewed, and covered by passing tests.

| Item | | Notes |
|---|---|---|
| Observation/privileged layout (`tert/envs/obs_spec.py`) | [C] | 48 proprio / 203 privileged / 251 total |
| Causal Transformer (`tert/models/transformer.py`) | [C] | Causality and attention-normalisation asserted in tests |
| Privileged encoder (`tert/models/encoder.py`) | [C] | 187/12/4 head widths |
| Teacher actor-critic (`tert/models/teacher.py`) | [C] | Shape-tested only; never trained |
| Observation normaliser (`tert/data/normalizer.py`) | [C] | Streaming fit; freezes after stage 1 |
| Rolling context window (`tert/data/context.py`) | [C] | Shared by online correction and deployment |
| Windowed dataset (`tert/data/dataset.py`) | [C] | Front-padded to match the context window |
| Rollout collection, both stages (`tert/training/collect.py`) | [C] | Exercised against a stub env only |
| Shared imitation loop (`tert/training/imitation.py`) | [C] | Both passes share one optimiser |
| VecEnv contract (`tert/backends/interface.py`) | [C] | No simulator bound yet |
| Reward terms and composer (`tert/envs/rewards.py`) | [C] | Includes the two smoothness penalties |
| Terrain curriculum (`tert/envs/terrain.py`) | [C] | NumPy heightfield; replaces `isaacgym.terrain_utils` |
| A1 constants (`configs/robot/a1.yaml`) | [C] | Gains, scales, randomisation ranges |
| PPO + rollout storage (`tert/training/ppo.py`) | [C] | GAE, clipped value loss, adaptive-KL lr |
| Teacher training loop (`tert/training/teacher_runner.py`) | [C] | Runs against the stub env only |
| Shared actor-critic base (`tert/models/actor_critic.py`) | [C] | Teacher and baselines differ only in `_features` |
| Latent-estimator baselines (`tert/models/baselines/latent_policy.py`) | [C] | Frozen teacher actor + swappable estimator |
| TCN encoder (`tert/models/baselines/tcn.py`) | [C] | 50-step history |
| PPO / StackedPPO (`tert/models/baselines/stacked.py`) | [C] | Stacking is a stateless obs transform |
| Reference notes (`docs/reference.md`) | [C] | Where every constant came from |
| Attribution ledger (`THIRD_PARTY.md`) | [C] | |

## Implemented but unverified

Code exists and imports cleanly, but has not been validated against real physics.

| Item | | Blocker |
|---|---|---|
| — | | |

## Partially implemented

| Item | | Missing |
|---|---|---|
| — | | |

## Planned

| Item | | |
|---|---|---|
| Isaac Gym backend binding the env to the simulator | [C] | Requires Isaac Gym Preview under WSL2 |
| Stage driver scripts wiring collection → fit → checkpoint | [C] | |
| GRU baseline + recurrent PPO storage | [C] | Needs sequence minibatches; not started |
| RMA latent-regression training loop | [C] | Models exist; the fit loop does not |
| Evaluation: return, smoothness, energy, fall rate, latency | [C] | |
| Ablations: single-pass, TCN encoder, latent target | [C] | |
| EKF / UKF base-state estimation | [E] | |
| Sensor models: IMU, encoders, contact, depth | [E] | |
| Frame transforms (world/body/sensor/joint) | [E] | |
| PD, trajectory tracking, MPC interface | [E] | |
| Terrain heightmap and traversability perception | [E] | |
| C++ TorchScript inference runner at 50 Hz | [E] | |
| ROS2 nodes, launch files, TF2 | [E] | |
| Marine (AUV/ASV) transfer analysis | [E] | Research note, not code |

## Known environment constraints

- Isaac Gym Preview is Linux-only and login-walled at NVIDIA; the target runtime is
  **WSL2 + Isaac Gym Preview 4** on an RTX 4080 Laptop (12 GB).
- The official stack is Python 3.8 + torch 1.10+cu113. This repo targets Python ≥3.10 and
  torch ≥2.2; divergence is recorded in `THIRD_PARTY.md`.
- Teacher training is the dominant cost: 20,000 PPO iterations × 4096 envs.

## On results

There are none yet, and the README says so. When the comparison and ablation scripts have
actually been run, their raw outputs go in `results/` and the claims here get updated to
match — not before.