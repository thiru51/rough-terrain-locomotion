# Implementation Status

Updated per commit. **No experiment has been run. No result in this repository is
measured.** Any number quoted from the paper is labelled as the authors' result, never ours.

Legend — **[P]** paper component · **[E]** our engineering extension

## Completed

Written, reviewed, and covered by passing tests.

| Item | | Notes |
|---|---|---|
| Observation/privileged layout (`tert/envs/obs_spec.py`) | [P] | 48/203/251 verified against official source |
| Causal Transformer (`tert/models/transformer.py`) | [P] | Causality and attention-normalisation asserted in tests |
| Privileged encoder (`tert/models/encoder.py`) | [P] | 187/12/4 head widths |
| Teacher actor-critic (`tert/models/teacher.py`) | [P] | Shape-tested only; never trained |
| Observation normaliser (`tert/data/normalizer.py`) | [P] | Streaming fit; freezes after stage 1 |
| Rolling context window (`tert/data/context.py`) | [P] | Shared by online correction and deployment |
| Windowed dataset (`tert/data/dataset.py`) | [P] | Front-padded to match the context window |
| Rollout collection, both stages (`tert/training/collect.py`) | [P] | Exercised against a stub env only |
| Shared imitation loop (`tert/training/imitation.py`) | [P] | Eq. 6 and Eq. 7 are one optimiser |
| VecEnv contract (`tert/backends/interface.py`) | [P] | No simulator bound yet |
| Reward terms and composer (`tert/envs/rewards.py`) | [P] | Includes TERT's two sim-to-real additions |
| Terrain curriculum (`tert/envs/terrain.py`) | [P] | NumPy heightfield; replaces `isaacgym.terrain_utils` |
| A1 constants (`configs/robot/a1.yaml`) | [P] | Transcribed from the official release |
| PPO + rollout storage (`tert/training/ppo.py`) | [P] | GAE, clipped value loss, adaptive-KL lr |
| Teacher training loop (`tert/training/teacher_runner.py`) | [P] | Runs against the stub env only |
| Paper→code map (`docs/code_map.md`) | [P] | |
| Attribution ledger (`THIRD_PARTY.md`) | [P] | |

## Implemented but unverified

Code exists and imports cleanly; behaviour has not been validated against the simulator
or the paper.

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
| Isaac Gym backend binding the env to the simulator | [P] | Requires Isaac Gym Preview under WSL2 |
| Stage driver scripts wiring collection → fit → checkpoint | [P] | |
| Baselines: RMA/TCN, PPO, StackedPPO, GRU | [P] | |
| Evaluation: return, smoothness, energy, fall rate, latency | [P] | |
| Ablations: w/o-OC, w/o-OP, TCN, Latent | [P] | |
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

## Reproduction claim

**None yet.** This repository is an *independent implementation in progress*. The wording
will only change to a reproduction claim when the ablation and comparison scripts have
actually been run and the outputs are committed under `results/`.
