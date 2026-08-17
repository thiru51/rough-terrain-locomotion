# Terrain Transformer — Independent Implementation

**Implementation in progress.** This repository does not reproduce the paper's results;
nothing here has been trained or evaluated. See [`docs/status.md`](docs/status.md).

An independent implementation of **Terrain Transformer (TERT)** — a causal Transformer
policy for blind quadrupedal locomotion over multiple terrains, trained by a two-stage
offline-pretraining / online-correction scheme on top of privileged learning.

## Paper

> Hang Lai, Weinan Zhang, Xialin He, Chen Yu, Zheng Tian, Yong Yu, Jun Wang.
> *Sim-to-Real Transfer for Quadrupedal Locomotion via Terrain Transformer.*
> IEEE International Conference on Robotics and Automation (ICRA), 2023.
> [arXiv:2212.07740](https://arxiv.org/abs/2212.07740) ·
> [project page](https://terrain-transformer.github.io/)

Original code is distributed by the authors as a Dropbox archive from the project page.
This repository is **not** affiliated with the authors. Licensing of the official release
and what may legally be reused is analysed in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Motivation

The standard privileged-learning recipe trains a teacher on information the robot cannot
sense — a terrain heightmap, contact forces, true friction and mass — compresses it into a
latent `l_t`, then trains a student to *regress that latent* from proprioception history.
The student inherits a brittleness: it is only as good as its estimate of `l_t`, and
degrades sharply when `l_t` leaves the training distribution.

TERT removes the latent from the deployed policy. A causal Transformer maps the recent
observation-action history directly to the teacher's action, so there is no intermediate
quantity to mis-estimate. The paper reports that this is what lets the policy cross sand
pits and descend stairs, where an RMA-style student fails outright.

## Method

```
teacher (privileged)                    TERT (deployable)
────────────────────                    ─────────────────
e_t = [heightmap 187 | contact 12 | params 4]
  │ MLP encoder μ
  ▼
l_t ∈ R¹²  ─┐
            ├─→ MLP [512,256,128] → ā_t        (o_{t-19..t}, a_{t-19..t})
o_t ∈ R⁴⁸ ─┘        trained by PPO                     │ causal Transformer
                                                        │ 3 blocks, d=256, T=20
                                                        ▼
                                                       â_t ∈ R¹²
                                                        │ PD: τ = kp(q_d−q) + kd(q̇_d−q̇)
                                                        ▼
                                                     12 joints @ 50 Hz
```

**Stage 1 — offline pretraining.** The teacher rolls out across the terrain curriculum.
The Transformer is fit by masked MSE to the teacher's actions, conditioned on the
*teacher's* observation-action sequences.

**Stage 2 — online correction.** The Transformer now drives the robot while the teacher
labels each state it actually visits. This is DAgger: it repairs the input-distribution
shift that stage 1 alone cannot, because a policy trained only on teacher trajectories has
never seen its own mistakes.

Neither stage alone suffices — the paper ablates both.

## Architecture notes

Tokens are interleaved `(o_1, a_1, …, o_T, a_T)`, so the context is 2T = 40 tokens for
T = 20 timesteps. The action head reads observation-token positions, which under causal
masking makes `â_t` a function of `o_1, a_1, …, o_t` and nothing later. Returns-to-go,
present in Decision Transformer, are deliberately dropped. Temporal position is a *learned*
embedding added to both observation and action tokens.

Attention weights are inspectable (`return_weights=True`) to reproduce the paper's
observation that attention is roughly periodic with the gait cycle and does not decay
toward recent timesteps.

## Repository structure

```
tert/                 TERT CORE — the paper
  envs/               observation spec, env wrapper, terrain, rewards
  models/             transformer, teacher, privileged encoder, baselines
  data/               trajectory buffer, dataset, normaliser
  training/           PPO teacher, offline pretraining, online correction
  eval/               metrics, rollout, ablations
  backends/           simulator bindings (Isaac Gym first)
robotics/             ENGINEERING EXTENSION — estimation, control, perception, transforms
cpp/                  ENGINEERING EXTENSION — TorchScript inference runner
ros2_ws/              ENGINEERING EXTENSION — nodes, launch, TF2
docs/                 code_map.md, math.md, status.md, architecture.md, sim_to_real.md
configs/  scripts/  tests/  results/
```

Everything under `tert/` tracks the paper. Everything under `robotics/`, `cpp/`, and
`ros2_ws/` is an extension that the paper does not contain, and is labelled as such in
`docs/status.md`.

## Installation

Target runtime is **WSL2 + Isaac Gym Preview 4** with an NVIDIA GPU. Isaac Gym is
Linux-only, proprietary, and must be obtained from NVIDIA — it cannot be vendored here.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Then install Isaac Gym Preview 4 from <https://developer.nvidia.com/isaac-gym> into the
same environment (`cd isaacgym/python && pip install -e .`).

The model code (`tert/models/`) and its tests need only PyTorch, and run anywhere.

```bash
pytest
```

## Training

Not yet implemented — see [`docs/status.md`](docs/status.md). The intended pipeline
mirrors the paper: train the teacher with PPO, collect teacher rollouts, pretrain the
Transformer offline, then run online correction against the teacher as labeller.

## Evaluation

Planned metrics: return, terrain traversal success, fall rate, velocity tracking error,
control smoothness `Σ|a_t − a_{t−1}|`, energy `Σ|τ · q̇|`, and inference latency — with
ablations over history length, privileged training, and randomisation level.

No metric in this repository has been measured. Results will only appear under `results/`
once the scripts have actually been run.

## Limitations

- Isaac Gym Preview is deprecated in favour of Isaac Lab; the paper's exact environment is
  a moving target.
- The paper omits the attention head count, dataset size, and several training details that
  had to be recovered from the official code. These are flagged in `docs/code_map.md`.
- The paper's real-robot results depend on hardware not modelled here; the deployment layer
  is an extension, not a reproduction.

## Future work

Adaptive history length; sensor-dropout and degradation training; vision- or LiDAR-based
terrain representation replacing the privileged heightmap; uncertainty-aware policies; and
a study of what transfers to marine vehicles (AUV/ASV terrain following and disturbance
rejection), where the partial-observability structure is analogous but the contact
dynamics that motivate the gait-periodic attention pattern are absent.

## License

MIT for original code — see [`LICENSE`](LICENSE). Third-party components retain their own
licenses; see [`THIRD_PARTY.md`](THIRD_PARTY.md). If you use this work, cite the original
paper.
