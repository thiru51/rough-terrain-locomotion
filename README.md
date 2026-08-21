# Rough Terrain Locomotion

A transformer policy that walks a quadruped over slopes, stairs and broken ground using
**proprioception only** — no camera, no lidar, no terrain map at runtime. The robot feels
the ground through its joints and infers what it is standing on from the last 20 control
steps.

Side project, in progress. The model, training pipeline, terrain generation and evaluation
harness are written and unit tested. **Nothing has been trained yet** — see
[`docs/status.md`](docs/status.md) for exactly what exists and what doesn't.

```
sensors → 20-step history → causal transformer → joint targets → PD → 12 motors @ 50 Hz
```

## Why this is hard

A blind quadruped on stairs is a partial observability problem. From a single timestep you
cannot tell a descending step from a compliant surface — both read as "the foot went lower
than expected". You need history, and you need to weigh it selectively: what mattered 300 ms
ago depends on where you are in the gait cycle.

That's a sequence modelling problem, which is why there's a transformer in here rather than
the usual MLP.

## How it works

Training happens in simulation, where you can cheat. A **teacher** policy is given
information the real robot will never have — a 17×11 height map around its body, per-foot
contact forces, and the true friction, mass and actuator gains. It compresses that into a
12-dim latent and learns to walk with PPO. This is easy, because it can see.

The **deployable policy** never gets any of that. It's a causal transformer over interleaved
observation-action tokens that predicts what the teacher would have done, given only the
recent proprioceptive history.

Training it happens in two passes:

1. **Offline** — the teacher drives, and the transformer learns to copy it.
2. **Online correction** — the transformer drives while the teacher labels every state it
   actually reaches.

The second pass is the one that matters. A policy trained only on the teacher's trajectories
has never seen its own mistakes, so the moment it drifts even slightly it's off-distribution
and compounds the error. Letting it drive and asking the teacher "what should I have done
here?" is what fixes that.

The usual approach instead trains a student to *estimate the teacher's latent* from history,
then feeds that to the teacher's frozen actor. I implemented that too, as a baseline — it's
brittle for a structural reason: the actor is downstream of the estimate and can't recover
from a bad one. Predicting the action end-to-end removes that bottleneck entirely.

## What's in here

```
tert/
  envs/        observation layout, reward terms, terrain curriculum
  models/      transformer, teacher, privileged encoder
    baselines/ TCN latent-estimator, stacked-history MLP
  training/    PPO, rollout collection, the two-pass imitation loop
  backends/    simulator interface (Isaac Gym adapter pending)
  eval/        metrics and ablations
robotics/      state estimation, control, transforms   [planned]
cpp/           TorchScript inference runner            [planned]
ros2_ws/       nodes, launch, TF2                      [planned]
```

Some design notes worth reading if you're poking around:

- **`tert/data/context.py`** — the rolling window is shared by training and deployment, so
  the sequence the policy sees on hardware is built by the same code that made its training
  inputs. It clears history per-environment on termination; carrying it across a reset means
  conditioning on a discontinuity that never happens on a real robot.
- **`tert/data/normalizer.py`** — observation statistics freeze after the first pass.
  Refitting during online correction would rescale inputs underneath an already-trained
  policy.
- **`tert/training/ppo.py`** — bootstraps value back into reward on timeout. Skip it and a
  truncated episode is indistinguishable from a fall, so the policy learns to avoid the time
  limit by terminating early. Nothing crashes; you just get a worse robot.
- **`tert/envs/terrain.py`** — terrain generation is plain NumPy heightfields rather than
  simulator calls, so it runs and tests anywhere.

## Setup

Everything except the simulator is plain PyTorch and NumPy, and runs anywhere:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

I write this on a Windows laptop and run it on a Linux workstation with a real GPU, so
nothing hard-codes a device or a path — modules take `device` and default to
`tert.device.default_device()`, which picks CUDA when it's there. The repo normalises to
LF via `.gitattributes`, so a clone on either OS is clean.

Training on the workstation additionally needs Isaac Gym Preview, which is Linux-only and
comes from NVIDIA directly. That's a lab-side setup step; none of the code here depends on
it being installed.

## Status

61 tests pass. The whole pipeline runs end to end against a stub environment; what's missing
is the adapter that fills observations from real physics.

No performance number appears anywhere in this repo, because none has been measured. When
training runs, results land in `results/`.

## Where it's going

- Isaac Gym backend, then a first teacher training run
- EKF/UKF base-state estimation from IMU and encoders
- C++ inference node and ROS2 integration
- Sensor dropout and degradation during training
- Whether any of this transfers to underwater vehicles, where the partial-observability
  structure is similar but contact dynamics — the thing driving the gait-periodic attention
  pattern — simply don't exist

## Credits

The method is Terrain Transformer, from [Lai et al., ICRA
2023](https://arxiv.org/abs/2212.07740). This is my own implementation of it; the
simulation environment builds on
[legged_gym](https://github.com/leggedrobotics/legged_gym) (BSD-3-Clause). Full attribution
and licensing in [`THIRD_PARTY.md`](THIRD_PARTY.md).

MIT licensed — see [`LICENSE`](LICENSE).
