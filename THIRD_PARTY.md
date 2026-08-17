# Third-Party Attribution

This project implements the **Terrain Transformer (TERT)** method.
It is not affiliated with, endorsed by, or released by the original authors.

## Original work

**Paper.** Hang Lai, Weinan Zhang, Xialin He, Chen Yu, Zheng Tian, Yong Yu, Jun Wang.
*Sim-to-Real Transfer for Quadrupedal Locomotion via Terrain Transformer.* ICRA 2023.
[arXiv:2212.07740](https://arxiv.org/abs/2212.07740) ·
[project page](https://terrain-transformer.github.io/)

**Official code.** Distributed as `TERT.zip` from the project page (Dropbox), not via a
public git repository. Inspected at commit-less snapshot downloaded 2026-08-17,
15,709,857 bytes, unpacking to `TERT_code/`.

## License status of the official release

`TERT_code` contains **no license file of its own**. It is a vendored fork composed of
three layers with different terms, which determines what this repository may reuse.

| Layer | Paths in `TERT_code` | License | Reused here |
|---|---|---|---|
| legged_gym / rsl_rl fork | `legged_gym/**`, `rsl_rl/**` | BSD-3-Clause (© 2021 ETH Zurich, Nikita Rudin; © 2021 NVIDIA CORPORATION & AFFILIATES) — SPDX headers present | Planned: yes, headers retained |
| Decision Transformer | `decision_transformer/model.py`, `decision_transformer/utils.py` | Derived from [min-decision-transformer](https://github.com/nikhilbarhate99/min-decision-transformer), MIT (© Nikhil Barhate) | No code copied; architecture referenced |
| TERT-authored | `legged_gym/scripts/{train_tert,offline_pretraining_collect,online_correction_collect,eval_tert}.py`, `tcn_encoder.py` | **None stated** → all rights reserved by default | **No.** Reference only |
| A1 robot model | `resources/robots/a1/**` | MPL-2.0 (via unitree_ros) | Not vendored; referenced by path |
| Pretrained weights | `eval_model/*.pt`, `*.npy` | None stated | **Not redistributed** |

Because the TERT-specific scripts carry no license grant, this project treats them as
**readable reference material only**. The algorithms they express (two-stage training,
DAgger-style online correction, evaluation metrics) are described in the paper and are
reimplemented here from that description plus verified constants.

## Upstream dependencies

| Project | License | Role |
|---|---|---|
| [legged_gym](https://github.com/leggedrobotics/legged_gym) — Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL*, CoRL 2021 | BSD-3-Clause | Environment, terrain curriculum, reward set |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) v1.0.2 | BSD-3-Clause | PPO used to train the teacher |
| [Isaac Gym Preview](https://developer.nvidia.com/isaac-gym) | NVIDIA proprietary EULA | Simulator. **Not redistributed** — install separately |
| [min-decision-transformer](https://github.com/nikhilbarhate99/min-decision-transformer) | MIT | Causal-GPT formulation referenced by `tert/models/transformer.py` |
| Unitree A1 URDF/meshes | MPL-2.0 | Robot model |

## Component ledger

Status as of the current commit. `docs/status.md` tracks implementation progress;
this table tracks **provenance**.

| Component | Origin | This repo |
|---|---|---|
| Observation layout (48 proprio / 203 privileged) | legged_gym + TERT modifications | Independently implemented (`tert/envs/obs_spec.py`), constants verified against official source |
| Privileged encoder `mu` | Paper Eq. 4; official `actor_critic_teacher.Encoder` | Independently implemented (`tert/models/encoder.py`); same 187/12/4 head widths |
| Teacher actor-critic | Official `TeacherActorCritic` (BSD-3) | Independently implemented (`tert/models/teacher.py`); fused into a shared MLP builder |
| Causal Transformer | Paper Eq. 2-3; min-DT (MIT) | Independently implemented (`tert/models/transformer.py`); fused QKV projection, SDPA fast path, optional pre-LN, inspectable attention weights |
| Terrain curriculum | legged_gym (BSD-3), modified by TERT | Planned: adapt with headers retained |
| PPO | rsl_rl (BSD-3) | Planned: vendor with headers retained |
| Data collection / two-stage training | Paper Sec. IV-B | Planned: independent implementation |
| EKF/UKF, ROS2 nodes, C++ inference, MPC | — | Original engineering extensions, not part of the paper |

## Modifications and deliberate deviations

Recorded so that differences from the official release are auditable rather than accidental.

- **Fused QKV projection** instead of three separate `nn.Linear` layers. Mathematically
  equivalent; changes parameter layout, so official checkpoints are not loadable (they are
  not redistributable anyway).
- **`torch.nn.functional.scaled_dot_product_attention`** on the training path, with an
  explicit softmax path retained for attention-map inspection (paper Fig. 6). Both paths
  are asserted equal in `tests/test_models.py`.
- **Pre-LN switchable.** The official release uses post-LN blocks. The default here stays
  post-LN for faithfulness; `TransformerConfig.pre_ln` makes the alternative an experiment.
- **Head count.** The paper does not state the number of attention heads; the official
  release uses `n_heads=1`. This repo follows the code, and records the paper's silence.
- **Modern Python/PyTorch.** The official stack is Python 3.8 + torch 1.10+cu113; this
  repo targets Python >=3.10 + torch >=2.2.

## Citation

If you build on this, cite the original paper:

```bibtex
@inproceedings{lai2023tert,
  title     = {Sim-to-Real Transfer for Quadrupedal Locomotion via Terrain Transformer},
  author    = {Lai, Hang and Zhang, Weinan and He, Xialin and Yu, Chen and
               Tian, Zheng and Yu, Yong and Wang, Jun},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2023}
}
```
