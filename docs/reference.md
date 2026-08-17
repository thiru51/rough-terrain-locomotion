# Reference notes

Where every constant in this codebase came from, so none of them are mystery numbers.

Values were read out of the original TERT release rather than guessed from the paper. A few
things the paper never states — attention head count, the exact privileged breakdown — only
exist in the code, and are marked **(code only)** below. Useful when something behaves oddly
and you need to know whether a number is load-bearing or arbitrary.

## Core method

| Paper component | Official file | Class / function | Shapes and values | This repo |
|---|---|---|---|---|
| POMDP, proprioception `o_t` (Sec. V-A) | `legged_gym/envs/base/legged_robot.py` | `compute_observations` | **48** = lin_vel 3 + ang_vel 3 + projected_gravity 3 + commands 3 + (dof_pos − default) 12 + dof_vel 12 + last_action 12 | `tert/envs/obs_spec.py` |
| Privileged info `e_t` (Sec. IV-A) | same, under `cfg.env.privileged_obs` | appended to `obs_buf` | **203** = heightmap 17×11 = **187**, foot contact force 4×3 = **12**, env params **4** (friction, added base mass, kp, kd). Total obs **251** | `tert/envs/obs_spec.py` |
| Obs scaling | same | `obs_scales` | lin_vel 2.0, ang_vel 0.25, dof_pos 1.0, dof_vel 0.05, height 5.0; contact force ×0.002; mass ×0.2; (kp−55)×0.1; (kd−0.8)×10 | planned `tert/envs/legged_env.py` |
| Encoder `mu`, latent `l_t` (Eq. 4) | `rsl_rl/modules/actor_critic_teacher.py` | `Encoder` | 3 parallel heads 187→128, 12→64, 4→64 → concat 256 → ELU → 256→256 → ELU → 256→**12** | `tert/models/encoder.py` |
| Teacher `π̄(o_t, l_t)` (Eq. 5) | same | `TeacherActorCritic.act` | `obs[:, :48] ⊕ l_t` → MLP **[512, 256, 128]**, ELU → 12; `Normal(mean, std)` with state-independent std | `tert/models/teacher.py` |
| Teacher training, PPO (Sec. IV-A) | `rsl_rl/algorithms/ppo.py`, `runners/train_teacher_runner.py` | `PPO.update` | clip 0.2, 5 epochs, 4 minibatches, adaptive lr from 1e-3 with `desired_kl` 0.01, γ 0.99, λ 0.95, 24 steps/env, 4096 envs, 20k iterations, entropy coef 0.01 | planned `tert/training/ppo_teacher.py` |
| Self-attention (Eq. 2) | `decision_transformer/model.py` | `MaskedCausalAttention` | causal mask over 2T tokens, scale `1/√d_head` | `tert/models/transformer.py` |
| Transformer (Eq. 3) | same | `DecisionTransformer` | 3 blocks, embed 256, **n_heads = 1 (code only)**, dropout 0.05, context T = **20** → sequence 2T = 40; MLP 4× GELU; **post-LN** residual blocks | `tert/models/transformer.py` |
| Temporal representation | same | `embed_timestep` | learned `nn.Embedding(4096, 256)`, added to *both* observation and action embeddings (not sinusoidal) | `tert/models/transformer.py` |
| Token layout | `DecisionTransformer.forward` | stack + permute | `(o_1, a_1, …, o_T, a_T)`; action head reads observation-token positions `h[:, 0]`. Returns-to-go removed vs. Decision Transformer (Sec. III-B) | `tert/models/transformer.py` |
| Offline pretraining (Eq. 6) | `scripts/offline_pretraining_collect.py` → `scripts/train_tert.py --exp_name=offline_pretraining` | — | teacher rolls out over 2048 envs; masked MSE to teacher action; AdamW lr 1e-4, wd 1e-4, linear warmup 10k, grad-clip 0.25, batch 64 | planned `tert/training/offline_pretrain.py` |
| Online correction (Eq. 7) | `scripts/online_correction_collect.py` → `train_tert.py --exp_name=online_correction` | — | **TERT acts, teacher labels** its own states (DAgger); resumes offline checkpoint; **normalisation stats frozen from stage 1**; 10 iterations × 2048 envs, first episode per env only | planned `tert/training/online_correct.py` |
| Observation normalisation | `decision_transformer/utils.py` | `D4RLTrajectoryDataset` | per-dimension mean/std over the dataset, +1e-6; 10% of sampled windows forced to start at `t=0` | planned `tert/data/normalizer.py` |
| Action → torque (Eq. 8) | `legged_gym/envs/base/legged_robot.py` | `_compute_torques` | `τ = kp(q_d − q) + kd(q̇_d − q̇)`; A1 kp 55, kd 0.8, action_scale 0.25, decimation 4 | planned `tert/envs/legged_env.py` |
| Terrain (Sec. V-A) | `legged_gym/utils/terrain.py` | `curriculum` | 10 rows (difficulty) × 20 cols (type); 5 types [smooth slope, rough slope, stair up, stair down, discrete]; trimesh, h-scale 0.1 m, v-scale 0.005 m. TERT replaced upstream `randomized_terrain` with a deterministic grid and **reduced** stair/obstacle difficulty scaling | planned `tert/envs/terrain.py` |
| Domain randomization | `legged_robot.py` | `_process_rigid_shape_props`, `_process_rigid_body_props` | friction [0.5, 1.25], added base mass [0, 5] kg, **kp [45, 65], kd [0.70, 0.90] (TERT addition)**, push 1 m/s every 15 s. Assigned by **deterministic bucketing on `env_id % 64`**, not i.i.d. sampling | planned `tert/envs/legged_env.py` |
| Reward (Sec. V-A) | `legged_robot.py` `_reward_*` | 22 terms | legged_gym set plus TERT's two: `action_magnitude` −0.01, `torques_smooth` −0.0003. A1 overrides: `torques` −0.0002, `dof_pos_limits` −10.0 | planned `tert/envs/rewards.py` |
| Command distribution | `legged_robot_config.py`, overridden in scripts | `commands.ranges` | training `lin_vel_x` fixed at **[0.4, 0.4]** m/s, lateral and yaw zero, heading command on | planned `configs/` |
| Evaluation metrics (Sec. V-B) | `decision_transformer/utils.py` | `evaluate_on_a1` | return; smoothness `Σ|a_t − a_{t−1}|`; energy `Σ|τ · q̇|`; episode length — all masked after first termination | planned `tert/eval/metrics.py` |
| Baselines (Sec. V-B) | `rsl_rl/modules/actor_critic_student.py`, `tcn_encoder.py` | RMA-style TCN over 50 steps → `l̂_t`; plus PPO, StackedPPO, GRU | — | planned `tert/models/baselines/` |
| Ablations (Sec. V-C) | — | TERT-w/o-OC, TERT-w/o-OP, TERT-TCN, TERT-Latent | — | planned `tert/eval/ablation.py` |
| Attention analysis (Fig. 6) | — | attention weights over a 200-step stair-down trajectory | — | supported via `return_weights=True` |

## Not present in the official release

These are described in the paper but have no released implementation, or are absent
entirely. They are ours to build, and are labelled as such.

| Item | Status |
|---|---|
| Real-robot deployment interface (A1, 50 Hz, kp 55 / kd 8, base linear velocity from accelerometer — Sec. V-D) | Paper text only; no code. Engineering extension |
| Sim-to-real state estimation | Not in release. Engineering extension (`robotics/estimation/`) |
| ROS2 integration, C++ inference | Not in paper or release. Engineering extension |
