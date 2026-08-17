"""Training loop for the privileged teacher.

Collect `num_steps_per_env` transitions across all environments, then run one
PPO update. Full training is ~20k such iterations at 4096 environments.
"""

from dataclasses import dataclass

import torch

from tert.training.ppo import PPO, PPOConfig, RolloutStorage


@dataclass
class TeacherTrainConfig:
    num_steps_per_env: int = 24
    max_iterations: int = 20_000
    save_interval: int = 2_000
    ppo: PPOConfig = None

    def __post_init__(self):
        if self.ppo is None:
            self.ppo = PPOConfig()


def collect_rollout(env, policy, storage, obs, num_steps):
    """Step the env under the stochastic policy, recording what PPO needs."""
    for _ in range(num_steps):
        with torch.no_grad():
            actions = policy.act(obs)
            storage_row = dict(
                values=policy.evaluate(obs),
                log_probs=policy.action_log_prob(actions),
                mu=policy.action_mean,
                sigma=policy.action_std,
            )
        next_obs, rewards, dones, infos = env.step(actions)
        storage.add(obs, actions, rewards, dones, timeouts=infos.get("time_outs"), **storage_row)
        obs = next_obs
    return obs


def train_teacher(env, policy, cfg: TeacherTrainConfig, device="cpu", on_iteration=None):
    ppo = PPO(policy, cfg.ppo, device)
    obs = env.reset()

    for iteration in range(cfg.max_iterations):
        storage = RolloutStorage(
            cfg.num_steps_per_env,
            env.num_envs,
            env.num_obs,
            env.num_actions,
            gamma=cfg.ppo.gamma,
            device=device,
        )
        obs = collect_rollout(env, policy, storage, obs, cfg.num_steps_per_env)

        with torch.no_grad():
            last_values = policy.evaluate(obs)
        storage.compute_returns(last_values, cfg.ppo.gamma, cfg.ppo.lam)

        stats = ppo.update(storage)
        if on_iteration is not None:
            on_iteration(iteration, stats)

    return policy
