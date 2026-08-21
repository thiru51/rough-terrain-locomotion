"""Flat and stacked-history PPO baselines. No privileged information."""

import torch

from tert.models.actor_critic import GaussianActorCritic


class ObservationStacker:
    """Rolling `(obs, action)` history flattened into a single wide observation."""

    def __init__(self, num_envs, obs_dim, act_dim, history_length, device="cpu"):
        self.history_length = history_length
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.obs = torch.zeros(num_envs, history_length, obs_dim, device=device)
        self.actions = torch.zeros(num_envs, history_length, act_dim, device=device)

    @property
    def stacked_dim(self) -> int:
        return self.history_length * (self.obs_dim + self.act_dim)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        elif env_ids.numel() == 0:
            return
        self.obs[env_ids] = 0.0
        self.actions[env_ids] = 0.0

    def push(self, obs, last_action):
        self.obs = self.obs.roll(-1, dims=1)
        self.actions = self.actions.roll(-1, dims=1)
        self.obs[:, -1], self.actions[:, -1] = obs, last_action
        return self.stacked()

    def stacked(self):
        return torch.cat([self.obs, self.actions], dim=-1).flatten(1)


class StackedActorCritic(GaussianActorCritic):
    def __init__(self, obs_dim, num_actions, history_length=1, **kwargs):
        super().__init__(
            feature_dim=history_length * (obs_dim + num_actions),
            num_actions=num_actions,
            **kwargs,
        )
        self.history_length = history_length
