"""Shared Gaussian actor-critic plumbing.

The teacher and the PPO-trained baselines differ only in how they turn an
observation into a feature vector. Everything downstream — the Gaussian head,
the state-independent log-std, the distribution bookkeeping PPO reads — is
identical, so subclasses override `_features` and nothing else.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(in_dim, hidden_dims, out_dim, activation=nn.ELU):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), activation()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class GaussianActorCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_actions: int,
        actor_hidden=(512, 256, 128),
        critic_hidden=(512, 256, 128),
        init_noise_std: float = 1.0,
    ):
        super().__init__()
        self.actor = mlp(feature_dim, actor_hidden, num_actions)
        self.critic = mlp(feature_dim, critic_hidden, 1)
        self.log_std = nn.Parameter(torch.full((num_actions,), init_noise_std).log())
        self.distribution: Normal | None = None

    def _features(self, obs):
        return obs

    def act(self, obs):
        mean = self.actor(self._features(obs))
        self.distribution = Normal(mean, self.log_std.exp().expand_as(mean))
        return self.distribution.sample()

    def act_inference(self, obs):
        """Deterministic action — the mean, with no exploration noise."""
        return self.actor(self._features(obs))

    def evaluate(self, obs):
        return self.critic(self._features(obs)).squeeze(-1)

    def action_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        """Hook for recurrent policies; stateless ones have nothing to clear."""

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
