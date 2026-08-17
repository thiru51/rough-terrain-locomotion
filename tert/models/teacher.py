"""Teacher policy with privileged access (paper Eq. 4-5), trained by PPO.

Observations arrive as `proprio | privileged`; the actor consumes
`proprio | mu(privileged)`. The critic is symmetric — it sees the same encoded
latent rather than the raw privileged block, matching the official release.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from tert.envs.obs_spec import A1, ObsSpec
from tert.models.encoder import PrivilegedEncoder


def mlp(in_dim, hidden_dims, out_dim, activation=nn.ELU):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), activation()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TeacherActorCritic(nn.Module):
    def __init__(
        self,
        spec: ObsSpec = A1,
        latent_dim: int = 12,
        actor_hidden=(512, 256, 128),
        critic_hidden=(512, 256, 128),
        init_noise_std: float = 1.0,
    ):
        super().__init__()
        self.spec = spec
        self.encoder = PrivilegedEncoder(spec, latent_dim)

        in_dim = spec.proprio_dim + latent_dim
        self.actor = mlp(in_dim, actor_hidden, spec.num_actions)
        self.critic = mlp(in_dim, critic_hidden, 1)

        # State-independent exploration noise, as in rsl_rl.
        self.log_std = nn.Parameter(torch.full((spec.num_actions,), init_noise_std).log())
        self.distribution: Normal | None = None

    def _features(self, obs):
        proprio, privileged = obs[:, : self.spec.proprio_dim], obs[:, self.spec.proprio_dim :]
        return torch.cat([proprio, self.encoder(privileged)], dim=-1)

    def act(self, obs):
        mean = self.actor(self._features(obs))
        self.distribution = Normal(mean, self.log_std.exp().expand_as(mean))
        return self.distribution.sample()

    def act_inference(self, obs):
        """Deterministic action — used to label data in both TERT training stages."""
        return self.actor(self._features(obs))

    def evaluate(self, obs):
        return self.critic(self._features(obs)).squeeze(-1)

    def action_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
