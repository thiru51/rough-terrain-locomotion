"""Teacher policy: proprioception plus encoded privileged obs, trained by PPO."""

import torch

from tert.envs.obs_spec import A1, ObsSpec
from tert.models.actor_critic import GaussianActorCritic
from tert.models.encoder import PrivilegedEncoder


class TeacherActorCritic(GaussianActorCritic):
    def __init__(
        self,
        spec: ObsSpec = A1,
        latent_dim: int = 12,
        actor_hidden=(512, 256, 128),
        critic_hidden=(512, 256, 128),
        init_noise_std: float = 1.0,
    ):
        super().__init__(
            feature_dim=spec.proprio_dim + latent_dim,
            num_actions=spec.num_actions,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
            init_noise_std=init_noise_std,
        )
        self.spec = spec
        self.encoder = PrivilegedEncoder(spec, latent_dim)

    def latent(self, obs):
        """The target the RMA-style baselines are trained to estimate."""
        return self.encoder(obs[:, self.spec.proprio_dim :])

    def _features(self, obs):
        return torch.cat([obs[:, : self.spec.proprio_dim], self.latent(obs)], dim=-1)
