"""Privileged encoder mu: e_t -> l_t (paper Eq. 4).

The three privileged modalities are embedded separately before fusion, so the
187-dim heightmap cannot swamp the 12-dim contact forces and 4 physics
parameters in a single input projection.
"""

import torch
import torch.nn as nn

from tert.envs.obs_spec import A1, ObsSpec


class PrivilegedEncoder(nn.Module):
    def __init__(self, spec: ObsSpec = A1, latent_dim: int = 12, hidden_dim: int = 256):
        super().__init__()
        self.spec = spec
        self.slices = spec.slices()

        self.embed_height = nn.Linear(spec.heightmap_dim, 128)
        self.embed_contact = nn.Linear(spec.contact_force, 64)
        self.embed_params = nn.Linear(spec.env_params, 64)

        self.trunk = nn.Sequential(
            nn.ELU(),
            nn.Linear(128 + 64 + 64, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, privileged_obs):
        """privileged_obs (B, 203) -> latent (B, latent_dim)."""
        s = self.slices
        x = torch.cat(
            [
                self.embed_height(privileged_obs[:, s["heightmap"]]),
                self.embed_contact(privileged_obs[:, s["contact_force"]]),
                self.embed_params(privileged_obs[:, s["env_params"]]),
            ],
            dim=-1,
        )
        return self.trunk(x)
