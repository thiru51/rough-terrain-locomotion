"""Vectorised env protocol, matching legged_gym's signature.

`step` returns the full `proprio | privileged` vector; slicing off the
deployable part is the caller's job.
"""

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class VecEnv(Protocol):
    num_envs: int
    num_obs: int
    num_actions: int
    device: torch.device
    dt: float
    max_episode_length: int

    def reset(self) -> torch.Tensor:
        """Reset every environment; returns obs (num_envs, num_obs)."""
        ...

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """actions (num_envs, num_actions) -> obs, rewards, dones, infos."""
        ...
