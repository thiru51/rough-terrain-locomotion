"""Vectorised environment contract.

Written against legged_gym's signature so the Isaac Gym backend is a thin adapter,
but nothing here imports a simulator: the training code is developed and tested
against stubs and bound to a backend later.

`step` returns the full observation `proprio | privileged` (251 for A1). Slicing
the deployable part is the caller's job — that asymmetry is the whole point of
privileged learning, so it stays visible rather than hidden behind the env.
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
