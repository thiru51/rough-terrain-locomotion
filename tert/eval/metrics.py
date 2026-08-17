"""Metrics for comparing locomotion policies.

Everything is accumulated **only up to each environment's first termination**.
Environments auto-reset, so continuing past a fall would silently average a
failed episode together with a fresh one and make a policy that falls constantly
look average. The mask is the difference between a meaningful number and a
meaningless one.

Reported quantities:

    return       discounted-free sum of reward over the episode
    smoothness   mean |a_t - a_{t-1}|, the chatter a real actuator pays for
    energy       mean |tau . qdot|, mechanical power drawn
    tracking     mean |v_xy - cmd_xy|, whether it actually goes where told
    fall rate    fraction of environments that terminated for any reason but timeout
    length       steps survived
"""

from dataclasses import dataclass, field

import torch


@dataclass
class EvalMetrics:
    episode_return: torch.Tensor
    episode_length: torch.Tensor
    smoothness: torch.Tensor
    energy: torch.Tensor
    tracking_error: torch.Tensor
    fell: torch.Tensor
    inference_ms: float = 0.0

    def summary(self) -> dict[str, float]:
        """Mean and standard deviation across environments, as plain floats."""
        out = {}
        for name in ("episode_return", "episode_length", "smoothness", "energy", "tracking_error"):
            values = getattr(self, name).float()
            out[name] = values.mean().item()
            out[f"{name}_std"] = values.std().item() if values.numel() > 1 else 0.0
        out["fall_rate"] = self.fell.float().mean().item()
        out["inference_ms"] = self.inference_ms
        return out


@dataclass
class MetricAccumulator:
    """Per-environment running totals, frozen at each environment's first done."""

    num_envs: int
    device: torch.device | str = "cpu"
    _alive: torch.Tensor = field(init=False)

    def __post_init__(self):
        z = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self.episode_return, self.episode_length = z(), z()
        self.smoothness, self.energy, self.tracking_error = z(), z(), z()
        self.fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_actions = None

    @property
    def finished(self) -> bool:
        return not bool(self._alive.any())

    def step(
        self,
        actions,
        rewards,
        dones,
        timeouts=None,
        torques=None,
        dof_vel=None,
        base_lin_vel=None,
        commands=None,
    ):
        alive = self._alive.float()

        if self._last_actions is None:
            self._last_actions = torch.zeros_like(actions)
        self.smoothness += (actions - self._last_actions).abs().sum(-1) * alive
        self._last_actions = actions.clone()

        self.episode_return += rewards * alive
        self.episode_length += alive

        if torques is not None and dof_vel is not None:
            self.energy += (torques * dof_vel).abs().sum(-1) * alive
        if base_lin_vel is not None and commands is not None:
            error = (base_lin_vel[:, :2] - commands[:, :2]).norm(dim=-1)
            self.tracking_error += error * alive

        # A timeout is not a fall; only unplanned terminations count against a policy.
        failure = dones if timeouts is None else (dones & ~timeouts)
        self.fell |= failure & self._alive
        self._alive &= ~dones

    def result(self, inference_ms: float = 0.0) -> EvalMetrics:
        steps = self.episode_length.clamp_min(1)
        return EvalMetrics(
            episode_return=self.episode_return,
            episode_length=self.episode_length,
            smoothness=self.smoothness / steps,
            energy=self.energy / steps,
            tracking_error=self.tracking_error / steps,
            fell=self.fell,
            inference_ms=inference_ms,
        )
