"""Reward terms for the multi-terrain locomotion task.

Terms are pure functions of a `RobotState` snapshot, so they can be read, unit
tested, and reweighted without a simulator in the loop. The environment's job is
to fill the snapshot; composing and scaling happens here.

The set follows legged_gym, plus two terms aimed at sim-to-real: a penalty on
action magnitude and one on torque rate. Both suppress the high-frequency
chatter a policy can exploit in simulation and that destroys real actuators.

Scales are supplied by config, not hard-coded — see `configs/robot/a1.yaml`.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class RobotState:
    """One control step, batched over environments."""

    base_lin_vel: torch.Tensor  # (N, 3) body frame
    base_ang_vel: torch.Tensor  # (N, 3) body frame
    projected_gravity: torch.Tensor  # (N, 3) gravity in body frame
    base_height: torch.Tensor  # (N,)
    commands: torch.Tensor  # (N, 3) lin_vel_x, lin_vel_y, ang_vel_yaw

    dof_pos: torch.Tensor  # (N, 12)
    dof_vel: torch.Tensor  # (N, 12)
    last_dof_vel: torch.Tensor  # (N, 12)
    torques: torch.Tensor  # (N, 12)
    last_torques: torch.Tensor  # (N, 12)

    actions: torch.Tensor  # (N, 12)
    last_actions: torch.Tensor  # (N, 12)

    feet_air_time: torch.Tensor  # (N, 4) seconds since last touchdown
    first_contact: torch.Tensor  # (N, 4) bool, touchdown this step
    contact_forces: torch.Tensor  # (N, num_bodies, 3)
    penalised_contact: torch.Tensor  # (N, num_penalised, 3)

    reset: torch.Tensor  # (N,) bool
    timeout: torch.Tensor  # (N,) bool
    dt: float

    dof_pos_soft_limits: tuple[torch.Tensor, torch.Tensor] | None = None
    tracking_sigma: float = 0.25
    base_height_target: float = 0.25


def _sq(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(x), dim=1)


# --- task terms -------------------------------------------------------------


def tracking_lin_vel(s: RobotState) -> torch.Tensor:
    error = _sq(s.commands[:, :2] - s.base_lin_vel[:, :2])
    return torch.exp(-error / s.tracking_sigma)


def tracking_ang_vel(s: RobotState) -> torch.Tensor:
    error = torch.square(s.commands[:, 2] - s.base_ang_vel[:, 2])
    return torch.exp(-error / s.tracking_sigma)


def feet_air_time(s: RobotState) -> torch.Tensor:
    """Reward for swing duration, credited at touchdown.

    Encourages long strides rather than the high-frequency shuffle that
    velocity tracking alone admits. Gated on a nonzero command so standing
    still is not rewarded for holding a foot up.
    """
    reward = torch.sum((s.feet_air_time - 0.5) * s.first_contact, dim=1)
    return reward * (torch.norm(s.commands[:, :2], dim=1) > 0.1)


# --- stability penalties ----------------------------------------------------


def lin_vel_z(s: RobotState) -> torch.Tensor:
    return torch.square(s.base_lin_vel[:, 2])


def ang_vel_xy(s: RobotState) -> torch.Tensor:
    return _sq(s.base_ang_vel[:, :2])


def orientation(s: RobotState) -> torch.Tensor:
    return _sq(s.projected_gravity[:, :2])


def base_height(s: RobotState) -> torch.Tensor:
    return torch.square(s.base_height - s.base_height_target)


# --- effort and smoothness --------------------------------------------------


def torques(s: RobotState) -> torch.Tensor:
    return _sq(s.torques)


def torques_smooth(s: RobotState) -> torch.Tensor:
    """Penalise torque rate, not just magnitude."""
    return _sq(s.last_torques - s.torques)


def dof_vel(s: RobotState) -> torch.Tensor:
    return _sq(s.dof_vel)


def dof_acc(s: RobotState) -> torch.Tensor:
    return _sq((s.last_dof_vel - s.dof_vel) / s.dt)


def action_rate(s: RobotState) -> torch.Tensor:
    return _sq(s.last_actions - s.actions)


def action_magnitude(s: RobotState) -> torch.Tensor:
    """Keep targets near the default pose.

    Actions are offsets from the nominal joint angles, so this pulls the gait
    toward the pose the hardware is designed around.
    """
    return _sq(s.actions)


# --- safety -----------------------------------------------------------------


def collision(s: RobotState) -> torch.Tensor:
    return torch.sum(torch.norm(s.penalised_contact, dim=-1) > 0.1, dim=1).float()


def termination(s: RobotState) -> torch.Tensor:
    """Failure only — a timeout is not a fall and must not be penalised."""
    return (s.reset & ~s.timeout).float()


def dof_pos_limits(s: RobotState) -> torch.Tensor:
    if s.dof_pos_soft_limits is None:
        return torch.zeros_like(s.dof_pos[:, 0])
    lower, upper = s.dof_pos_soft_limits
    out_of_range = (lower - s.dof_pos).clip(min=0.0) + (s.dof_pos - upper).clip(min=0.0)
    return torch.sum(out_of_range, dim=1)


TERMS = {
    "tracking_lin_vel": tracking_lin_vel,
    "tracking_ang_vel": tracking_ang_vel,
    "feet_air_time": feet_air_time,
    "lin_vel_z": lin_vel_z,
    "ang_vel_xy": ang_vel_xy,
    "orientation": orientation,
    "base_height": base_height,
    "torques": torques,
    "torques_smooth": torques_smooth,
    "dof_vel": dof_vel,
    "dof_acc": dof_acc,
    "action_rate": action_rate,
    "action_magnitude": action_magnitude,
    "collision": collision,
    "termination": termination,
    "dof_pos_limits": dof_pos_limits,
}


@dataclass
class RewardComposer:
    """Weighted sum of the enabled terms.

    Scales are multiplied by `dt` at construction, matching legged_gym, so that
    reward magnitudes are invariant to control frequency. `only_positive` clips
    the total at zero: early in training the penalties dominate, and a policy
    that can accumulate negative return learns to terminate itself.
    """

    scales: dict[str, float]
    dt: float
    only_positive: bool = True
    _active: dict = field(init=False)

    def __post_init__(self):
        unknown = set(self.scales) - set(TERMS)
        if unknown:
            raise KeyError(f"unknown reward terms: {sorted(unknown)}")
        self._active = {
            name: (TERMS[name], scale * self.dt)
            for name, scale in self.scales.items()
            if scale != 0.0
        }

    def __call__(self, state: RobotState) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts = {name: fn(state) * scale for name, (fn, scale) in self._active.items()}
        total = torch.stack(list(parts.values())).sum(0)
        if self.only_positive:
            total = total.clip(min=0.0)
        return total, parts
