"""Observation layout for the A1 multi-terrain task.

Mirrors the concatenation order in legged_gym's `LeggedRobot.compute_observations`.
The teacher sees `proprio | privileged` (251); TERT sees only the leading 48.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObsSpec:
    # proprioception, in concatenation order
    base_lin_vel: int = 3
    base_ang_vel: int = 3
    projected_gravity: int = 3
    commands: int = 3
    dof_pos: int = 12
    dof_vel: int = 12
    last_action: int = 12

    # privileged, in concatenation order
    height_points: tuple[int, int] = (17, 11)
    contact_force: int = 12
    env_params: int = 4  # friction, added base mass, kp, kd

    num_actions: int = 12

    @property
    def proprio_dim(self) -> int:
        return (
            self.base_lin_vel
            + self.base_ang_vel
            + self.projected_gravity
            + self.commands
            + self.dof_pos
            + self.dof_vel
            + self.last_action
        )

    @property
    def heightmap_dim(self) -> int:
        return self.height_points[0] * self.height_points[1]

    @property
    def privileged_dim(self) -> int:
        return self.heightmap_dim + self.contact_force + self.env_params

    @property
    def total_dim(self) -> int:
        return self.proprio_dim + self.privileged_dim

    def slices(self) -> dict[str, slice]:
        """Named slices into the privileged block (relative to its own start)."""
        h = self.heightmap_dim
        c = h + self.contact_force
        return {
            "heightmap": slice(0, h),
            "contact_force": slice(h, c),
            "env_params": slice(c, c + self.env_params),
        }


A1 = ObsSpec()  # 48 proprio + 203 privileged = 251
