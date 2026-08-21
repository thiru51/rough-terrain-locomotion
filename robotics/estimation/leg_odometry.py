"""Base velocity from planted feet, assuming no slip.

The assumption fails on loose ground, so feet are fused individually and
disagreement between them is used to detect slip.
"""

from dataclasses import dataclass

import numpy as np

from robotics.transforms.rotations import skew


@dataclass
class LegOdometryConfig:
    base_noise: float = 0.05  # m/s, per-foot measurement std on rigid ground
    slip_threshold: float = 0.25  # m/s, spread above which feet are disagreeing
    min_contacts: int = 1
    inflate_on_disagreement: float = 20.0  # covariance multiplier when feet disagree


class LegOdometry:
    def __init__(self, cfg: LegOdometryConfig | None = None):
        self.cfg = cfg or LegOdometryConfig()

    def per_foot_velocity(self, foot_pos_body, foot_vel_body, angular_velocity):
        """Base velocity implied by each foot independently. Shape (num_feet, 3)."""
        omega_cross = skew(angular_velocity)
        return -(np.asarray(foot_vel_body) + (omega_cross @ np.asarray(foot_pos_body).T).T)

    def measure(self, foot_pos_body, foot_vel_body, angular_velocity, contacts):
        """Fused base velocity and its covariance.

        Returns `(velocity, covariance, info)`, or `(None, None, info)` when too
        few feet are planted to say anything — a flight phase is not a failure,
        and the filter should coast on IMU rather than be fed a fabricated zero.
        """
        contacts = np.asarray(contacts, dtype=bool)
        info = {"num_contacts": int(contacts.sum()), "slipping": False, "spread": 0.0}

        if contacts.sum() < self.cfg.min_contacts:
            return None, None, info

        candidates = self.per_foot_velocity(foot_pos_body, foot_vel_body, angular_velocity)
        planted = candidates[contacts]

        # Spread across planted feet: agreement means the no-slip assumption holds.
        spread = float(np.max(np.linalg.norm(planted - planted.mean(axis=0), axis=1)))
        info["spread"] = spread

        noise = self.cfg.base_noise**2 / max(len(planted), 1)
        if len(planted) > 1 and spread > self.cfg.slip_threshold:
            info["slipping"] = True
            noise *= self.cfg.inflate_on_disagreement

        return planted.mean(axis=0), np.eye(3) * noise, info


def contacts_from_force(contact_forces, threshold: float = 5.0):
    """Boolean contact per foot from vertical ground reaction force.

    A force threshold rather than a foot-height test: height needs a terrain map,
    which is the thing the robot does not have.
    """
    return np.asarray(contact_forces)[:, 2] > threshold
