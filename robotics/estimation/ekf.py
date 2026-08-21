"""Error-state EKF for quadruped base state.

Estimates what the policy is handed but no sensor measures — `base_lin_vel` —
by fusing IMU propagation with leg odometry.

**Why error-state.** Orientation lives on a manifold; a quaternion has four
numbers and three degrees of freedom. Carrying it directly in the filter state
means the covariance is singular and normalisation fights the update. Instead
the nominal state holds the full quaternion, the filter tracks a small
*rotation-vector error* about it, and after each update the error is folded back
in and reset to zero. The error stays near zero, so the linearisation the EKF
depends on is valid where it is actually evaluated.

    nominal   p (3)   v (3)   q (4)   b_a (3)   b_w (3)
    error    dp (3)  dv (3)  dtheta (3)  db_a (3)  db_w (3)     -> 15 x 15 covariance

**What is observable.** Roll and pitch are observable — gravity is a persistent
reference. Yaw is not, from IMU and legs alone; it drifts, and no amount of
tuning fixes that. Absolute position is likewise unobservable without an
external reference. Velocity is observable *only while feet are planted and not
slipping*, which is precisely the assumption that fails on the hard terrain. The
filter cannot manufacture the information; the honest thing is to report the
growing covariance, which `velocity_uncertainty` exposes.

ENGINEERING EXTENSION — not part of the locomotion method.
"""

from dataclasses import dataclass, field

import numpy as np

from robotics.transforms.rotations import (
    quat_from_rotvec,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    skew,
)

P, V, TH, BA, BW = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)


@dataclass
class EKFConfig:
    accel_noise: float = 0.1  # m/s^2/sqrt(Hz)
    gyro_noise: float = 0.01  # rad/s/sqrt(Hz)
    accel_bias_walk: float = 1e-3
    gyro_bias_walk: float = 1e-4
    gravity: float = 9.81
    # Gate on normalised innovation squared. 3 DoF, ~99% -> 11.3. Slip produces
    # a large, consistent innovation, and swallowing it biases velocity exactly
    # when the policy is most sensitive to it.
    chi2_gate: float = 11.3


@dataclass
class BaseState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self):
        return BaseState(
            self.position.copy(),
            self.velocity.copy(),
            self.orientation.copy(),
            self.accel_bias.copy(),
            self.gyro_bias.copy(),
        )


class ErrorStateEKF:
    def __init__(self, cfg: EKFConfig | None = None, initial_covariance: float = 1e-2):
        self.cfg = cfg or EKFConfig()
        self.state = BaseState()
        self.covariance = np.eye(15) * initial_covariance
        self.rejected_updates = 0

    # --- prediction ---------------------------------------------------------

    def predict(self, accel_meas, gyro_meas, dt):
        """Propagate on raw IMU. Bias-corrected, gravity removed, rotated to world."""
        cfg, s = self.cfg, self.state
        accel = np.asarray(accel_meas) - s.accel_bias
        gyro = np.asarray(gyro_meas) - s.gyro_bias

        R = quat_to_matrix(s.orientation)
        accel_world = R @ accel + np.array([0.0, 0.0, -cfg.gravity])

        s.position = s.position + s.velocity * dt + 0.5 * accel_world * dt**2
        s.velocity = s.velocity + accel_world * dt
        s.orientation = quat_normalize(quat_multiply(s.orientation, quat_from_rotvec(gyro * dt)))

        F = np.eye(15)
        F[P, V] = np.eye(3) * dt
        F[V, TH] = -R @ skew(accel) * dt  # attitude error tilts gravity into velocity
        F[V, BA] = -R * dt
        F[TH, TH] = quat_to_matrix(quat_from_rotvec(gyro * dt)).T
        F[TH, BW] = -np.eye(3) * dt

        Q = np.zeros((15, 15))
        Q[V, V] = np.eye(3) * (cfg.accel_noise**2 * dt)
        Q[TH, TH] = np.eye(3) * (cfg.gyro_noise**2 * dt)
        Q[BA, BA] = np.eye(3) * (cfg.accel_bias_walk**2 * dt)
        Q[BW, BW] = np.eye(3) * (cfg.gyro_bias_walk**2 * dt)

        self.covariance = F @ self.covariance @ F.T + Q
        return s

    # --- correction ---------------------------------------------------------

    def update_velocity(self, velocity_body, covariance):
        """Fuse a body-frame velocity measurement, typically from leg odometry.

        Returns True if accepted, False if the chi-squared gate rejected it.
        """
        R_wb = quat_to_matrix(self.state.orientation)
        predicted = R_wb.T @ self.state.velocity

        H = np.zeros((3, 15))
        H[:, V] = R_wb.T
        # Rotating world velocity into the body frame makes the measurement
        # depend on attitude error too; dropping this term is a common bug that
        # shows up as velocity error correlated with body pitch.
        H[:, TH] = skew(R_wb.T @ self.state.velocity)

        innovation = np.asarray(velocity_body) - predicted
        S = H @ self.covariance @ H.T + covariance

        if float(innovation @ np.linalg.solve(S, innovation)) > self.cfg.chi2_gate:
            self.rejected_updates += 1
            return False

        K = self.covariance @ H.T @ np.linalg.inv(S)
        self._apply_correction(K @ innovation)

        # Joseph form: stays symmetric positive-definite under finite precision,
        # where the textbook (I - KH)P does not over a long run.
        IKH = np.eye(15) - K @ H
        self.covariance = IKH @ self.covariance @ IKH.T + K @ covariance @ K.T
        return True

    def _apply_correction(self, error):
        """Fold the error state into the nominal state, then reset it to zero."""
        s = self.state
        s.position += error[P]
        s.velocity += error[V]
        s.orientation = quat_normalize(quat_multiply(s.orientation, quat_from_rotvec(error[TH])))
        s.accel_bias += error[BA]
        s.gyro_bias += error[BW]

    # --- reporting ----------------------------------------------------------

    @property
    def velocity_uncertainty(self) -> float:
        """1-sigma velocity magnitude. Grows during flight phases and on slip."""
        return float(np.sqrt(np.trace(self.covariance[V, V])))

    def observation_velocity(self) -> np.ndarray:
        """Body-frame linear velocity, in the form the policy expects."""
        return quat_to_matrix(self.state.orientation).T @ self.state.velocity
