import numpy as np
import pytest

from robotics.estimation.ekf import EKFConfig, ErrorStateEKF
from robotics.estimation.leg_odometry import (
    LegOdometry,
    LegOdometryConfig,
    contacts_from_force,
)
from robotics.transforms.rotations import (
    gravity_from_quaternion,
    quat_from_rotvec,
    quat_multiply,
    quat_to_matrix,
    quat_to_rotvec,
    rotate,
    skew,
)

DT = 0.002  # 500 Hz IMU


# --- rotations --------------------------------------------------------------


def test_skew_is_the_cross_product():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_rotvec_roundtrip():
    for rotvec in [np.zeros(3), np.array([0.1, -0.2, 0.3]), np.array([0.0, 0.0, np.pi / 2])]:
        np.testing.assert_allclose(quat_to_rotvec(quat_from_rotvec(rotvec)), rotvec, atol=1e-9)


def test_rotation_matrix_is_orthonormal():
    q = quat_from_rotvec(np.array([0.3, -0.7, 1.1]))
    R = quat_to_matrix(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_quarter_turn_about_z():
    q = quat_from_rotvec(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(rotate(q, np.array([1.0, 0.0, 0.0])), [0, 1, 0], atol=1e-9)


def test_gravity_points_down_when_level_and_tilts_with_pitch():
    level = gravity_from_quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(level, [0, 0, -9.81], atol=1e-9)

    pitched = gravity_from_quaternion(quat_from_rotvec(np.array([0.0, np.pi / 2, 0.0])))
    assert pitched[0] == pytest.approx(9.81, abs=1e-6)  # gravity moves into body x


# --- leg odometry -----------------------------------------------------------


def test_planted_foot_pins_base_velocity():
    """A foot moving backward at 1 m/s in body frame means the base moves forward."""
    odom = LegOdometry()
    foot_pos = np.array([[0.2, 0.1, -0.3]] * 4)
    foot_vel = np.array([[-1.0, 0.0, 0.0]] * 4)

    velocity, cov, info = odom.measure(foot_pos, foot_vel, np.zeros(3), np.ones(4, dtype=bool))
    np.testing.assert_allclose(velocity, [1.0, 0.0, 0.0])
    assert info["num_contacts"] == 4 and not info["slipping"]
    assert cov.shape == (3, 3)


def test_angular_velocity_is_compensated():
    """Body rotation moves the feet even with the base translating nowhere."""
    odom = LegOdometry()
    omega = np.array([0.0, 0.0, 1.0])
    foot_pos = np.array([[0.3, 0.0, -0.3]])
    foot_vel = -np.cross(omega, foot_pos[0])[None, :]

    velocity, _, _ = odom.measure(foot_pos, foot_vel, omega, np.array([True]))
    np.testing.assert_allclose(velocity, np.zeros(3), atol=1e-12)


def test_flight_phase_returns_no_measurement():
    """No planted feet is not a failure — the filter should coast, not be fed a zero."""
    odom = LegOdometry()
    velocity, cov, info = odom.measure(
        np.zeros((4, 3)), np.zeros((4, 3)), np.zeros(3), np.zeros(4, dtype=bool)
    )
    assert velocity is None and cov is None and info["num_contacts"] == 0


def test_slipping_foot_inflates_covariance():
    odom = LegOdometry(LegOdometryConfig(slip_threshold=0.1))
    foot_pos = np.array([[0.2, 0.1, -0.3]] * 4)
    agreeing = np.array([[-1.0, 0.0, 0.0]] * 4)
    slipping = agreeing.copy()
    slipping[0] = [-3.0, 0.0, 0.0]  # one foot sliding

    _, clean_cov, clean = odom.measure(foot_pos, agreeing, np.zeros(3), np.ones(4, dtype=bool))
    _, slip_cov, info = odom.measure(foot_pos, slipping, np.zeros(3), np.ones(4, dtype=bool))

    assert not clean["slipping"] and info["slipping"]
    assert slip_cov[0, 0] > clean_cov[0, 0]


def test_contact_detection_uses_vertical_force():
    forces = np.array([[0, 0, 40.0], [0, 0, 0.5], [1.0, 2.0, 30.0], [0, 0, 0]])
    assert contacts_from_force(forces).tolist() == [True, False, True, False]


# --- EKF --------------------------------------------------------------------


def level_imu(accel_z=9.81):
    """What a level, stationary accelerometer reads: it senses the reaction to gravity."""
    return np.array([0.0, 0.0, accel_z]), np.zeros(3)


def test_stationary_level_robot_does_not_drift():
    ekf = ErrorStateEKF()
    accel, gyro = level_imu()
    for _ in range(500):
        ekf.predict(accel, gyro, DT)
    np.testing.assert_allclose(ekf.state.velocity, np.zeros(3), atol=1e-9)


def test_constant_acceleration_integrates_to_velocity():
    ekf = ErrorStateEKF()
    accel = np.array([2.0, 0.0, 9.81])  # 2 m/s^2 forward, holding against gravity
    for _ in range(500):  # 1 second
        ekf.predict(accel, np.zeros(3), DT)
    assert ekf.state.velocity[0] == pytest.approx(2.0, abs=1e-3)


def test_gyro_integrates_into_orientation():
    ekf = ErrorStateEKF()
    gyro = np.array([0.0, 0.0, np.pi / 2])  # quarter turn per second
    for _ in range(500):
        ekf.predict(np.array([0.0, 0.0, 9.81]), gyro, DT)
    yaw = quat_to_rotvec(ekf.state.orientation)[2]
    assert yaw == pytest.approx(np.pi / 2, abs=1e-3)


def test_covariance_grows_without_measurements():
    """Dead reckoning must advertise its own decay."""
    ekf = ErrorStateEKF()
    accel, gyro = level_imu()
    before = ekf.velocity_uncertainty
    for _ in range(500):
        ekf.predict(accel, gyro, DT)
    assert ekf.velocity_uncertainty > before


def test_velocity_update_corrects_a_biased_estimate():
    ekf = ErrorStateEKF()
    accel, gyro = level_imu()
    for _ in range(100):
        ekf.predict(accel, gyro, DT)
    ekf.state.velocity = np.array([0.5, 0.0, 0.0])  # drifted

    before = ekf.velocity_uncertainty
    assert ekf.update_velocity(np.zeros(3), np.eye(3) * 1e-4)
    assert abs(ekf.state.velocity[0]) < 0.5
    assert ekf.velocity_uncertainty < before


def test_chi2_gate_rejects_an_outlier():
    """A slipping foot produces a large consistent innovation; swallowing it biases velocity."""
    ekf = ErrorStateEKF(EKFConfig(chi2_gate=11.3))
    accel, gyro = level_imu()
    for _ in range(100):
        ekf.predict(accel, gyro, DT)

    assert not ekf.update_velocity(np.array([50.0, 0.0, 0.0]), np.eye(3) * 1e-6)
    assert ekf.rejected_updates == 1
    np.testing.assert_allclose(ekf.state.velocity, np.zeros(3), atol=1e-9)


def test_covariance_stays_symmetric_and_positive_definite():
    """Joseph form should survive many updates where (I-KH)P would degrade."""
    rng = np.random.default_rng(0)
    ekf = ErrorStateEKF()
    accel, gyro = level_imu()
    for _ in range(300):
        ekf.predict(accel + rng.normal(0, 0.02, 3), gyro, DT)
        ekf.update_velocity(rng.normal(0, 0.01, 3), np.eye(3) * 2.5e-3)

    P = ekf.covariance
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    assert np.linalg.eigvalsh(P).min() > 0


def test_accel_bias_is_estimated_from_a_stationary_robot():
    """A robot known to be still lets the filter absorb a constant accelerometer offset."""
    ekf = ErrorStateEKF()
    bias = np.array([0.3, -0.2, 0.0])
    for _ in range(4000):
        ekf.predict(np.array([0.0, 0.0, 9.81]) + bias, np.zeros(3), DT)
        ekf.update_velocity(np.zeros(3), np.eye(3) * 1e-4)

    assert np.linalg.norm(ekf.state.accel_bias - bias) < np.linalg.norm(bias)
    np.testing.assert_allclose(ekf.state.velocity, np.zeros(3), atol=1e-2)


def test_policy_observation_is_body_frame():
    ekf = ErrorStateEKF()
    ekf.state.velocity = np.array([1.0, 0.0, 0.0])  # world frame, moving along +x
    ekf.state.orientation = quat_from_rotvec(np.array([0.0, 0.0, np.pi / 2]))  # yawed 90 deg
    np.testing.assert_allclose(ekf.observation_velocity(), [0.0, -1.0, 0.0], atol=1e-9)


def test_end_to_end_imu_plus_legs():
    """The full path: IMU propagation corrected by leg odometry, walking forward."""
    ekf, odom = ErrorStateEKF(), LegOdometry()
    speed = 0.4
    foot_pos = np.array([[0.2, 0.1, -0.3]] * 4)
    foot_vel = np.array([[-speed, 0.0, 0.0]] * 4)
    contacts = np.array([True, True, False, False])  # trot: diagonal pair down

    for _ in range(1000):  # 2 seconds
        ekf.predict(np.array([0.0, 0.0, 9.81]), np.zeros(3), DT)
        velocity, cov, _ = odom.measure(foot_pos, foot_vel, np.zeros(3), contacts)
        if velocity is not None:
            ekf.update_velocity(velocity, cov)

    assert ekf.observation_velocity()[0] == pytest.approx(speed, abs=0.05)
    assert quat_multiply(ekf.state.orientation, ekf.state.orientation).shape == (4,)
