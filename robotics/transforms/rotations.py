"""Quaternion helpers. Hamilton convention, [w, x, y, z], body-to-world.

NumPy rather than torch: one robot, one CPU, and this should port to the C++
node later.
"""

import numpy as np


def skew(v):
    """Cross-product matrix: skew(a) @ b == cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_to_matrix(q):
    """Body-to-world rotation matrix."""
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_rotvec(rotvec):
    """Exponential map: a rotation vector (axis * angle) to a quaternion.

    Small-angle branch avoids dividing by a vanishing norm, which is the usual
    case here — error-state corrections are milliradians.
    """
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return quat_normalize(np.array([1.0, *(0.5 * np.asarray(rotvec))]))
    axis = np.asarray(rotvec) / theta
    return np.array([np.cos(theta / 2), *(np.sin(theta / 2) * axis)])


def quat_to_rotvec(q):
    w, x, y, z = quat_normalize(q)
    vec = np.array([x, y, z])
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return 2.0 * vec
    return 2.0 * np.arctan2(norm, w) * vec / norm


def rotate(q, v):
    """Apply a body-to-world rotation to a vector."""
    return quat_to_matrix(q) @ v


def gravity_from_quaternion(q, g=9.81):
    """Gravity expressed in the body frame — what an accelerometer sees at rest.

    This is the `projected_gravity` the policy consumes, and the quantity that
    makes roll and pitch observable from an accelerometer alone. Yaw is not
    observable this way, which is why it drifts.
    """
    return quat_to_matrix(q).T @ np.array([0.0, 0.0, -g])
