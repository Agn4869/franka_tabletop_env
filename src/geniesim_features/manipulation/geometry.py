"""Column-vector rigid transforms; all distances are metres, rotations radians."""
from __future__ import annotations

import numpy as np


def immutable_array(value, shape=None, *, dtype=np.float64, finite=True):
    if np.iscomplexobj(value):
        raise ValueError("complex values are not valid physical quantities")
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    if finite and not np.isfinite(array).all():
        raise ValueError("array contains non-finite values")
    # An immutable bytes owner prevents callers re-enabling WRITEABLE.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def rigid_transform(value):
    transform = immutable_array(value, (4, 4))
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("invalid homogeneous transform last row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0
    ):
        raise ValueError("transform must contain a proper rigid rotation, not scale/reflection")
    return transform


def inverse_transform(transform):
    transform = rigid_transform(transform)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def grasp_end_effector_pose(T_W_O, T_O_G, T_E_G):
    """T_W_E = T_W_O @ T_O_G @ inverse(T_E_G)."""
    return rigid_transform(rigid_transform(T_W_O) @ rigid_transform(T_O_G) @ inverse_transform(T_E_G))


def base_frame_target(T_W_B, T_W_E):
    return rigid_transform(inverse_transform(T_W_B) @ rigid_transform(T_W_E))


def pregrasp_pose(T_W_O, T_O_G, T_E_G, approach_direction_G, distance_m):
    direction = immutable_array(approach_direction_G, (3,))
    norm = np.linalg.norm(direction)
    if norm < 1e-12 or not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("nonzero approach direction and positive distance required")
    target = grasp_end_effector_pose(T_W_O, T_O_G, T_E_G).copy()
    direction_W = (rigid_transform(T_W_O) @ rigid_transform(T_O_G))[:3, :3] @ (direction / norm)
    target[:3, 3] -= distance_m * direction_W
    return rigid_transform(target)


def pose_error(actual, expected):
    actual, expected = rigid_transform(actual), rigid_transform(expected)
    relative = expected[:3, :3].T @ actual[:3, :3]
    angle = np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])), float(angle)
