"""Depth conversion uses axial plane-Z meters and optical X-right/Y-down/Z-forward."""
from functools import lru_cache
from pathlib import Path
import numpy as np

from .models import CameraCalibration, frozen_array, validate_intrinsics, validate_transform


@lru_cache(maxsize=32)
def _ray_grid(height, width, K_values):
    K = np.array(K_values).reshape(3, 3)
    v, u = np.indices((height, width), dtype=np.float64)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    rays = pixels @ np.linalg.inv(K).T
    return frozen_array(rays / rays[..., 2:3])


def normalized_pixel_grid(height, width, intrinsics):
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in (height, width)):
        raise ValueError("positive integer dimensions required")
    K = validate_intrinsics(intrinsics)
    # K is part of the key: same resolution with new intrinsics never reuses old rays.
    return _ray_grid(int(height), int(width), tuple(K.ravel()))


def depth_valid_mask(depth, near=0.0, far=np.inf):
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.dtype.kind not in "fiu":
        raise ValueError("depth must be numeric HxW")
    if not np.isfinite(near) or near < 0 or np.isnan(far) or far <= near:
        raise ValueError("clipping requires 0 <= near < far")
    return np.isfinite(depth) & (depth > 0) & (depth >= near) & (depth <= far)


def range_to_plane_z(depth_range, intrinsics):
    distance = np.asarray(depth_range, dtype=np.float64)
    if distance.ndim != 2:
        raise ValueError("range depth must be HxW")
    rays = normalized_pixel_grid(*distance.shape, intrinsics)
    return (distance / np.linalg.norm(rays, axis=-1)).astype(np.float32)


def depth_to_points(depth_z_m, intrinsics, valid_mask=None, T_world_optical=None, *, near=None, far=None):
    depth = np.asarray(depth_z_m)
    if isinstance(intrinsics, CameraCalibration):
        calibration = intrinsics
        if depth.shape != (calibration.height, calibration.width):
            raise ValueError("depth/calibration resolution mismatch")
        intrinsics = calibration.K
        if T_world_optical is None:
            T_world_optical = calibration.T_world_optical
        near = calibration.near if near is None else near
        far = calibration.far if far is None else far
    mask = depth_valid_mask(depth, 0.0 if near is None else near, np.inf if far is None else far)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.dtype != np.bool_ or valid_mask.shape != depth.shape:
            raise ValueError("valid_mask must be bool HxW")
        mask &= valid_mask
    rays = normalized_pixel_grid(*depth.shape, intrinsics)
    points = rays[mask] * depth[mask, None]
    if T_world_optical is not None:
        T = validate_transform(T_world_optical)
        points = points @ T[:3, :3].T + T[:3, 3]
    return points.astype(np.float32)


def visible_bboxes(instance_ids, id_to_object, object_ids=()):
    """Return per-object visible [xmin,ymin,xmax_exclusive,ymax_exclusive], None if hidden."""
    labels = np.asarray(instance_ids)
    if labels.ndim != 2 or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("instance_ids must be a nonnegative integer image")
    result = {}
    for object_id in sorted(set(id_to_object.values()) | set(object_ids)):
        ids = [int(i) for i, value in id_to_object.items() if value == object_id and int(i) != 0]
        rows, cols = np.where(np.isin(labels, ids))
        result[object_id] = None if not len(rows) else [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]
    return result


def write_ply(path, points, colors=None):
    """Exclusive-create ASCII PLY; caller chooses only valid, finite points."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx3")
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != points.shape or colors.dtype != np.uint8:
            raise ValueError("colors must be matching Nx3 uint8")
    path = Path(path)
    with path.open("x", encoding="ascii") as handle:
        handle.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for index, point in enumerate(points):
            line = " ".join(format(float(value), ".9g") for value in point)
            if colors is not None:
                line += " " + " ".join(str(int(value)) for value in colors[index])
            handle.write(line + "\n")
    return path
