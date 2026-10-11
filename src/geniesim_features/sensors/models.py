"""Owned, immutable pinhole observations; no simulator imports or fake rendering."""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
import re

import numpy as np


def safe_identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("identifier must be one safe ASCII path component")
    return value


def frozen_array(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    # A bytes owner prevents callers from re-enabling the writeable flag.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def validate_intrinsics(value):
    K = np.asarray(value, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("K must be finite 3x3")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("invalid pinhole intrinsics")
    if not np.isclose(K[1, 0], 0):
        raise ValueError("K must be upper triangular")
    return K


def validate_transform(value):
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("T_world_optical must be finite 4x4")
    R = T[:3, :3]
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("invalid homogeneous row")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError("transform rotation must be right-handed orthonormal")
    return T


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    K: np.ndarray
    T_world_optical: np.ndarray
    near: float = 0.01
    far: float = 100.0

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in (self.width, self.height)):
            raise ValueError("positive integer image dimensions required")
        if not np.isfinite([self.near, self.far]).all() or not 0 < self.near < self.far:
            raise ValueError("clipping requires finite 0 < near < far")
        object.__setattr__(self, "K", frozen_array(validate_intrinsics(self.K)))
        object.__setattr__(self, "T_world_optical", frozen_array(validate_transform(self.T_world_optical)))

    def to_dict(self):
        return {"width": int(self.width), "height": int(self.height), "K": self.K.tolist(),
                "T_world_optical": self.T_world_optical.tolist(), "near": float(self.near), "far": float(self.far),
                "depth_semantics": "distance_to_image_plane_m", "projection_model": "pinhole",
                "optical_axes": "x_right_y_down_z_forward"}


@dataclass(frozen=True)
class SensorFrame:
    camera_id: str
    physics_step: int
    rgb: np.ndarray
    depth: np.ndarray
    instance_ids: np.ndarray
    calibration: CameraCalibration
    id_to_object: Mapping[int, str]
    reserved_instance_ids: Mapping[int, str] = field(default_factory=lambda: {0: "BACKGROUND"})
    raw_id_to_label: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self):
        safe_identifier(self.camera_id)
        if isinstance(self.physics_step, bool) or not isinstance(self.physics_step, (int, np.integer)) or self.physics_step < 0:
            raise ValueError("physics_step must be a nonnegative integer")
        if not isinstance(self.calibration, CameraCalibration):
            raise TypeError("calibration must be CameraCalibration")
        shape = (self.calibration.height, self.calibration.width)
        rgb, depth, labels = np.asarray(self.rgb), np.asarray(self.depth), np.asarray(self.instance_ids)
        if rgb.shape != (*shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("rgb must be HxWx3 uint8; convert RGBA explicitly at the adapter")
        if depth.shape != shape or depth.dtype.kind not in "fiu":
            raise ValueError("depth must be numeric HxW in axial meters")
        if labels.shape != shape or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels > np.iinfo(np.uint32).max):
            raise ValueError("instance_ids must be HxW nonnegative uint32-compatible integers")
        mapping = {}
        for key, value in self.id_to_object.items():
            if isinstance(key, bool) or not isinstance(key, (int, np.integer)) or not 0 <= key <= np.iinfo(np.uint32).max:
                raise ValueError("instance map keys must be uint32-compatible integers")
            if not isinstance(value, str) or not value:
                raise ValueError("object ids must be nonempty strings")
            mapping[int(key)] = value
        reserved, raw = {}, {}
        for source, destination in ((self.reserved_instance_ids, reserved), (self.raw_id_to_label, raw)):
            for key, value in source.items():
                if isinstance(key, bool) or not isinstance(key, (int, np.integer)) or not 0 <= key <= np.iinfo(np.uint32).max:
                    raise ValueError("renderer mapping keys must be uint32-compatible integers")
                if not isinstance(value, str) or not value:
                    raise ValueError("renderer labels must be nonempty strings")
                destination[int(key)] = value
        if any(label not in {"BACKGROUND", "UNLABELLED"} for label in reserved.values()):
            raise ValueError("only renderer BACKGROUND/UNLABELLED labels can be reserved")
        if set(reserved) & set(mapping):
            raise ValueError("reserved renderer ids cannot also name business objects")
        if raw and any(raw.get(key) != value for key, value in reserved.items()):
            raise ValueError("reserved ids must match the supplied original renderer labels")
        missing = set(map(int, np.unique(labels))) - set(mapping) - set(reserved)
        if missing:
            raise ValueError(f"unmapped visible instance ids: {sorted(missing)}")
        object.__setattr__(self, "rgb", frozen_array(rgb))
        object.__setattr__(self, "depth", frozen_array(depth, np.float32))
        object.__setattr__(self, "instance_ids", frozen_array(labels, np.uint32))
        object.__setattr__(self, "id_to_object", MappingProxyType(mapping))
        object.__setattr__(self, "reserved_instance_ids", MappingProxyType(reserved))
        object.__setattr__(self, "raw_id_to_label", MappingProxyType(raw))


@dataclass(frozen=True)
class SensorBundle:
    physics_step: int
    simulation_time: float
    frames: Mapping[str, SensorFrame]
    frame_index: int = 0

    def __post_init__(self):
        for value in (self.physics_step, self.frame_index):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError("step/index must be nonnegative integers")
        if not np.isfinite(self.simulation_time) or self.simulation_time < 0:
            raise ValueError("simulation_time must be finite and nonnegative")
        if not self.frames:
            raise ValueError("bundle needs at least one camera")
        copied = dict(self.frames)
        for key, frame in copied.items():
            if not isinstance(frame, SensorFrame) or key != frame.camera_id:
                raise ValueError("frame mapping must match camera ids")
            if frame.physics_step != self.physics_step:
                raise ValueError("all frames must belong to exactly the bundle physics_step")
        object.__setattr__(self, "frames", MappingProxyType(copied))
