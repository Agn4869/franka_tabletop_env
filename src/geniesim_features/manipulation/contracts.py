"""Engine-independent contracts. No command below teleports a robot or object."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from .geometry import immutable_array, rigid_transform


def joint_names_tuple(names):
    names = tuple(names)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("joint names must be nonempty, unique strings")
    return names


def _frozen_metadata(value):
    """Small observation metadata, independent of the disk recorder."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError("observation metadata keys must be nonempty strings")
        return MappingProxyType({key: _frozen_metadata(child) for key, child in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_frozen_metadata(child) for child in value)
    if isinstance(value, np.ndarray):
        return immutable_array(value)
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError("observation metadata must contain finite serializable values")


@dataclass(frozen=True)
class AngularGripper:
    joint_name: str
    closed_angle_rad: float
    open_angle_rad: float
    lower_limit_rad: float
    upper_limit_rad: float
    max_effort: float

    def __post_init__(self):
        values = [self.closed_angle_rad, self.open_angle_rad, self.lower_limit_rad,
                  self.upper_limit_rad, self.max_effort]
        if not self.joint_name or not np.isfinite(values).all() or self.max_effort <= 0:
            raise ValueError("invalid angular gripper calibration")
        if self.lower_limit_rad >= self.upper_limit_rad or self.closed_angle_rad == self.open_angle_rad:
            raise ValueError("gripper calibration must span distinct valid angles")
        if not all(self.lower_limit_rad <= angle <= self.upper_limit_rad
                   for angle in (self.closed_angle_rad, self.open_angle_rad)):
            raise ValueError("calibration exceeds gripper joint limits")

    def angle(self, open_ratio: float) -> float:
        if not np.isfinite(open_ratio) or not 0 <= open_ratio <= 1:
            raise ValueError("open_ratio must be within [0, 1], not a width in metres")
        return float(self.closed_angle_rad + open_ratio * (self.open_angle_rad - self.closed_angle_rad))


@dataclass(frozen=True)
class WorldObservation:
    step_index: int
    sim_time: float
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    T_W_E: np.ndarray
    T_W_O: np.ndarray
    T_W_B: np.ndarray
    object_linear_velocity: np.ndarray
    object_angular_velocity: np.ndarray
    gripper_open_ratio: float
    world_version: str
    contact_confirmed: bool = False
    unexpected_collision: bool = False
    sensors: Any = None
    wall_time: float | None = None
    scene_poses: Mapping[str, np.ndarray] = field(default_factory=dict)
    articulated_states: Mapping[str, Any] = field(default_factory=dict)
    contacts: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self):
        names = joint_names_tuple(self.joint_names)
        object.__setattr__(self, "joint_names", names)
        for name in ("joint_positions", "joint_velocities"):
            object.__setattr__(self, name, immutable_array(getattr(self, name), (len(names),)))
        for name in ("T_W_E", "T_W_O", "T_W_B"):
            object.__setattr__(self, name, rigid_transform(getattr(self, name)))
        for name in ("object_linear_velocity", "object_angular_velocity"):
            object.__setattr__(self, name, immutable_array(getattr(self, name), (3,)))
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, (int, np.integer)) or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if not np.isfinite(self.sim_time) or self.sim_time < 0 or not isinstance(self.world_version, str) or not self.world_version:
            raise ValueError("invalid simulation time or world version")
        for name in ("contact_confirmed", "unexpected_collision"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be actual boolean evidence")
            object.__setattr__(self, name, bool(getattr(self, name)))
        if not np.isfinite(self.gripper_open_ratio) or not 0 <= self.gripper_open_ratio <= 1:
            raise ValueError("invalid observed gripper open ratio")
        if self.wall_time is not None and not np.isfinite(self.wall_time):
            raise ValueError("invalid wall time")
        if not isinstance(self.scene_poses, Mapping) or any(not isinstance(key, str) or not key for key in self.scene_poses):
            raise ValueError("scene_poses requires object IDs")
        object.__setattr__(self, "scene_poses", MappingProxyType(
            {key: rigid_transform(pose) for key, pose in self.scene_poses.items()}))
        if not isinstance(self.articulated_states, Mapping):
            raise ValueError("articulated_states must be a mapping")
        if not isinstance(self.contacts, (tuple, list)) or any(not isinstance(item, Mapping) for item in self.contacts):
            raise ValueError("contacts must be a sequence of contact summaries")
        object.__setattr__(self, "articulated_states", _frozen_metadata(self.articulated_states))
        object.__setattr__(self, "contacts", _frozen_metadata(self.contacts))


@dataclass(frozen=True)
class ControlCommand:
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    gripper_open_ratio: float
    gripper_angle_rad: float
    stage: str
    world_version: str
    plan_id: str = ""
    control_mode: str = "joint_position"
    candidate_id: str = ""

    def __post_init__(self):
        names = joint_names_tuple(self.joint_names)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "joint_positions", immutable_array(self.joint_positions, (len(names),)))
        if self.control_mode != "joint_position":
            raise ValueError("only dynamic joint-position controller targets are supported")
        if not np.isfinite([self.gripper_open_ratio, self.gripper_angle_rad]).all() or not 0 <= self.gripper_open_ratio <= 1:
            raise ValueError("invalid gripper command")


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    T_O_G: np.ndarray
    approach_direction_G: np.ndarray = field(default_factory=lambda: np.array([0., 0., -1.]))
    pregrasp_distance_m: float = 0.1

    def __post_init__(self):
        if not self.candidate_id:
            raise ValueError("candidate id required")
        object.__setattr__(self, "T_O_G", rigid_transform(self.T_O_G))
        direction = immutable_array(self.approach_direction_G, (3,))
        norm = np.linalg.norm(direction)
        if norm < 1e-12 or not np.isfinite(self.pregrasp_distance_m) or self.pregrasp_distance_m <= 0:
            raise ValueError("invalid approach direction/distance")
        object.__setattr__(self, "approach_direction_G", immutable_array(direction / norm))


@dataclass(frozen=True)
class PickPlaceTask:
    object_id: str
    candidates: tuple[GraspCandidate, ...]
    T_E_G: np.ndarray
    T_W_O_place: np.ndarray
    bin_min_W: np.ndarray
    bin_max_W: np.ndarray
    object_half_extents: np.ndarray
    control_dt: float = 1 / 30
    physics_steps_per_interval: int = 4
    stage_timeout_s: float = 20.0
    episode_timeout_s: float = 180.0
    planner_timeout_s: float = 30.0
    max_replans: int = 3
    max_candidates: int = 8
    lift_height_m: float = 0.12
    min_lift_m: float = 0.05
    lift_hold_s: float = 1.0
    grip_hold_s: float = 0.2
    settle_hold_s: float = 0.5
    retreat_distance_m: float = 0.1
    joint_tolerance_rad: float = 0.04
    tracking_tolerance_rad: float = 0.35
    pose_tolerance_m: float = 0.02
    rotation_tolerance_rad: float = 0.15
    slip_tolerance_m: float = 0.025
    slip_tolerance_rad: float = 0.25
    settle_linear_speed: float = 0.02
    settle_angular_speed: float = 0.1
    placement_tolerance_m: float = 0.002
    execution_mode: str = "contact_physics"

    def __post_init__(self):
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.object_id or not self.candidates or len({c.candidate_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("object and distinct finite grasp candidates required")
        for name in ("T_E_G", "T_W_O_place"):
            object.__setattr__(self, name, rigid_transform(getattr(self, name)))
        for name in ("bin_min_W", "bin_max_W", "object_half_extents"):
            object.__setattr__(self, name, immutable_array(getattr(self, name), (3,)))
        if np.any(self.bin_min_W >= self.bin_max_W) or np.any(self.object_half_extents <= 0):
            raise ValueError("invalid bin bounds/object extents")
        for name in ("control_dt", "stage_timeout_s", "episode_timeout_s", "planner_timeout_s", "lift_height_m",
                     "min_lift_m", "lift_hold_s", "grip_hold_s", "settle_hold_s", "retreat_distance_m",
                     "joint_tolerance_rad", "tracking_tolerance_rad", "pose_tolerance_m", "rotation_tolerance_rad",
                     "slip_tolerance_m", "slip_tolerance_rad", "settle_linear_speed", "settle_angular_speed",
                     "placement_tolerance_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("physics_steps_per_interval", "max_candidates", "max_replans"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.lift_height_m <= self.min_lift_m:
            raise ValueError("lift target must exceed the minimum observed lift")
        if self.execution_mode != "contact_physics":
            raise ValueError("this executor supports contact_physics only; no attachment constraints")


@dataclass(frozen=True)
class TaskProgress:
    done: bool
    success: bool
    stage: str
    failure_reason: str | None = None
    candidate_id: str | None = None
    planning_wall_time_s: float = 0.0
    planning_attempts: int = 0
    planning_events: tuple[dict, ...] = ()


class MotionPlannerAdapter(Protocol):
    """sync_world must preserve attached geometry and scoped collision policy.

    plan must enforce its own finite wall-time deadline; the executor additionally
    rejects a returned plan when its wall-time budget was exceeded. attach/detach
    change PLANNER geometry only, never a physical constraint in the engine.
    """
    def sync_world(self, observation: WorldObservation) -> str: ...
    def plan(self, joint_names: tuple[str, ...], joint_positions: np.ndarray,
             T_B_E: np.ndarray, world_version: str) -> "JointTrajectory": ...
    def attach(self, object_id: str, T_E_O: np.ndarray) -> None: ...
    def detach(self, object_id: str, T_W_O: np.ndarray) -> None: ...
    def cleanup(self) -> None: ...


# Forward reference without introducing an import cycle.
from .trajectory import JointTrajectory  # noqa: E402
