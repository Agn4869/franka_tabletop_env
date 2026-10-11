"""Time-based interpolation and limits; this does not implement a motion planner."""
from __future__ import annotations

from dataclasses import dataclass, field
import uuid
import numpy as np

from .geometry import immutable_array


def _names(value):
    names = tuple(value)
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("joint names must be distinct nonempty strings")
    return names


@dataclass(frozen=True)
class JointLimits:
    joint_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    max_velocity: np.ndarray
    max_acceleration: np.ndarray

    def __post_init__(self):
        names = _names(self.joint_names)
        object.__setattr__(self, "joint_names", names)
        for name in ("lower", "upper", "max_velocity", "max_acceleration"):
            object.__setattr__(self, name, immutable_array(getattr(self, name), (len(names),)))
        if np.any(self.lower >= self.upper) or np.any(self.max_velocity <= 0) or np.any(self.max_acceleration <= 0):
            raise ValueError("invalid joint limits")

    def ordered(self, names):
        names = _names(names)
        if set(names) != set(self.joint_names):
            raise ValueError("planned and controlled joint sets differ")
        indices = [self.joint_names.index(name) for name in names]
        return tuple(getattr(self, field)[indices] for field in ("lower", "upper", "max_velocity", "max_acceleration"))


@dataclass(frozen=True)
class JointTrajectory:
    joint_names: tuple[str, ...]
    times: np.ndarray
    positions: np.ndarray
    world_version: str
    velocities: np.ndarray | None = None
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):
        names = _names(self.joint_names)
        times = immutable_array(self.times)
        if times.ndim != 1 or len(times) < 2 or abs(times[0]) > 1e-10 or np.any(np.diff(times) <= 0):
            raise ValueError("trajectory times must start at zero and strictly increase, with >=2 samples")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", immutable_array(self.positions, (len(times), len(names))))
        if self.velocities is not None:
            object.__setattr__(self, "velocities", immutable_array(self.velocities, self.positions.shape))
        if not isinstance(self.world_version, str) or not self.world_version or not self.plan_id:
            raise ValueError("world version and plan id required")

    @property
    def duration(self):
        return float(self.times[-1])

    def sample(self, time_s: float):
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("invalid trajectory sample time")
        return immutable_array([np.interp(time_s, self.times, self.positions[:, i]) for i in range(len(self.joint_names))])

    def resample(self, dt: float, *, max_samples: int = 1_000_000):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite sample period required")
        if not isinstance(max_samples, int) or max_samples < 2 or self.duration / dt >= max_samples:
            raise ValueError("resampled trajectory exceeds sample budget")
        # The final interval may be shorter; runtime sample() holds the endpoint.
        times = np.arange(0, self.duration, dt)
        times = np.append(times[times < self.duration - 1e-10], self.duration)
        return JointTrajectory(self.joint_names, times, np.vstack([self.sample(t) for t in times]),
                               self.world_version, plan_id=self.plan_id)

    def reordered(self, joint_names):
        names = _names(joint_names)
        if set(names) != set(self.joint_names):
            raise ValueError("joint mapping must be bijective")
        indices = [self.joint_names.index(name) for name in names]
        velocity = None if self.velocities is None else self.velocities[:, indices]
        return JointTrajectory(names, self.times, self.positions[:, indices], self.world_version, velocity, self.plan_id)

    def validate_limits(self, limits: JointLimits, tolerance=1e-8):
        lower, upper, velocity_limit, acceleration_limit = limits.ordered(self.joint_names)
        if np.any(self.positions < lower - tolerance) or np.any(self.positions > upper + tolerance):
            raise ValueError("trajectory_position_limit")
        interval = np.diff(self.times)
        slopes = np.diff(self.positions, axis=0) / interval[:, None]
        if np.any(np.abs(slopes) > velocity_limit + tolerance):
            raise ValueError("trajectory_velocity_limit")
        if len(slopes) > 1:
            accelerations = np.diff(slopes, axis=0) / ((interval[:-1] + interval[1:]) / 2)[:, None]
            if np.any(np.abs(accelerations) > acceleration_limit + tolerance):
                raise ValueError("trajectory_acceleration_limit")
        if self.velocities is not None:
            if np.any(np.abs(self.velocities) > velocity_limit + tolerance):
                raise ValueError("trajectory_velocity_limit")
            if np.any(np.abs(np.diff(self.velocities, axis=0) / interval[:, None]) > acceleration_limit + tolerance):
                raise ValueError("trajectory_acceleration_limit")
        return self
