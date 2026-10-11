"""Validated time-varying commands, not predicted or simulated drawer motion."""
from dataclasses import dataclass
import bisect
import numpy as np


@dataclass(frozen=True)
class DrawerCommandSchedule:
    mode: str
    times_s: tuple
    values: tuple
    interpolation: str
    after_end: str
    limits_m: tuple
    max_force_n: float
    direction_handle: tuple

    @classmethod
    def from_spec(cls, spec, *, limits_m=(0., .25), max_force_n=40.):
        if not isinstance(spec, dict) or set(spec) - {"schema_version", "mode", "interpolation", "keyframes", "after_end", "direction_handle"}:
            raise ValueError("Invalid schedule specification keys")
        if spec.get("schema_version", 1) != 1:
            raise ValueError("Schedule schema_version must be 1")
        mode = spec.get("mode")
        if mode not in {"external_force", "joint_drive"}:
            raise ValueError("Schedule mode must be external_force or joint_drive")
        limits = np.asarray(limits_m, float)
        max_force_n = float(max_force_n)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[1] <= limits[0]:
            raise ValueError("Invalid joint travel limits")
        if not np.isfinite(max_force_n) or max_force_n <= 0:
            raise ValueError("max_force_n must be positive and finite")
        direction = np.asarray(spec.get("direction_handle", [1, 0, 0]), float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-9:
            raise ValueError("direction_handle must be a finite nonzero Vector3")
        direction = direction / np.linalg.norm(direction)
        if mode == "joint_drive" and "direction_handle" in spec:
            raise ValueError("direction_handle is only for external_force schedules")
        interpolation = spec.get("interpolation", "hold")
        after_end = spec.get("after_end", "zero" if mode == "external_force" else "hold")
        if interpolation not in {"hold", "linear"} or after_end not in {"hold", "zero"}:
            raise ValueError("Unknown interpolation/end behavior")
        if mode == "joint_drive" and after_end == "zero" and not limits[0] <= 0 <= limits[1]:
            raise ValueError("Zero after_end drive target lies outside travel limits")
        keyframes = spec.get("keyframes")
        if not isinstance(keyframes, list) or not 1 <= len(keyframes) <= 10000:
            raise ValueError("Schedule requires 1..10000 keyframes")
        times, values, shape = [], [], None
        for entry in keyframes:
            if not isinstance(entry, dict) or set(entry) != {"time_s", "value"}:
                raise ValueError("Each keyframe requires exactly time_s and value")
            timestamp, value = float(entry["time_s"]), np.asarray(entry["value"], float)
            if not np.isfinite(timestamp) or timestamp < 0 or (times and timestamp <= times[-1]):
                raise ValueError("Keyframe times must be nonnegative and strictly increasing")
            if not np.isfinite(value).all() or value.shape not in ((), (3,)) or (shape is not None and value.shape != shape):
                raise ValueError("Values must be finite, consistently scalar or handle-frame Vector3")
            if mode == "joint_drive" and (value.shape != () or not limits[0] <= value <= limits[1]):
                raise ValueError("Drive schedule targets must be scalar within travel limits")
            if mode == "external_force" and np.linalg.norm(value) > max_force_n:
                raise ValueError("Force schedule exceeds force-norm limit")
            if value.shape == (3,) and "direction_handle" in spec:
                raise ValueError("Vector force keyframes already specify direction")
            shape = value.shape
            times.append(timestamp)
            values.append(float(value) if value.shape == () else tuple(map(float, value)))
        if times[0] != 0:
            raise ValueError("Schedule must explicitly define the command at time_s=0")
        return cls(mode, tuple(times), tuple(values), interpolation, after_end, tuple(map(float, limits)), max_force_n, tuple(map(float, direction)))

    @property
    def duration_s(self):
        return self.times_s[-1]

    def evaluate(self, elapsed_s):
        elapsed_s = float(elapsed_s)
        if not np.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("Schedule evaluation needs finite nonnegative elapsed time")
        index = max(0, bisect.bisect_right(self.times_s, elapsed_s)-1)
        value = np.asarray(self.values[index], float)
        if elapsed_s > self.duration_s and self.after_end == "zero":
            value = np.zeros_like(value)
        elif self.interpolation == "linear" and index < len(self.times_s)-1:
            ratio = (elapsed_s-self.times_s[index])/(self.times_s[index+1]-self.times_s[index])
            value = value*(1-ratio) + np.asarray(self.values[index+1])*ratio
        native = float(value) if value.shape == () else value.tolist()
        field = "position_m" if self.mode == "joint_drive" else ("force_n" if value.shape == () else "force_handle_n")
        return {"mode": self.mode, "time_s": elapsed_s, "value": native, field: native,
                "direction_handle": list(self.direction_handle),
                "finished": elapsed_s >= self.duration_s, "interpolation": self.interpolation,
                "after_end": self.after_end, "keyframe_index": index}

    def to_dict(self):
        return {"schema_version": 1, "mode": self.mode, "interpolation": self.interpolation,
                "after_end": self.after_end, "keyframes": [{"time_s": t, "value": v} for t, v in zip(self.times_s, self.values)],
                "limits_m": list(self.limits_m), "max_force_n": self.max_force_n,
                "direction_handle": list(self.direction_handle),
                "units": "m" if self.mode == "joint_drive" else "N", "vector_force_frame": "handle_local"}
