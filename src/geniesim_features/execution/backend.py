"""Explicit Franka execution backend selection; no planner fallback."""
from __future__ import annotations

from dataclasses import dataclass


class BackendUnavailable(RuntimeError):
    """A requested planner backend cannot be initialized in this environment."""


@dataclass(frozen=True)
class BackendSpec:
    name: str
    planner_timeout_s: float = 30.0
    control_dt: float = 1 / 60


def create_motion_backend(name: str, *, session, robot, spec: BackendSpec, pick_path=None):
    """Construct the requested Franka backend; never silently substitute the other."""
    if name == "curobo_motiongen":
        from .curobo_backend import CuroboMotionGenBackend
        if pick_path is None:
            raise ValueError("cuRobo Franka backend needs the pick object collision path")
        return CuroboMotionGenBackend(session, robot, pick_path=pick_path,
                                      control_dt=spec.control_dt, timeout_s=spec.planner_timeout_s)
    if name == "franka_pickplace":
        from .franka_backend import FrankaPickPlaceBackend
        return FrankaPickPlaceBackend(session, robot)
    raise ValueError(f"unknown Franka motion backend: {name}")
