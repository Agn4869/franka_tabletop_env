"""Finite deterministic 2-D OBB/SAT placement, not a physics validator."""

from __future__ import annotations

import math
import re
import numpy as np

from .usd_utils import number, vector


class LayoutError(ValueError):
    def __init__(self, object_id, attempts, detail="no collision-free candidate inside the layout region"):
        self.object_id, self.attempts = object_id, attempts
        super().__init__(f"layout_unsatisfiable: object={object_id}, attempts={attempts}: {detail}")


def corners(center_xy, size_xy, yaw_deg, clearance=0.0):
    half = (np.asarray(size_xy, dtype=float) + clearance)/2
    theta = math.radians(yaw_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    points = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])*half
    return points @ rotation.T + np.asarray(center_xy)


def overlaps(a, b):
    """Separating-axis test with touching boundaries permitted."""
    if np.any(a.max(axis=0) <= b.min(axis=0)) or np.any(b.max(axis=0) <= a.min(axis=0)):
        return False
    for poly in (a, b):
        for i in (0, 1):
            edge = poly[i+1] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            ap, bp = a@axis, b@axis
            if ap.max() <= bp.min()+1e-12 or bp.max() <= ap.min()+1e-12:
                return False
    return True


def sample_layout(spec: dict, resolved: dict) -> list[dict]:
    layout = spec.get("layout", {})
    table = spec.get("table", {})
    table_size = vector(table.get("size_m", [1.2, .8, .05]), 3, "table size", positive=True)
    region = np.asarray(layout.get("region_xy_m", [[-.5*table_size[0],-.5*table_size[1]], [.5*table_size[0],.5*table_size[1]]]), dtype=float)
    if region.shape != (2,2) or not np.isfinite(region).all() or np.any(region[1] <= region[0]):
        raise ValueError("layout region must be finite [[xmin,ymin],[xmax,ymax]] with positive area")
    if np.any(region[0] < -np.asarray(table_size[:2])/2) or np.any(region[1] > np.asarray(table_size[:2])/2):
        raise ValueError("layout region extends beyond table")
    clearance = number(layout.get("clearance_m", 0.03), "layout clearance", nonnegative=True)
    attempts = layout.get("max_attempts_per_object", 100)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 10000:
        raise ValueError("max_attempts_per_object must be integer in [1,10000]")
    seed = spec.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    rng = np.random.default_rng(seed)
    occupied = []
    for excluded in layout.get("exclusion_zones", []):
        center = vector(excluded["center_xy_m"], 2, "exclusion center")
        size = vector(excluded["size_xy_m"], 2, "exclusion size", positive=True)
        occupied.append(corners(center, size, number(excluded.get("yaw_deg", 0), "exclusion yaw"), clearance))
    result = []
    # Fixed objects consume space before randomly placed objects, irrespective
    # of their order in the input configuration.
    objects = sorted(spec.get("objects", []), key=lambda obj: "fixed_xy_m" not in obj)
    for obj in objects:
        object_id = obj["id"]
        asset = resolved[object_id]
        size = vector(obj.get("size_m", asset.size_m), 3, f"{object_id} size", positive=True)
        if asset.primitive == "external" and "size_m" in obj:
            raise ValueError(f"{object_id}: external assets cannot be arbitrarily resized")
        footprint = size[:2]
        if asset.primitive == "drawer":
            # Reserve the full opening sweep conservatively about its root.
            upper = vector(obj.get("drawer", {}).get("limits_m", [0, .25]), 2, "drawer limits")[1]
            footprint = [size[0]+2*(upper+.032), size[1]]
        yaw = obj.get("yaw_deg", [0,0])
        yaw_range = [number(yaw, "yaw"), number(yaw, "yaw")] if isinstance(yaw, (int,float)) else vector(yaw, 2, "yaw range")
        if yaw_range[1] < yaw_range[0]:
            raise ValueError(f"{object_id}: yaw range must be ordered")
        fixed = vector(obj["fixed_xy_m"], 2, "fixed XY") if "fixed_xy_m" in obj else None
        limit = 1 if fixed is not None and yaw_range[0] == yaw_range[1] else attempts
        for index in range(limit):
            xy = fixed if fixed is not None else rng.uniform(region[0], region[1]).tolist()
            angle = float(rng.uniform(*yaw_range))
            poly = corners(xy, footprint, angle, clearance)
            if np.any(poly < region[0]-1e-12) or np.any(poly > region[1]+1e-12):
                continue
            if any(overlaps(poly, other) for other in occupied):
                continue
            occupied.append(poly)
            result.append({"object_id": object_id, "xy_m": list(xy), "yaw_deg": angle,
                           "size_m": size, "reserved_footprint_m": footprint, "attempts": index+1})
            break
        else:
            raise LayoutError(object_id, limit)
    return result


def validate_object_ids(objects):
    seen = {"table", "ground"}
    for obj in objects:
        object_id = obj.get("id")
        if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", object_id):
            raise ValueError(f"object id must be a USD-safe identifier: {object_id!r}")
        if object_id in seen:
            raise ValueError(f"duplicate/reserved object id: {object_id}")
        seen.add(object_id)
