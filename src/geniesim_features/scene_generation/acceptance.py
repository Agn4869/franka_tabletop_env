"""Bounded acceptance of OBSERVED rigid-body samples, never a physics substitute.

No pairwise AABB overlap test is used: container cavities and articulated
structures overlap in their coarse bounds without colliding. Only an explicitly
declared solid box may use its exact corners against a declared planar support.
Other penetration evidence must come from measured contact separation/depth.
"""
from copy import deepcopy
import itertools
import numpy as np

from ..sensors.models import validate_transform


DEFAULTS = {"stable_window_s": .25, "max_duration_s": 3., "linear_speed_m_s": .01,
            "angular_speed_rad_s": .05, "penetration_tolerance_m": .002,
            "penetration_duration_s": .15, "fallen_margin_m": .05,
            "max_sample_gap_s": .1, "table_edge_tolerance_m": .002,
            "support_gap_tolerance_m": .02, "max_tilt_deg": None,
            "require_penetration_evidence": False}


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite length-{size} vector")
    return result


class SceneSettleMonitor:
    """Consume monotonic samples, retain reasons, and never advance a world.

    object_specs values: size_m; optional solid_box=True; table_supported=True;
    upright_axis_local; max_tilt_deg. Articulated/static objects are excluded by
    from_manifest. Direct callers must make that same explicit selection.
    """
    def __init__(self, object_specs, bounds_xy_m, table_top_z_m, config=None, *, excluded_objects=None):
        self.object_specs = deepcopy(object_specs)
        self.bounds = np.asarray(bounds_xy_m, dtype=float)
        if self.bounds.shape != (2, 2) or not np.isfinite(self.bounds).all() or np.any(self.bounds[1] <= self.bounds[0]):
            raise ValueError("table bounds must be [[xmin,ymin],[xmax,ymax]]")
        self.table_top = float(table_top_z_m)
        if not np.isfinite(self.table_top):
            raise ValueError("table height must be finite")
        config = {} if config is None else config
        if set(config) - set(DEFAULTS):
            raise ValueError(f"Unknown acceptance settings: {sorted(set(config)-set(DEFAULTS))}")
        self.config = {**DEFAULTS, **config}
        for key, value in self.config.items():
            if key == "require_penetration_evidence":
                if type(value) is not bool:
                    raise ValueError("require_penetration_evidence must be boolean")
            elif key == "max_tilt_deg":
                if value is not None and (not np.isfinite(value) or not 0 < value <= 180):
                    raise ValueError("max_tilt_deg must be None or within (0,180]")
            elif isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.config["max_duration_s"] < self.config["stable_window_s"]:
            raise ValueError("max_duration_s must cover the stable window")
        for object_id, spec in self.object_specs.items():
            if not isinstance(object_id, str) or not object_id or not isinstance(spec, dict):
                raise ValueError("Invalid monitored object specification")
            if "size_m" in spec and np.any(_vector(spec["size_m"], 3, "size_m") <= 0):
                raise ValueError("object size must be positive")
            if spec.get("solid_box") and "size_m" not in spec:
                raise ValueError("exact solid-box checks require size_m")
            axis = _vector(spec.get("upright_axis_local", [0, 0, 1]), 3, "upright_axis_local")
            if np.linalg.norm(axis) < 1e-9:
                raise ValueError("upright axis must be nonzero")
            tilt = spec.get("max_tilt_deg", self.config["max_tilt_deg"])
            if tilt is not None and (not np.isfinite(tilt) or not 0 < tilt <= 180):
                raise ValueError("object max_tilt_deg must be None or within (0,180]")
        self.excluded = deepcopy(excluded_objects or {})
        self._start_time = self._last_time = self._last_step = None
        self._stable_since = None
        self._penetrating_since = {}
        self._sample_count = 0
        self._result = None

    @classmethod
    def from_manifest(cls, manifest, config=None):
        resolved = manifest.get("resolved_config", {})
        table = resolved.get("table", {})
        size = _vector(table.get("size_m", [1.2, .8, .05]), 3, "table size")
        if np.any(size <= 0):
            raise ValueError("table dimensions must be positive")
        specs, excluded = {}, {}
        source_objects = {item["id"]: item for item in resolved.get("objects", [])}
        for object_id, binding in manifest.get("bindings", {}).items():
            if binding.get("joint_prim") or binding.get("body_prim"):
                excluded[object_id] = "articulated_object_requires_its_joint_acceptance"
                continue
            if not binding.get("dynamic", False):
                excluded[object_id] = "static_object_not_a_settling_rigid_body"
                continue
            asset = manifest.get("assets", {}).get(object_id, {})
            object_options = source_objects.get(object_id, {}).get("acceptance", {})
            specs[object_id] = {"size_m": binding["size_m"],
                "solid_box": asset.get("primitive") == "box",
                "table_supported": binding.get("support_prim") == "/World/Table/Top",
                "upright_axis_local": object_options.get("upright_axis_local", [0, 0, 1]),
                "max_tilt_deg": object_options.get("max_tilt_deg", (config or {}).get("max_tilt_deg"))}
        origin = _vector(table.get("origin_xy_m", [0.0, 0.0]), 2, "table origin")
        bounds = [[origin[0] - size[0]/2, origin[1] - size[1]/2],
                  [origin[0] + size[0]/2, origin[1] + size[1]/2]]
        return cls(specs, bounds, table.get("top_z_m", .75), config,
                   excluded_objects=excluded)

    @property
    def monitored_object_ids(self):
        return tuple(self.object_specs)

    def _report(self, status, reasons, per_object, timestamp):
        duration = 0. if self._start_time is None else timestamp - self._start_time
        stable = 0. if self._stable_since is None else timestamp - self._stable_since
        self._result = {"schema_version": 1, "status": status, "accepted": status == "accepted",
            "terminal": status != "pending", "reasons": reasons, "per_object": per_object,
            "physics_step": self._last_step, "simulation_time": timestamp, "sample_count": self._sample_count,
            "duration_s": float(duration), "stable_duration_s": float(max(0., stable)),
            "monitored_object_ids": list(self.object_specs), "excluded_objects": self.excluded,
            "thresholds": self.config, "evidence_scope": {
                "pose_and_velocity": "caller_supplied_observed_rigid_body_samples",
                "pairwise_collision_check": "not_performed",
                "arbitrary_concave_geometry_penetration": "requires_engine_contact_depth",
                "robot_reachability": "not_checked"}}
        return deepcopy(self._result)

    def update(self, snapshot):
        if self._result and self._result["status"] == "rejected":
            return deepcopy(self._result)
        step, timestamp = snapshot.get("physics_step"), snapshot.get("simulation_time")
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("snapshot.physics_step must be a nonnegative integer")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float, np.number)) or not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("snapshot.simulation_time must be finite and nonnegative")
        if self._last_step is not None and (step <= self._last_step or timestamp <= self._last_time):
            raise ValueError("acceptance samples must advance both physics step and simulation time")
        if self._last_time is not None and timestamp - self._last_time > self.config["max_sample_gap_s"] + 1e-9:
            self._stable_since = None
            self._penetrating_since.clear()
        if self._start_time is None:
            self._start_time = float(timestamp)
        self._last_step, self._last_time = int(step), float(timestamp)
        self._sample_count += 1
        observations = snapshot.get("objects", {})
        per_object, reasons = {}, []
        all_still = bool(self.object_specs)
        for object_id, spec in self.object_specs.items():
            if object_id not in observations:
                reasons.append({"object_id": object_id, "code": "missing_observation"})
                continue
            state = observations[object_id]
            try:
                T = validate_transform(state["T_world_object"])
                linear = _vector(state["linear_velocity_m_s"], 3, "linear velocity")
                angular = _vector(state["angular_velocity_rad_s"], 3, "angular velocity")
            except (KeyError, ValueError, TypeError) as error:
                reasons.append({"object_id": object_id, "code": "invalid_observed_state", "detail": str(error)})
                continue
            linear_speed, angular_speed = float(np.linalg.norm(linear)), float(np.linalg.norm(angular))
            still = linear_speed <= self.config["linear_speed_m_s"] and angular_speed <= self.config["angular_speed_rad_s"]
            all_still &= still
            metrics = {"position_world_m": T[:3, 3].tolist(), "linear_speed_m_s": linear_speed,
                       "angular_speed_rad_s": angular_speed, "velocity_within_threshold": bool(still),
                       "penetration_check": "not_available", "max_penetration_m": None}
            per_object[object_id] = metrics
            table_supported = spec.get("table_supported", True)
            corners = None
            if spec.get("solid_box"):
                half = np.asarray(spec["size_m"]) / 2
                local = np.array(list(itertools.product(*[(-value, value) for value in half])))
                corners = local @ T[:3, :3].T + T[:3, 3]
            if table_supported:
                xy = corners[:, :2] if corners is not None else T[None, :2, 3]
                edge = self.config["table_edge_tolerance_m"]
                if np.any(xy < self.bounds[0]-edge) or np.any(xy > self.bounds[1]+edge):
                    reasons.append({"object_id": object_id, "code": "outside_table_bounds",
                                    "extent_basis": "exact_box_corners" if corners is not None else "object_origin_only"})
                if T[2, 3] < self.table_top - self.config["fallen_margin_m"]:
                    reasons.append({"object_id": object_id, "code": "fallen_below_support"})
                if corners is not None:
                    metrics["support_gap_m"] = float(corners[:, 2].min() - self.table_top)
                    metrics["max_penetration_m"] = max(0., -metrics["support_gap_m"])
                    metrics["penetration_check"] = "exact_solid_box_against_declared_table_plane"
            contacts = state.get("contact_summary", {})
            measured_depth = contacts.get("max_penetration_m") if isinstance(contacts, dict) and contacts.get("available") else None
            if measured_depth is not None:
                if not isinstance(measured_depth, (int, float, np.number)) or not np.isfinite(measured_depth) or measured_depth < 0:
                    reasons.append({"object_id": object_id, "code": "invalid_contact_depth"})
                else:
                    metrics["max_penetration_m"] = max(float(measured_depth), metrics["max_penetration_m"] or 0.)
                    metrics["penetration_check"] += "+engine_contact_depth" if metrics["penetration_check"] != "not_available" else ""
                    if metrics["penetration_check"] == "not_available":
                        metrics["penetration_check"] = "engine_contact_depth"
            depth = metrics["max_penetration_m"]
            if depth is not None and depth > self.config["penetration_tolerance_m"]:
                since = self._penetrating_since.setdefault(object_id, float(timestamp))
                all_still = False  # Never accept while waiting for penetration persistence.
                metrics["penetration_duration_s"] = float(timestamp - since)
                if timestamp - since + 1e-9 >= self.config["penetration_duration_s"]:
                    reasons.append({"object_id": object_id, "code": "persistent_penetration", "depth_m": depth,
                                    "duration_s": float(timestamp-since), "basis": metrics["penetration_check"]})
            else:
                self._penetrating_since.pop(object_id, None)
            if self.config["require_penetration_evidence"] and depth is None:
                reasons.append({"object_id": object_id, "code": "penetration_evidence_unavailable"})
            max_tilt = spec.get("max_tilt_deg", self.config["max_tilt_deg"])
            if max_tilt is not None:
                axis = np.asarray(spec.get("upright_axis_local", [0, 0, 1]), float)
                cosine = float((T[:3, :3] @ (axis/np.linalg.norm(axis)))[2])
                tilt = float(np.rad2deg(np.arccos(np.clip(cosine, -1., 1.))))
                metrics["tilt_deg"] = tilt
                if tilt > max_tilt:
                    reasons.append({"object_id": object_id, "code": "tilted_beyond_configured_limit", "tilt_deg": tilt})
        if not self.object_specs:
            reasons.append({"code": "no_eligible_dynamic_objects", "detail": "No rigid-body stability test was performed"})
        if reasons:
            self._stable_since = None
            return self._report("rejected", reasons, per_object, float(timestamp))
        if all_still:
            if self._stable_since is None:
                self._stable_since = float(timestamp)
        else:
            self._stable_since = None
        stable_time = 0. if self._stable_since is None else timestamp - self._stable_since
        if stable_time + 1e-9 >= self.config["stable_window_s"]:
            floating = [{"object_id": key, "code": "unsupported_or_floating", "support_gap_m": item["support_gap_m"]}
                        for key, item in per_object.items() if item.get("support_gap_m", 0) > self.config["support_gap_tolerance_m"]]
            return self._report("rejected" if floating else "accepted", floating, per_object, float(timestamp))
        if timestamp - self._start_time + 1e-9 >= self.config["max_duration_s"]:
            return self._report("rejected", [{"code": "unstable_timeout"}], per_object, float(timestamp))
        return self._report("pending", [], per_object, float(timestamp))

    def finish(self):
        if self._result and self._result["terminal"]:
            return deepcopy(self._result)
        reason = "no_observations" if self._result is None else "settle_window_not_met"
        return self._report("rejected", [{"code": reason}], {} if self._result is None else self._result["per_object"],
                            0. if self._last_time is None else self._last_time)
