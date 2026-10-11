"""Validate asset-specific robot information before starting a GPU application."""

from pathlib import Path
import hashlib
import re
import numpy as np

from ..config import load_config
from ..manipulation import AngularGripper, JointLimits
from ..manipulation.geometry import rigid_transform


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prim_path(value, name):
    # Pure-Python preflight: importing pxr/Isaac before SimulationApp can load
    # the wrong USD runtime in an actual Isaac interpreter.
    if not isinstance(value, str) or not re.fullmatch(r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*", value):
        raise ValueError(f"{name} must be an absolute non-root USD prim path")
    return value


def _link(value, root, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty link name or prim path")
    if value.startswith("/"):
        _prim_path(value, name)
        if not value.startswith(root+"/"):
            raise ValueError(f"{name} must be under robot root {root}")
    elif not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"{name} must be a leaf link name or absolute prim path")
    return value


def load_robot_profile(path):
    path = Path(path).resolve()
    profile = load_config(path)
    required = ["robot_usd", "root_prim", "base_link", "end_effector_link", "joint_names",
                "initial_joint_positions", "inactive_joint_positions", "limits", "gripper",
                "T_E_G", "curobo_config", "asset_sha256", "license_source"]
    missing = [name for name in required if profile.get(name) is None]
    if missing:
        raise ValueError("Robot profile needs asset calibration: " + ", ".join(missing))
    root = _prim_path(profile["root_prim"], "root_prim")
    for name in ("base_link", "end_effector_link"):
        _link(profile[name], root, name)
    if not isinstance(profile["joint_names"], list):
        raise ValueError("joint_names must be an explicit list, not a string")
    names = tuple(profile["joint_names"])
    if not isinstance(profile["limits"], dict) or set(profile["limits"]) != {"lower", "upper", "max_velocity", "max_acceleration"}:
        raise ValueError("limits requires lower, upper, max_velocity and max_acceleration arrays")
    if any(value is None for value in profile["limits"].values()):
        raise ValueError("Robot profile limits need asset calibration; null values cannot be used")
    limits = JointLimits(names, **profile["limits"])
    initial = np.asarray(profile["initial_joint_positions"], float)
    if initial.shape != (len(names),) or not np.isfinite(initial).all():
        raise ValueError("initial_joint_positions must match active joints")
    lower, upper, _, _ = limits.ordered(names)
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("Initial active joint positions exceed limits")
    if not isinstance(profile["inactive_joint_positions"], dict):
        raise ValueError("inactive_joint_positions must be a joint-name to position mapping")
    for name, value in profile["inactive_joint_positions"].items():
        if not isinstance(name, str) or not name or name in names or value is None or not np.isfinite(float(value)):
            raise ValueError("Inactive joint mapping overlaps active joints or is invalid")
    passive = profile.get("passive_joint_positions", {})
    if not isinstance(passive, dict):
        raise ValueError("passive_joint_positions must be a mapping")
    for name, value in passive.items():
        if not isinstance(name, str) or not name or name in names or name in profile["inactive_joint_positions"] or value is None or not np.isfinite(float(value)):
            raise ValueError("Passive/mimic DOFs must be finite, distinct and not actively held")
    gripper_spec = profile["gripper"]
    if not isinstance(gripper_spec, dict):
        raise ValueError("gripper must be an asset-specific calibration mapping")
    if gripper_spec.get("type") != "angular":
        raise ValueError("G2 requires calibrated angular, not metre-width, gripper control")
    gripper_fields = ["joint_name", "closed_angle_rad", "open_angle_rad", "lower_limit_rad", "upper_limit_rad", "max_effort"]
    missing = [field for field in gripper_fields if gripper_spec.get(field) is None]
    if missing:
        raise ValueError("Gripper needs asset calibration: " + ", ".join(missing))
    if not isinstance(gripper_spec["joint_name"], str) or not gripper_spec["joint_name"]:
        raise ValueError("gripper joint_name must be a nonempty string")
    gripper = AngularGripper(**{key: gripper_spec[key] for key in gripper_fields})
    if gripper.joint_name in names or gripper.joint_name in profile["inactive_joint_positions"] or gripper.joint_name in passive:
        raise ValueError("Gripper joint cannot also be an active arm joint or an inactive held joint")
    fingers = gripper_spec.get("contact_link_paths")
    if not isinstance(fingers, list) or len(fingers) < 2 or any(not isinstance(finger, str) for finger in fingers) or len(set(fingers)) != len(fingers):
        raise ValueError("At least two distinct physical finger link paths are required")
    for finger in fingers:
        _link(finger, root, "contact_link_paths")
    pairs = profile.get("allowed_contact_pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("allowed_contact_pairs must be a list of [robot_link,environment_prim]")
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("allowed_contact_pairs require two entries per pair")
        _link(pair[0], root, "allowed contact robot link")
        _prim_path(pair[1], "allowed contact environment path")
    rigid_transform(profile["T_E_G"])
    for key in ("robot_usd", "curobo_config"):
        if not isinstance(profile[key], str) or not profile[key]:
            raise ValueError(f"{key} must be a nonempty file path")
        target = Path(profile[key]).expanduser()
        if not target.is_absolute():
            target = path.parent / target
        target = target.resolve(strict=True)
        if not target.is_file():
            raise ValueError(f"{key} is not a file")
        profile[key] = str(target)
    expected_digest = profile["asset_sha256"]
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected_digest):
        raise ValueError("asset_sha256 must be exactly 64 hexadecimal characters")
    digest = _file_digest(profile["robot_usd"])
    if digest != expected_digest.lower():
        raise ValueError("Robot USD SHA256 does not match the supplied asset manifest")
    if not isinstance(profile["license_source"], str) or not profile["license_source"].strip():
        raise ValueError("Explicit nonempty asset license source is required")
    profile["asset_sha256"] = digest
    profile["curobo_config_sha256"] = _file_digest(profile["curobo_config"])
    profile["profile_file"] = str(path)
    profile["profile_sha256"] = _file_digest(path)
    return profile, limits, gripper
