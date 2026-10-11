"""Small, explicit reproduction records; no asset downloads or simulated results."""
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .config import write_json


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_value(value):
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def environment_record():
    versions = {}
    for name in ("geniesim-features", "numpy", "usd-core", "isaacsim", "torch", "nvidia-curobo", "PyYAML"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=3, check=False)
        device = {"exit_code": gpu.returncode, "devices": gpu.stdout.strip().splitlines()}
    except (OSError, subprocess.TimeoutExpired) as error:
        device = {"available": False, "reason": type(error).__name__}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": versions, "gpu_query": device, "captured_wall_time": time.time()}


def scene_dependencies(scene):
    """Freeze the actual composed dependency closure, not only the entry USDA."""
    from pxr import Sdf, UsdUtils
    layers, assets, missing = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(scene)))
    if missing:
        raise ValueError(f"Scene contains unresolved dependencies: {list(missing)}")
    paths = {str(Path(layer.realPath).resolve(strict=True)) for layer in layers if layer.realPath}
    paths.update(str(Path(asset).resolve(strict=True)) for asset in assets)
    return {path: sha256(path) for path in sorted(paths)}


def configuration_record(args, scene, sensors, *, profile=None, task=None):
    sources = {}
    for name in ("config", "robot_profile", "task"):
        value = getattr(args, name, None)
        if value is not None:
            path = Path(value).resolve(strict=True)
            sources[name] = {"path": str(path), "sha256": sha256(path)}
    # Pure Python preflight: do not import pxr before SimulationApp chooses USD.
    dependencies = {str(Path(scene).resolve()): sha256(scene)}
    for name in ("bindings.json", "scene_manifest.json"):
        path = Path(scene).parent / name
        dependencies[str(path)] = sha256(path)
    if profile:
        dependencies[profile["robot_usd"]] = sha256(profile["robot_usd"])
        dependencies[profile["curobo_config"]] = sha256(profile["curobo_config"])
    return json_value({"schema_version": 1, "scene_directory": str(Path(scene).parent),
                       "physics_hz": args.physics_hz, "control_hz": args.control_hz,
                       "camera_config": sensors, "robot_profile": profile, "task": task,
                       "max_intervals": getattr(args, "max_intervals", None),
                       "sources": sources, "dependency_sha256": dependencies})


def check_reproduction_inputs(configuration):
    for path, digest in configuration["dependency_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Replay dependency digest mismatch: {path}")
    for record in configuration["sources"].values():
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Replay configuration digest mismatch: {record['path']}")


def load_randomization(scene_dir):
    path = Path(scene_dir) / "randomization.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"enabled": False}


def apply_pending_cameras(session, randomization):
    """Apply sampled optics AFTER camera authoring, BEFORE PhysX initialization."""
    from copy import deepcopy
    from .domain_randomization import DomainRandomizer, UsdRandomizationAdapter
    record = deepcopy(randomization)
    pending = record.pop("pending_camera_components", {})
    if pending:
        applied = DomainRandomizer().apply({**record, "components": pending}, UsdRandomizationAdapter(session.stage))
        record["components"].update(applied["components"])
        record.update(applied=True, application_phase="usd_then_pre_initialize_cameras",
                      camera_readback_verified=True)
    return record


def archive_configuration(output, configuration, randomization):
    output = Path(output)
    write_json(output / "configuration.json", configuration)
    write_json(output / "randomization-request.json", randomization)
    write_json(output / "environment.json", environment_record())


def archive_asset_dependencies(output, scene, profile=None):
    """Call only after SimulationApp startup, while its USD runtime is selected."""
    dependencies = scene_dependencies(scene)
    if profile:
        dependencies.update(scene_dependencies(profile["robot_usd"]))
    write_json(Path(output) / "asset_dependencies.json", dependencies)
    return dependencies
