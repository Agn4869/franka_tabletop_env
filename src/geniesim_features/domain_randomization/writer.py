"""Static observation export only. No actions and no generated stand-in images."""
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from uuid import uuid4

import numpy as np
from PIL import Image

from ..sensors import SensorBundle, depth_valid_mask, depth_to_points, visible_bboxes, write_ply
from ..sensors.models import safe_identifier


def _json_value(value):
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    payload = json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")


class StaticSampleWriter:
    def __init__(self, output_dir, *, export_ply=False):
        self.output_dir = Path(output_dir).absolute()
        if any(path.is_symlink() for path in (self.output_dir, *self.output_dir.parents)):
            raise ValueError("output directory must not traverse symlinks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_ply = bool(export_ply)

    def write(self, sensors: SensorBundle, scene: dict, randomization: dict, sample_id=None) -> Path:
        if not isinstance(sensors, SensorBundle):
            raise TypeError("sensors must be SensorBundle")
        if any(path.is_symlink() for path in (self.output_dir, *self.output_dir.parents)):
            raise ValueError("output directory must not traverse symlinks")
        sample_id = safe_identifier(sample_id or f"sample_{sensors.frame_index:06d}_{uuid4().hex[:12]}")
        target = self.output_dir / sample_id
        # mkdir is the exclusive reservation; existing complete/partial samples
        # and symlinks cannot be overwritten by this writer.
        target.mkdir(exist_ok=False)
        _write_json(target / "partial.json", {"sample_id": sample_id, "data_status": "partial"})
        annotations = {"physics_step": int(sensors.physics_step), "simulation_time": float(sensors.simulation_time),
                       "frame_index": int(sensors.frame_index), "scene": scene, "cameras": {}}
        object_ids = set()
        if isinstance(scene, Mapping):
            for key in ("object_poses", "bindings"):
                if isinstance(scene.get(key), Mapping):
                    object_ids.update(scene[key])
        for camera_id, frame in sensors.frames.items():
            safe_identifier(camera_id)
            camera_dir = target / "cameras" / camera_id
            camera_dir.mkdir(parents=True, exist_ok=False)
            with (camera_dir / "rgb.png").open("xb") as handle:
                Image.fromarray(frame.rgb).save(handle, format="PNG")
            mask = depth_valid_mask(frame.depth, frame.calibration.near, frame.calibration.far)
            for filename, value in (("depth.npy", frame.depth), ("instance_ids.npy", frame.instance_ids), ("valid_mask.npy", mask)):
                with (camera_dir / filename).open("xb") as handle:
                    np.save(handle, value, allow_pickle=False)
            _write_json(camera_dir / "calibration.json", frame.calibration.to_dict())
            _write_json(camera_dir / "instance_mapping.json", frame.id_to_object)
            _write_json(camera_dir / "renderer_instance_mapping.json", {
                "reserved_instance_ids": frame.reserved_instance_ids, "raw_id_to_label": frame.raw_id_to_label})
            annotations["cameras"][camera_id] = {"visible_bboxes_xyxy_exclusive": visible_bboxes(frame.instance_ids, frame.id_to_object, object_ids),
                                                    "valid_depth_pixels": int(mask.sum()), "physics_step": int(frame.physics_step)}
            if self.export_ply:
                write_ply(camera_dir / "points_world.ply", depth_to_points(frame.depth, frame.calibration), frame.rgb[mask])
        _write_json(target / "randomization.json", randomization)
        _write_json(target / "annotations.json", annotations)
        artifacts = {}
        for path in sorted(target.rglob("*")):
            if path.is_file() and path.name != "partial.json":
                data = path.read_bytes()
                artifacts[path.relative_to(target).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        _write_json(target / "manifest.json", {"schema_version": 1, "sample_id": sample_id, "data_status": "complete",
                                                "sample_type": "static_observation", "physics_step": int(sensors.physics_step),
                                                "camera_ids": list(sensors.frames), "files": artifacts})
        (target / "partial.json").unlink()
        return target
