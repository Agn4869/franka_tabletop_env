"""Synchronous, bounded-memory episode persistence with explicit partial state.

This writer intentionally has no worker thread: append returns only when payloads
and both JSONL records have been flushed/fsynced. The host therefore backpressures
at the control boundary, and close has no unjoined background writes. It does not
claim atomic multi-file transactions: any failed write leaves a partial episode.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

import numpy as np

from ..sensors import CameraCalibration, SensorFrame, depth_valid_mask


class RecordingError(RuntimeError):
    pass


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _items(value):
    if is_dataclass(value) and not isinstance(value, type):
        return ((field.name, getattr(value, field.name)) for field in fields(value))
    if isinstance(value, Mapping):
        return value.items()
    return None


def _get(value, *names, default=None):
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def immutable_snapshot(value, max_bytes=256 * 1024 * 1024):
    """Copy buffers before persistence; handles frozen dataclasses/MappingProxyType.

    Budget includes array payload and a conservative overhead per metadata node.
    No Stage, GPU buffer, custom live object or object-dtype array is accepted.
    """
    budget = [0]

    def copy(item, depth=0):
        if depth > 32:
            raise ValueError("record nesting exceeds 32 levels")
        budget[0] += 64
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject or item.dtype.kind not in "biufc":
                raise ValueError("only numeric, non-object CPU ndarray payloads are supported")
            budget[0] += item.nbytes
            if budget[0] > max_bytes:
                raise ValueError("record exceeds memory budget")
            return np.frombuffer(item.tobytes(order="C"), dtype=item.dtype).reshape(item.shape)
        pairs = _items(item)
        if pairs is not None:
            result = {}
            for key, child in pairs:
                if not isinstance(key, (str, int, np.integer)) or isinstance(key, bool):
                    raise ValueError("metadata keys must be strings or integer IDs")
                key = str(key)
                if key in result:
                    raise ValueError("metadata keys collide after JSON conversion")
                budget[0] += len(key.encode("utf-8"))
                result[key] = copy(child, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return tuple(copy(child, depth + 1) for child in item)
        if isinstance(item, np.generic):
            return copy(item.item(), depth + 1)
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("nonfinite scalar metadata")
            if isinstance(item, str):
                budget[0] += len(item.encode("utf-8"))
            if budget[0] > max_bytes:
                raise ValueError("record exceeds memory budget")
            return item
        raise TypeError(f"unsupported live record object: {type(item).__name__}")

    result = copy(value)
    if budget[0] > max_bytes:
        raise ValueError("record exceeds memory budget")
    return result


class EpisodeWriter:
    def __init__(self, output_dir, *, max_observation_bytes=256 * 1024 * 1024, max_actions=100_000):
        self.output_dir = Path(output_dir)
        if max_observation_bytes <= 0 or max_actions <= 0:
            raise ValueError("positive recording bounds required")
        self.max_observation_bytes = int(max_observation_bytes)
        self.max_actions = int(max_actions)
        self._state = "new"
        self._actions = self._observations = None
        self._manifest = {}
        self._last_step = self._last_time = None
        self._camera_ids = None
        self._error = None

    @property
    def manifest(self):
        return json.loads(json.dumps(self._manifest))

    @staticmethod
    def _json(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    def _sync_directory(self):
        fd = os.open(self.output_dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _save_manifest(self):
        temporary = self.output_dir / f".manifest-{uuid.uuid4().hex}.tmp"
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(self._json(self._manifest) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.output_dir / "manifest.json")
        self._sync_directory()

    def _close_streams(self):
        first_error = None
        for name in ("_actions", "_observations"):
            stream = getattr(self, name)
            if stream is not None:
                try:
                    stream.close()
                except Exception as error:
                    first_error = first_error or error
                setattr(self, name, None)
        if first_error:
            raise first_error

    def _fail(self, error):
        self._error = error
        self._state = "failed"
        self._manifest["data_status"] = "partial"
        self._manifest["recording_error"] = f"{type(error).__name__}:{error}"
        self._manifest["outcome"] = "failure"
        self._manifest["failure_reason"] = f"recording_error:{type(error).__name__}:{error}"
        # Best effort metadata repair; never mask the original disk/validation error.
        try:
            self._close_streams()
        except Exception:
            pass
        try:
            self._collect_files()
            self._save_manifest()
        except Exception:
            pass
        raise RecordingError(f"episode remains partial: {error}") from error

    def _collect_files(self):
        """Commit payload/configuration digests without following symlinks.

        The manifest is the local trust anchor, not a cryptographic signature.
        Hashes detect payload corruption, not deliberate replacement of the manifest.
        """
        files = {}
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError("episode must not contain symlinks")
            if not path.is_file() or path == self.output_dir / "manifest.json" or path.name.startswith(".manifest-"):
                continue
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            files[path.relative_to(self.output_dir).as_posix()] = {
                "sha256": file_digest(path), "size_bytes": path.stat().st_size}
        for directory in [self.output_dir, *(path for path in self.output_dir.rglob("*") if path.is_dir())]:
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        self._manifest["integrity_version"] = 1
        self._manifest["files"] = files

    def reserve(self, manifest):
        """Reserve a new partial episode before expensive engine initialization."""
        if self._state != "new":
            raise RecordingError("reserve can only be called once")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._state = "reserved"
        try:
            metadata = immutable_snapshot(manifest, self.max_observation_bytes)
            if not isinstance(metadata, dict):
                raise ValueError("manifest must be a mapping or dataclass")
            self._json(metadata)
            self._manifest = {**metadata, "schema_version": 1, "episode_id": metadata.get("episode_id", self.output_dir.name),
                              "data_status": "partial", "outcome": None, "failure_reason": None,
                              "action_count": 0, "observation_count": 0, "writer_mode": "synchronous",
                              "integrity_version": 1, "files": {}}
            self._save_manifest()
        except Exception as error:
            self._fail(error)

    def fail(self, error, details=None):
        """Persist an exact host failure, keeping partial separate from outcome.

        Disk failure is propagated; durability cannot be promised on a failed disk.
        Already-failed recorder errors are retained alongside the host error.
        """
        if self._state not in ("reserved", "initializing", "active", "failed"):
            raise RecordingError(f"writer is {self._state}; cannot record a host failure")
        message = f"{type(error).__name__}:{error}" if isinstance(error, BaseException) else str(error)
        if not message:
            raise ValueError("host failure needs a nonempty reason")
        self._state = "failed"
        self._error = error if isinstance(error, BaseException) else RuntimeError(message)
        self._manifest.update(data_status="partial", outcome="failure", failure_reason=message, host_error=message)
        try:
            if details is not None:
                self._manifest["failure_details"] = immutable_snapshot(details, self.max_observation_bytes)
            self._close_streams()
            self._collect_files()
            self._save_manifest()
        except Exception as disk_error:
            raise RecordingError(f"could not persist host failure {message}: {disk_error}") from disk_error

    def _check_active(self):
        if self._state != "active":
            raise RecordingError(f"writer is {self._state}; cannot append/finish") from self._error

    def _check_observation(self, observation, *, initial=False):
        step = _get(observation, "step_index", "physics_step")
        timestamp = _get(observation, "sim_time", "simulation_time")
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("observation requires a nonnegative physical step")
        if not isinstance(timestamp, (int, float, np.number)) or not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("observation requires finite simulation time")
        if not initial:
            if step <= self._last_step or timestamp <= self._last_time:
                raise ValueError("observation time/physical step did not advance")
            physics_hz, control_hz = self._manifest.get("physics_hz"), self._manifest.get("control_hz")
            if physics_hz is not None and control_hz is not None:
                ratio = physics_hz / control_hz
                if not np.isclose(ratio, round(ratio)) or step - self._last_step != round(ratio):
                    raise ValueError("action interval has an incorrect physical step count")
                if not np.isclose(timestamp - self._last_time, 1 / control_hz, atol=1e-7, rtol=1e-6):
                    raise ValueError("action interval has an incorrect simulation duration")
        sensors = _get(observation, "sensors")
        ids = ()
        if sensors is not None:
            sensor_step = _get(sensors, "physics_step", "step_index")
            sensor_time = _get(sensors, "simulation_time", "sim_time")
            if sensor_step != step or sensor_time is None or not np.isclose(sensor_time, timestamp, atol=1e-8, rtol=0):
                raise ValueError("sensor bundle is not aligned to observation")
            frames = _get(sensors, "frames")
            if not isinstance(frames, Mapping):
                raise ValueError("sensor frames must be keyed by camera ID")
            ids = tuple(sorted(frames))
            for camera_id, frame in frames.items():
                if not isinstance(camera_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", camera_id):
                    raise ValueError("unsafe camera ID")
                if _get(frame, "camera_id") != camera_id or _get(frame, "physics_step", "step_index") != step:
                    raise ValueError("camera identity/physical step mismatch")
                depth = _get(frame, "depth")
                calibration = _get(frame, "calibration")
                if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.dtype.kind not in "fiu":
                    raise ValueError("camera depth must be numeric HxW")
                near, far = _get(calibration, "near", default=.01), _get(calibration, "far", default=100.)
                camera = CameraCalibration(_get(calibration, "width"), _get(calibration, "height"),
                                           _get(calibration, "K"), _get(calibration, "T_world_optical"), near, far)
                # Public callers may pass dicts, not just previously validated
                # SensorFrame instances. Validate the complete image contract.
                def id_map(name, default):
                    value = _get(frame, name, default=default)
                    if not isinstance(value, Mapping):
                        raise ValueError(f"{name} must be a renderer-ID mapping")
                    result = {}
                    for key, label in value.items():
                        integer = int(key)
                        if str(integer) != str(key) or integer in result:
                            raise ValueError("noncanonical/duplicate renderer ID")
                        result[integer] = label
                    return result
                checked = SensorFrame(camera_id, step, _get(frame, "rgb"), depth,
                                      _get(frame, "instance_ids"), camera, id_map("id_to_object", {}),
                                      id_map("reserved_instance_ids", {"0": "BACKGROUND"}), id_map("raw_id_to_label", {}))
                frame["rgb"], frame["depth"], frame["instance_ids"] = checked.rgb, checked.depth, checked.instance_ids
                frame["reserved_instance_ids"] = {str(key): value for key, value in checked.reserved_instance_ids.items()}
                frame["depth_semantics"] = "distance_to_image_plane_m"
                # This is a derived validity annotation, not imputed depth.
                frame["valid_mask"] = depth_valid_mask(checked.depth, near, far)
        expected = tuple(sorted(self._manifest.get("camera_ids", ids)))
        if ids != expected or (self._camera_ids is not None and ids != self._camera_ids):
            raise ValueError("camera set changed or expected cameras are missing")
        self._camera_ids = ids
        return int(step), float(timestamp)

    def _payload_path(self, group, index, trail, extension):
        # Sensor dataclass layout: sensors.frames.<camera_id>.<payload>.
        if len(trail) >= 4 and trail[:2] == ("sensors", "frames"):
            camera_id = trail[2]
            name = hashlib.sha256(json.dumps(trail[3:]).encode()).hexdigest()[:16]
            leaf = re.sub(r"[^A-Za-z0-9_-]", "_", trail[-1])[:48]
            path = Path("cameras") / camera_id / f"{index:06d}_{leaf}_{name}{extension}"
        else:
            name = hashlib.sha256(json.dumps(trail).encode()).hexdigest()
            path = Path("arrays") / group / f"{index:06d}_{name}{extension}"
        return path

    def _persist(self, value, group, index, trail=()):
        if isinstance(value, np.ndarray):
            is_rgb = bool(trail and trail[-1] == "rgb")
            if is_rgb and (value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] not in (3, 4)):
                raise ValueError("RGB arrays must be uint8 HxWx3/4")
            relative = self._payload_path(group, index, trail, ".png" if is_rgb else ".npy")
            target = self.output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents overwriting any already emitted artifact.
            with target.open("xb") as stream:
                if is_rgb:
                    from PIL import Image
                    Image.fromarray(value).save(stream, format="PNG")
                else:
                    np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            return {"path": relative.as_posix(), "dtype": str(value.dtype), "shape": list(value.shape),
                    "encoding": "png" if is_rgb else "npy"}
        if isinstance(value, dict):
            return {key: self._persist(child, group, index, trail + (key,)) for key, child in value.items()}
        if isinstance(value, tuple):
            return [self._persist(child, group, index, trail + (str(i),)) for i, child in enumerate(value)]
        return value

    def _write_line(self, stream, value):
        stream.write(self._json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    def begin(self, manifest, initial_observation):
        if self._state not in ("new", "reserved"):
            raise RecordingError("begin can only be called once")
        if self._state == "new":
            self.reserve(manifest)
        self._state = "initializing"
        try:
            metadata = immutable_snapshot(manifest, self.max_observation_bytes)
            if not isinstance(metadata, dict):
                raise ValueError("manifest must be a mapping or dataclass")
            # Validate JSON-only manifest before producing any complete marker.
            self._json(metadata)
            for field in ("physics_hz", "control_hz", "camera_hz"):
                if field in metadata and (not np.isfinite(metadata[field]) or metadata[field] <= 0):
                    raise ValueError(f"invalid {field}")
            self._manifest = {**self._manifest, **metadata, "schema_version": 1, "episode_id": metadata.get("episode_id", self.output_dir.name),
                              "data_status": "partial", "outcome": None, "failure_reason": None,
                              "action_count": 0, "observation_count": 0, "writer_mode": "synchronous"}
            self._save_manifest()
            self._actions = (self.output_dir / "actions.jsonl").open("x", encoding="utf-8")
            self._observations = (self.output_dir / "observations.jsonl").open("x", encoding="utf-8")
            snapshot = immutable_snapshot(initial_observation, self.max_observation_bytes)
            step, timestamp = self._check_observation(snapshot, initial=True)
            self._manifest["camera_ids"] = list(self._camera_ids)
            record = self._persist(snapshot, "observations", 0)
            self._write_line(self._observations, {"observation_index": 0, "observation": record})
            self._manifest["observation_count"] = 1
            self._last_step, self._last_time = step, timestamp
            self._save_manifest()
            self._state = "active"
        except Exception as error:
            self._fail(error)

    def append(self, action, next_observation):
        self._check_active()
        try:
            if self._manifest["action_count"] >= self.max_actions:
                raise ValueError("episode action bound exceeded")
            action = immutable_snapshot(action, self.max_observation_bytes)
            def validate_action(value):
                if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                    raise ValueError("action contains nonfinite values")
                if isinstance(value, dict):
                    for child in value.values():
                        validate_action(child)
                elif isinstance(value, tuple):
                    for child in value:
                        validate_action(child)
            validate_action(action)
            observation = immutable_snapshot(next_observation, self.max_observation_bytes)
            step, timestamp = self._check_observation(observation)
            index = self._manifest["action_count"]
            action_record = self._persist(action, "actions", index)
            observation_record = self._persist(observation, "observations", index + 1)
            self._write_line(self._actions, {"action_index": index, "from_observation": index,
                                             "to_observation": index + 1, "start_step": self._last_step,
                                             "end_step": step, "action": action_record})
            self._manifest["action_count"] += 1
            self._write_line(self._observations, {"observation_index": index + 1, "observation": observation_record})
            self._manifest["observation_count"] += 1
            self._last_step, self._last_time = step, timestamp
            self._save_manifest()
        except Exception as error:
            self._fail(error)

    def finish(self, outcome):
        self._check_active()
        try:
            result = immutable_snapshot(outcome, self.max_observation_bytes)
            if isinstance(result, str):
                result = {"outcome": result}
            if not isinstance(result, dict):
                raise ValueError("outcome must be success/failure or mapping")
            if "done" in result:
                if not result["done"]:
                    raise ValueError("cannot finish an ongoing task")
                result["outcome"] = "success" if result.get("success") else "failure"
            value = result.get("outcome")
            if value not in ("success", "failure"):
                raise ValueError("task outcome must be success or failure")
            if value == "success" and result.get("failure_reason"):
                raise ValueError("success cannot carry a failure reason")
            if value == "failure" and not result.get("failure_reason"):
                raise ValueError("failure outcome needs a failure_reason")
            if self._manifest["observation_count"] != self._manifest["action_count"] + 1:
                raise ValueError("N+1 observation / N action invariant violated")
            self._close_streams()
            self._collect_files()
            self._manifest.update({"outcome": value, "failure_reason": result.get("failure_reason"),
                                   "task_result": result, "data_status": "complete"})
            self._save_manifest()
            self._state = "complete"
            return self.output_dir
        except Exception as error:
            self._fail(error)

    def close(self):
        if self._state in ("new", "complete", "closed", "failed"):
            return
        try:
            self._close_streams()
            self._manifest["data_status"] = "partial"
            self._manifest["recording_error"] = "writer_closed_before_finish"
            self._collect_files()
            self._save_manifest()
            self._state = "closed"
        except Exception as error:
            self._fail(error)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception is not None and self._state in ("reserved", "initializing", "active"):
            self.fail(exception)
        self.close()
        return False
