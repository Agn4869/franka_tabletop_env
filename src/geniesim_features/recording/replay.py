"""Verified, bounded, lazy episode reading and recorded-observation display.

This module never starts a simulator. An HTML replay displays saved observations;
it is not physics re-execution or evidence that a recorded task succeeded.
"""
from __future__ import annotations

import html
import json
from pathlib import Path, PurePosixPath
import shutil

import numpy as np

from .episode import file_digest


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject)


def _integer(value, label, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"invalid {label}")
    return value


class EpisodeReader:
    def __init__(self, episode_path, *, require_complete=True, verify_hashes=True,
                 max_records=100_001, max_payload_bytes=256 * 1024 * 1024):
        self.root = Path(episode_path).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("episode must be a directory")
        self.max_records = _integer(max_records, "max_records", 10_000_001)
        self.max_payload_bytes = _integer(max_payload_bytes, "max_payload_bytes", 4 * 1024**3)
        if not self.max_records or not self.max_payload_bytes:
            raise ValueError("positive reader resource bounds required")
        self.verify_hashes = bool(verify_hashes)
        self._fingerprints = {}
        manifest_path = self._path("manifest.json")
        if manifest_path.stat().st_size > 32 * 1024**2:
            raise ValueError("manifest exceeds reader metadata budget")
        self._manifest = _json(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(self._manifest, dict) or self._manifest.get("schema_version") != 1:
            raise ValueError("unsupported episode manifest")
        status = self._manifest.get("data_status")
        if status not in ("complete", "partial") or (require_complete and status != "complete"):
            raise ValueError("replay requires a complete episode")
        self.action_count = _integer(self._manifest.get("action_count"), "action_count", self.max_records - 1)
        self.observation_count = _integer(self._manifest.get("observation_count"), "observation_count", self.max_records)
        if status == "complete":
            if self.observation_count != self.action_count + 1:
                raise ValueError("episode violates N+1 observations / N actions")
            outcome = self._manifest.get("outcome")
            if outcome not in ("success", "failure") or (outcome == "failure" and not self._manifest.get("failure_reason")):
                raise ValueError("complete episode needs an explicit task outcome")
            if outcome == "success" and self._manifest.get("failure_reason"):
                raise ValueError("success episode carries a failure reason")
        self._files = self._manifest.get("files")
        if self._manifest.get("integrity_version") != 1 or not isinstance(self._files, dict):
            raise ValueError("episode lacks committed file integrity metadata")
        if len(self._files) > self.max_records * 200:
            raise ValueError("episode file index exceeds resource bound")
        for name, metadata in self._files.items():
            if not isinstance(metadata, dict) or set(metadata) != {"sha256", "size_bytes"}:
                raise ValueError("invalid file integrity record")
            digest = metadata["sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("invalid SHA256")
            _integer(metadata["size_bytes"], "file size", 1024**5)
            self._verify_file(name)
        self._observation_offsets, self._observation_times = self._index_observations()
        self._action_offsets = self._index_actions()

    @property
    def manifest(self):
        return json.loads(json.dumps(self._manifest))

    def _path(self, name):
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError("invalid episode-relative path")
        path = PurePosixPath(name)
        if path.is_absolute() or any(part in (".", "..") for part in path.parts) or path.as_posix() != name:
            raise ValueError("episode path traversal or noncanonical path")
        result = self.root
        for part in path.parts:
            result = result / part
            if result.is_symlink():
                raise ValueError("episode symlinks are not allowed")
        if not result.is_file() or not result.resolve().is_relative_to(self.root):
            raise ValueError(f"missing or escaped episode file: {name}")
        return result

    @staticmethod
    def _fingerprint(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def _verify_file(self, name):
        if name not in self._files or name == "manifest.json":
            raise ValueError(f"unindexed episode payload: {name}")
        path = self._path(name)
        metadata = self._files[name]
        if path.stat().st_size != metadata["size_bytes"]:
            raise ValueError(f"episode file size mismatch: {name}")
        if self.verify_hashes and file_digest(path) != metadata["sha256"]:
            raise ValueError(f"episode file hash mismatch: {name}")
        self._fingerprints[name] = self._fingerprint(path)
        return path

    def _unchanged_path(self, name):
        path = self._path(name)
        if name not in self._fingerprints or self._fingerprint(path) != self._fingerprints[name]:
            raise ValueError(f"episode file changed after validation: {name}")
        return path

    def read_json(self, name):
        """Read a committed ancillary configuration, not an unindexed new file."""
        path = self._unchanged_path(name)
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError("configuration exceeds reader metadata budget")
        result = _json(path.read_text(encoding="utf-8"))
        self._unchanged_path(name)
        if not isinstance(result, dict):
            raise ValueError("configuration must be a JSON mapping")
        return result

    def _rows(self, name, expected):
        if name not in self._files:
            if expected == 0 and self._manifest["data_status"] == "partial":
                return
            raise ValueError(f"missing episode index: {name}")
        with self._unchanged_path(name).open("rb") as stream:
            index = 0
            while True:
                offset = stream.tell()
                line = stream.readline(4 * 1024**2 + 1)
                if not line:
                    break
                if len(line) > 4 * 1024**2 or not line.endswith(b"\n"):
                    raise ValueError("oversized or incomplete JSONL record")
                if index >= expected:
                    raise ValueError("JSONL count differs from manifest")
                row = _json(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL rows must be mappings")
                yield index, offset, row
                index += 1
            if index != expected:
                raise ValueError("JSONL count differs from manifest")

    def _validate_refs(self, value, budget=None, depth=0):
        budget = [0] if budget is None else budget
        if depth > 32:
            raise ValueError("record nesting exceeds reader bound")
        if isinstance(value, dict):
            if "encoding" in value and "path" in value:
                if set(value) != {"path", "dtype", "shape", "encoding"} or value["encoding"] not in ("png", "npy"):
                    raise ValueError("invalid payload descriptor")
                if value["path"] not in self._files:
                    raise ValueError("unindexed payload reference")
                self._unchanged_path(value["path"])
                dtype = np.dtype(value["dtype"])
                shape = value["shape"]
                if dtype.hasobject or dtype.kind not in "biufc" or not isinstance(shape, list) or len(shape) > 8:
                    raise ValueError("invalid numeric payload dtype/shape")
                count = 1
                for size in shape:
                    count *= _integer(size, "payload dimension", self.max_payload_bytes)
                budget[0] += count * dtype.itemsize
                if budget[0] > self.max_payload_bytes:
                    raise ValueError("observation exceeds reader payload budget")
            else:
                for child in value.values():
                    self._validate_refs(child, budget, depth + 1)
        elif isinstance(value, list):
            for child in value:
                self._validate_refs(child, budget, depth + 1)

    def _index_observations(self):
        offsets, times = [], []
        cameras = self._manifest.get("camera_ids", [])
        for index, offset, row in self._rows("observations.jsonl", self.observation_count):
            if row.get("observation_index") != index or not isinstance(row.get("observation"), dict):
                raise ValueError("observation index mismatch")
            observation = row["observation"]
            step = _integer(observation.get("step_index", observation.get("physics_step")), "physics step", 2**63 - 1)
            timestamp = observation.get("sim_time", observation.get("simulation_time"))
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp) or timestamp < 0:
                raise ValueError("invalid observation simulation time")
            if times and (step <= times[-1][0] or timestamp <= times[-1][1]):
                raise ValueError("observation times must strictly advance")
            if times and self._manifest.get("physics_hz") and self._manifest.get("control_hz"):
                ratio = self._manifest["physics_hz"] / self._manifest["control_hz"]
                if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9) or step-times[-1][0] != round(ratio):
                    raise ValueError("physical/control step mismatch")
                if not np.isclose(timestamp-times[-1][1], 1/self._manifest["control_hz"], rtol=1e-6, atol=1e-7):
                    raise ValueError("control interval time mismatch")
            sensors = observation.get("sensors")
            if sensors is not None:
                if sensors.get("physics_step") != step or not np.isclose(sensors.get("simulation_time", -1), timestamp, rtol=0, atol=1e-8):
                    raise ValueError("sensor/robot observation not aligned")
                frames = sensors.get("frames", {})
                if sorted(frames) != sorted(cameras):
                    raise ValueError("episode camera set changed")
                for name, frame in frames.items():
                    if frame.get("camera_id") != name or frame.get("physics_step") != step:
                        raise ValueError("camera identity/physical step mismatch")
            elif cameras:
                raise ValueError("expected cameras missing")
            self._validate_refs(observation)
            offsets.append(offset)
            times.append((step, float(timestamp)))
        return offsets, times

    def _index_actions(self):
        offsets = []
        for index, offset, row in self._rows("actions.jsonl", self.action_count):
            if row.get("action_index") != index or row.get("from_observation") != index or row.get("to_observation") != index+1:
                raise ValueError("action-to-observation index mismatch")
            if index + 1 >= len(self._observation_times):
                raise ValueError("action is missing its next observation")
            if row.get("start_step") != self._observation_times[index][0] or row.get("end_step") != self._observation_times[index+1][0]:
                raise ValueError("action-to-observation physical step mismatch")
            if not isinstance(row.get("action"), dict):
                raise ValueError("action must be a mapping")
            self._validate_refs(row["action"])
            offsets.append(offset)
        return offsets

    def _load_arrays(self, value):
        if isinstance(value, dict):
            if "encoding" in value and "path" in value:
                path = self._unchanged_path(value["path"])
                if value["encoding"] == "png":
                    from PIL import Image
                    with Image.open(path) as image:
                        shape = value["shape"]
                        if len(shape) != 3 or list(image.size) != [shape[1], shape[0]]:
                            raise ValueError("image size differs from payload descriptor")
                        array = np.asarray(image)
                else:
                    array = np.load(path, allow_pickle=False, mmap_mode="r")
                if array.dtype != np.dtype(value["dtype"]) or list(array.shape) != value["shape"] or array.nbytes > self.max_payload_bytes:
                    raise ValueError("payload dtype/shape differs from descriptor")
                return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
            return {key: self._load_arrays(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self._load_arrays(child) for child in value]
        return value

    def _record(self, name, offsets, index, key, load_arrays):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(offsets):
            raise IndexError(index)
        with self._unchanged_path(name).open("rb") as stream:
            stream.seek(offsets[index])
            result = _json(stream.readline(4 * 1024**2 + 1))[key]
        return self._load_arrays(result) if load_arrays else result

    def observation(self, index, *, load_arrays=True):
        return self._record("observations.jsonl", self._observation_offsets, index, "observation", load_arrays)

    def action(self, index, *, load_arrays=True):
        return self._record("actions.jsonl", self._action_offsets, index, "action", load_arrays)

    def iter_transitions(self):
        observation = self.observation(0)
        for index in range(self.action_count):
            next_observation = self.observation(index+1)
            yield observation, self.action(index), next_observation
            observation = next_observation

    def export_observation_replay(self, output_dir, *, max_frames=1000):
        """Export standalone HTML with copied RGB frames and recorded summaries."""
        _integer(max_frames, "max_frames", self.max_records)
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        image_dir = output / "images"
        image_dir.mkdir()
        count = min(self.observation_count, max_frames)
        path = output / "index.html"
        with path.open("x", encoding="utf-8") as stream:
            stream.write('<!doctype html><html lang="en"><meta charset="utf-8"><title>Recorded observations</title>'
                         '<style>body{font:16px sans-serif;max-width:1200px;margin:auto}img{max-width:45%;height:auto}'
                         'section{border-top:1px solid #aaa;padding:1em}pre{white-space:pre-wrap}</style><body>'
                         '<h1>Recorded observation replay</h1><p>Saved data only. No physics re-execution.</p>')
            stream.write(f'<p>Source mode: {html.escape(str(self._manifest.get("execution_mode", "unspecified")))}</p>')
            stream.write('<nav><button id="prev">Previous</button> <button id="play">Play / pause</button> '
                         '<button id="next">Next</button> <input id="frame" type="range" min="0" '
                         f'max="{max(0, count-1)}" value="0" aria-label="Observation index"> '
                         '<output id="position"></output></nav>')
            stream.write(f'<p>Showing {count}/{self.observation_count} observations. Outcome: '
                         f'{html.escape(str(self._manifest.get("outcome")))}; status: {html.escape(self._manifest["data_status"])}.</p>')
            for index in range(count):
                observation = self.observation(index, load_arrays=False)
                summary = {key: observation.get(key) for key in ("step_index", "sim_time", "wall_time", "world_version",
                                                                  "gripper_open_ratio", "contact_confirmed", "unexpected_collision")}
                if index < self.action_count:
                    action = self.action(index, load_arrays=False)
                    summary["next_action"] = {key: action.get(key) for key in ("stage", "candidate_id", "plan_id", "control_mode")}
                stream.write(f'<section><h2>Observation {index}</h2><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>')
                sensors = observation.get("sensors") or {}
                for camera_index, (camera, frame) in enumerate(sensors.get("frames", {}).items()):
                    rgb = frame.get("rgb")
                    if not isinstance(rgb, dict) or rgb.get("encoding") != "png":
                        raise ValueError("recorded RGB preview requires PNG payloads")
                    filename = f"{index:06d}_{camera_index:03d}.png"
                    shutil.copyfile(self._unchanged_path(rgb["path"]), image_dir / filename)
                    stream.write(f'<figure><img src="images/{filename}" alt="{html.escape(camera, quote=True)}">'
                                 f'<figcaption>{html.escape(camera)}</figcaption></figure>')
                stream.write('</section>')
            stream.write('''<script>
const frames=Array.from(document.querySelectorAll('section'));
const slider=document.getElementById('frame'), position=document.getElementById('position');
let timer=null;
function show(i){if(!frames.length)return; i=Math.max(0,Math.min(frames.length-1,i));
  slider.value=i; frames.forEach((f,j)=>f.hidden=j!==i); position.textContent=(i+1)+' / '+frames.length;}
document.getElementById('prev').onclick=()=>show(Number(slider.value)-1);
document.getElementById('next').onclick=()=>show(Number(slider.value)+1);
slider.oninput=()=>show(Number(slider.value));
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return;}
  timer=setInterval(()=>{let n=Number(slider.value)+1;if(n>=frames.length){clearInterval(timer);timer=null;return;}show(n);},200);};
show(0);
</script></body></html>''')
        with (output / "replay_summary.json").open("x", encoding="utf-8") as stream:
            json.dump({"mode": "recorded_observation_display", "physics_reexecuted": False,
                       "source_episode": str(self.root), "source_manifest_sha256": file_digest(self.root/"manifest.json"),
                       "observation_count": self.observation_count, "displayed_frames": count,
                       "action_count": self.action_count, "hashes_verified": self.verify_hashes}, stream, indent=2)
        return path
