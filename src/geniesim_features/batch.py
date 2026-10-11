"""Bounded local batch supervision, not a service or a substitute simulator.

Each seed is consumed by the real USD randomizer in an isolated authoring
process. Each Franka demonstration gets a separate Isaac process. A zero exit
code alone is never a successful grasp: committed child artifacts and physical
task results must validate too. Failed and timed-out directories are retained.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import finite_float, load_config, positive_int, safe_name, write_json


_FIELDS = {"mode", "scene", "scene_config", "seed", "randomization_config", "sensor_config", "cameras",
           "robot_profile", "task", "max_intervals", "physics_hz", "control_hz", "headless", "timeout_seconds"}
_PATH_FIELDS = ("scene", "scene_config", "randomization_config", "sensor_config", "robot_profile", "task")
_CATEGORIES = ("succeeded", "task_failed", "configuration_failed", "scene_generation_failed", "randomization_failed", "process_failed",
               "dependency_unavailable", "timeout", "invalid_result", "interrupted", "not_run")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return result


def _timeout(value):
    value = finite_float(value, "timeout_seconds")
    if not 0 < value <= 86400:
        raise ValueError("timeout_seconds must be in (0,86400]")
    return value


def load_batch(path, *, timeout_seconds=600.0):
    """Resolve an explicit episode list; missing input files fail their own item.

    Unknown keys, duplicate IDs, unsafe paths and malformed seeds are structural
    errors and reject the batch before creating output. Every path in YAML is
    anchored to the batch YAML directory. No executable/shell command comes from
    the manifest; the caller optionally selects a Python launcher separately.
    """
    path = Path(path).resolve(strict=True)
    spec = load_config(path)
    if spec.get("schema_version") != 1:
        raise ValueError("Batch manifest requires explicit schema_version: 1")
    if set(spec) - {"schema_version", "batch_id", "defaults", "episodes"}:
        raise ValueError("Unknown batch configuration keys")
    batch_id = safe_name(spec.get("batch_id", path.stem))
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, dict) or set(defaults) - (_FIELDS - {"seed"}):
        raise ValueError("Batch defaults contain unknown keys; each episode needs its own explicit seed")
    entries = spec.get("episodes")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 10000:
        raise ValueError("Batch episodes must be a list containing 1 to 10000 entries")
    result, ids = [], set()
    for raw in entries:
        if not isinstance(raw, dict) or set(raw) - (_FIELDS | {"id"}):
            raise ValueError("Episode must be a mapping with only supported fields")
        episode_id = safe_name(raw.get("id"))
        if len(episode_id) > 128 or episode_id in ids:
            raise ValueError(f"Duplicate or overlong episode id: {episode_id}")
        ids.add(episode_id)
        episode = {"mode": "franka-demonstrate", "max_intervals": 3600, "physics_hz": 60,
                   "control_hz": 60, "headless": True, "timeout_seconds": _timeout(timeout_seconds),
                   **defaults, **raw}
        # An explicitly selected source type overrides the other source type
        # inherited from defaults; specifying both in the same entry is invalid.
        if "scene_config" in raw and "scene" not in raw:
            episode.pop("scene", None)
        elif "scene" in raw and "scene_config" not in raw:
            episode.pop("scene_config", None)
        if ("scene" in episode) == ("scene_config" in episode):
            raise ValueError(f"Episode {episode_id}: provide exactly one of scene or scene_config")
        if episode["mode"] not in {"franka-demonstrate", "curobo-demonstrate"}:
            raise ValueError(f"Episode {episode_id}: unsupported mode {episode['mode']}")
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError(f"Episode {episode_id}: explicit seed must be an unsigned 64-bit integer")
        required = {"randomization_config", "sensor_config", "task"}
        if episode["mode"] == "curobo-demonstrate":
            required.add("robot_profile")
        for key in required:
            if key not in episode:
                raise ValueError(f"Episode {episode_id}: missing {key}")
        for key in _PATH_FIELDS:
            if key not in episode:
                continue
            if not isinstance(episode[key], str) or not episode[key] or "\x00" in episode[key]:
                raise ValueError(f"Episode {episode_id}: {key} must be a nonempty path")
            target = Path(episode[key]).expanduser()
            if not target.is_absolute():
                target = path.parent/target
            episode[key] = str(target.resolve())
        for field, maximum in (("max_intervals",100000), ("physics_hz",10000), ("control_hz",10000)):
            positive_int(episode[field], field, maximum)
        if episode["physics_hz"] % episode["control_hz"]:
            raise ValueError("physics_hz must be divisible by control_hz")
        if not isinstance(episode["headless"], bool):
            raise ValueError("headless must be a boolean")
        cameras = episode.get("cameras")
        if cameras is not None:
            if not isinstance(cameras, list) or not cameras:
                raise ValueError("cameras must be a nonempty list when supplied")
            cameras = [safe_name(name) for name in cameras]
            if len(set(cameras)) != len(cameras):
                raise ValueError("Duplicate camera ids in episode")
            episode["cameras"] = cameras
        episode["timeout_seconds"] = _timeout(episode["timeout_seconds"])
        result.append(episode)
    return {"schema_version": 1, "batch_id": batch_id, "config_file": str(path), "config_sha256": _sha(path), "episodes": result}


def _launcher(value):
    # Do NOT resolve symlinks: .venv/bin/python must retain its environment path.
    path = Path(value).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Python launcher must be an existing executable file: {path}")
    return str(path)


def _terminate(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=5.0)
    finally:
        # The leader may exit on SIGTERM while one of its descendants ignores
        # that signal. Kill any remaining members of OUR isolated process group.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_process(command, *, stdout_path, stderr_path, timeout_seconds, cwd):
    """Real subprocess supervision with disk logs and bounded group teardown."""
    timeout_seconds = _timeout(timeout_seconds)
    start = time.monotonic()
    process = None
    with Path(stdout_path).open("xb") as stdout, Path(stderr_path).open("xb") as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=cwd,
                                       stdin=subprocess.DEVNULL, shell=False, start_new_session=(os.name == "posix"))
            try:
                code = process.wait(timeout=timeout_seconds)
                return {"returncode": code, "timed_out": False, "elapsed_seconds": time.monotonic()-start}
            except subprocess.TimeoutExpired:
                _terminate(process)
                return {"returncode": process.returncode, "timed_out": True, "elapsed_seconds": time.monotonic()-start}
        except BaseException:
            if process is not None:
                _terminate(process)
            raise


def _tail(path, maximum=8192):
    if not Path(path).is_file():
        return ""
    with Path(path).open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell()-maximum))
        return handle.read(maximum).decode("utf-8", errors="replace")


def classify_execution(output, episode):
    """Validate a complete, hash-verified episode and its physical task result."""
    from .recording.replay import EpisodeReader
    reader = EpisodeReader(output, require_complete=True, verify_hashes=True)
    manifest = reader.manifest
    outcome = manifest.get("outcome")
    task_result = manifest.get("task_result")
    if manifest.get("execution_mode") != "contact_physics":
        raise ValueError("Demonstration has no contact-physics execution evidence")
    if not isinstance(task_result, dict) or type(task_result.get("success")) is not bool:
        raise ValueError("Demonstration has no physical success result")
    success = task_result["success"]
    if outcome != ("success" if success else "failure"):
        raise ValueError("Demonstration outcome conflicts with physical success result")
    return {"category": "succeeded" if success else "task_failed", "outcome": outcome,
            "failure_reason": manifest.get("failure_reason"), "execution_mode": manifest["execution_mode"],
            "action_count": reader.action_count, "observation_count": reader.observation_count,
            "evidence": "validated_child_episode_artifacts", "physics_grasp_success": success}


def verify_pending_camera_application(output, prepared):
    """Requested camera samples are not applied until the engine authors cameras."""
    import numpy as np
    pending = prepared.get("pending_camera_components") or {}
    if not pending:
        return None
    path = Path(output)/"randomization.json"
    applied = _read_json(path)
    if applied.get("applied") is not True or applied.get("pending_camera_components"):
        raise ValueError("Engine did not apply all pending camera randomization components")
    if applied.get("context",{}).get("global_seed") != prepared.get("context",{}).get("global_seed"):
        raise ValueError("Engine camera randomization seed differs from the prepared request")
    for component_id,requested in pending.items():
        actual = applied.get("components",{}).get(component_id,{})
        if actual.get("kind") != "camera" or actual.get("target") != requested.get("target"):
            raise ValueError(f"Missing/mismatched applied camera component: {component_id}")
        if set(actual.get("actual",{})) != set(requested["requested"]):
            raise ValueError(f"Missing actual camera read-back fields: {component_id}")
        for field,value in requested["requested"].items():
            if not np.allclose(actual["actual"][field],value,rtol=1e-5,atol=1e-6):
                raise ValueError(f"Applied camera read-back mismatch: {component_id}.{field}")
    return {"path":str(path),"sha256":_sha(path),"camera_readback_verified":True}


def _commands(episode, directory, authoring_python, engine_python):
    randomized = directory/"randomization"
    variant = randomized/"variant_000000"
    source_scene = episode.get("scene") or str(directory/"source_scene")
    prepare = [authoring_python,"-m","geniesim_features","randomize","--scene",source_scene,
               "--config",episode["randomization_config"],"--output",str(randomized),"--seed",str(episode["seed"]),"--samples","1"]
    execute = [engine_python, "-m", "geniesim_features", episode["mode"],
               "--scene", str(variant), "--config", episode["sensor_config"],
               "--output", str(directory / "run"),
               "--physics-hz", str(episode["physics_hz"]),
               "--control-hz", str(episode["control_hz"]),
               "--headless" if episode["headless"] else "--gui",
               "--task", episode["task"], "--max-intervals", str(episode["max_intervals"])]
    if episode["mode"] == "curobo-demonstrate":
        execute.extend(["--robot-profile", episode["robot_profile"]])
    for camera in episode.get("cameras") or []:
        execute.extend(["--camera",camera])
    return prepare, execute, variant


def _check_inputs(episode):
    files = []
    if "scene" in episode:
        scene = Path(episode["scene"])
        if not scene.is_dir():
            raise ValueError(f"Episode scene bundle directory is missing: {scene}")
        files.extend(scene/name for name in ("scene.usda", "scene_manifest.json", "bindings.json"))
    files.extend(Path(episode[key]) for key in _PATH_FIELDS if key != "scene" and key in episode)
    for path in files:
        if not path.is_file():
            raise ValueError(f"Episode input file is missing: {path}")
    return {str(path): _sha(path) for path in files}


def _execute_episode(episode, directory, authoring_python, engine_python, cwd):
    start = time.monotonic()
    deadline = start+episode["timeout_seconds"]
    result = {"id": episode["id"], "mode": episode["mode"], "seed": episode["seed"],
              "category": "configuration_failed", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_scene": episode.get("scene") or str(directory/"source_scene"),
              "scene_config": episode.get("scene_config"), "output": str(directory/"run"), "processes": []}
    write_json(directory/"attempt.json", {"episode": episode, "status": "started", "started_at_utc": result["started_at_utc"]})
    try:
        result["input_sha256"] = _check_inputs(episode)
        prepare, execute, variant = _commands(episode,directory,authoring_python,engine_python)
        result["randomized_scene"] = str(variant)
        randomization = None
        phases = []
        if "scene_config" in episode:
            phases.append(("scene_generation", [authoring_python,"-m","geniesim_features","scene","--config",episode["scene_config"],
                                                  "--seed",str(episode["seed"]),"--output",str(directory/"source_scene")]))
        phases.extend((("randomization",prepare), ("execution",execute)))
        for phase, command in phases:
            result["category"] = {"scene_generation":"scene_generation_failed","randomization":"randomization_failed","execution":"process_failed"}[phase]
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                result.update(category="timeout", failure_reason=f"episode_budget_exhausted_before_{phase}")
                break
            process_record = {"phase":phase, "command":command, "stdout":f"{phase}.stdout.log", "stderr":f"{phase}.stderr.log"}
            result["processes"].append(process_record)
            process_record.update(run_process(command, stdout_path=directory/process_record["stdout"],
                                               stderr_path=directory/process_record["stderr"], timeout_seconds=remaining, cwd=cwd))
            if process_record["timed_out"]:
                result.update(category="timeout", failure_reason=f"{phase}_timeout")
                break
            if process_record["returncode"] != 0:
                detail = _tail(directory/process_record["stderr"])
                category = "dependency_unavailable" if "DependencyUnavailable" in detail else result["category"]
                result.update(category=category, failure_reason=f"{phase}_exit_{process_record['returncode']}", stderr_tail=detail)
                break
            if phase == "scene_generation":
                generated = _read_json(directory/"source_scene/scene_manifest.json")
                if generated.get("seed") != episode["seed"] or not (directory/"source_scene/scene.usda").is_file():
                    raise ValueError("Scene generation did not use the requested seed")
                result["scene_generation"] = {"manifest":str(directory/"source_scene/scene_manifest.json"),
                                               "sha256":_sha(directory/"source_scene/scene_manifest.json"),
                                               "seed":generated["seed"],"asset_selection":generated.get("asset_selection",{})}
            elif phase == "randomization":
                randomization = _read_json(variant/"randomization.json")
                if not (variant/"scene.usda").is_file() or randomization.get("context",{}).get("global_seed") != episode["seed"]:
                    raise ValueError("Randomizer did not produce a real scene with the requested seed")
                if not randomization.get("components") and not randomization.get("pending_camera_components"):
                    raise ValueError("Randomizer produced no parameter components; seed was not consumed")
                result["seed_application"] = {"randomization_json":str(variant/"randomization.json"),
                                              "sha256":_sha(variant/"randomization.json"),
                                              "context":randomization["context"]}
            else:
                result["category"] = "invalid_result"
                applied_camera = verify_pending_camera_application(directory/"run",randomization)
                if applied_camera:
                    result["applied_camera_randomization"] = applied_camera
                result.update(classify_execution(directory/"run",episode))
        result["elapsed_seconds"] = time.monotonic()-start
    except KeyboardInterrupt:
        result.update(category="interrupted", failure_reason="operator_keyboard_interrupt", elapsed_seconds=time.monotonic()-start)
        write_json(directory/"result.json", result)
        raise
    except Exception as error:
        result.update(failure_reason=f"{type(error).__name__}:{error}", elapsed_seconds=time.monotonic()-start)
    write_json(directory/"result.json", result)
    return result


def run_batch(config_path, output_dir, *, python_executable=None, timeout_seconds=600.0):
    spec = load_batch(config_path, timeout_seconds=timeout_seconds)
    authoring_python = _launcher(sys.executable)
    engine_python = _launcher(python_executable or sys.executable)
    output = Path(output_dir).expanduser().absolute()
    if any(path.is_symlink() for path in (output,*output.parents)):
        raise ValueError("Batch output path must not traverse symlinks")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite batch output: {output}")
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"resolved_batch.json",spec)
    results = []
    interrupted = False
    started = time.monotonic()
    try:
        for episode in spec["episodes"]:
            directory = output/episode["id"]
            directory.mkdir(exist_ok=False)
            try:
                result = _execute_episode(episode,directory,authoring_python,engine_python,Path(spec["config_file"]).parent)
                results.append(result)
            except KeyboardInterrupt:
                interrupted = True
                results.append(_read_json(directory/"result.json"))
                raise
    finally:
        completed = {result["id"] for result in results}
        entries = results + [{"id":episode["id"],"mode":episode["mode"],"seed":episode["seed"],"category":"not_run"}
                             for episode in spec["episodes"] if episode["id"] not in completed]
        counts = Counter(item["category"] for item in entries)
        summary = {"schema_version":1,"batch_id":spec["batch_id"],"status":"interrupted" if interrupted else ("complete" if len(results)==len(spec["episodes"]) else "partial"),
                   "execution_policy":"separate_authoring_and_engine_processes_per_episode", "config_sha256":spec["config_sha256"],
                   "authoring_python":authoring_python,"engine_python":engine_python,"elapsed_seconds":time.monotonic()-started,
                   "requested_episodes":len(spec["episodes"]),"finished_episodes":len(results),
                   "all_episodes_succeeded":len(results)==len(spec["episodes"]) and counts["succeeded"]==len(results),
                   "counts":{category:counts[category] for category in _CATEGORIES},
                   "failure_reasons":dict(Counter(item.get("failure_reason") or item["category"] for item in entries if item["category"]!="succeeded")),
                   "results":entries,
                   "validation_scope":"Child artifact integrity/outcomes only; batch supervision is not independent physics validation."}
        write_json(output/"batch_manifest.json",summary)
    return summary
