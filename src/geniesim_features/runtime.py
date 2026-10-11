"""Single-process Isaac workflows, imported only for explicitly requested engine runs."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .config import load_config, write_json, positive_int, safe_name, finite_float
from .adapters.isaac_world import IsaacSession, IsaacDrawer


def scene_bundle(directory):
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("scene must be a bundle directory")
    scene = directory / "scene.usda"
    scene.resolve(strict=True)
    binding_document = json.loads((directory / "bindings.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "scene_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("status") != "authored":
        raise ValueError("scene bundle requires a committed schema_version=1 authored manifest")
    if not isinstance(binding_document, dict) or binding_document.get("schema_version") != 1:
        raise ValueError("scene bindings require schema_version=1")
    bindings = binding_document.get("objects")
    if not isinstance(bindings, dict) or not bindings or not all(isinstance(v, dict) for v in bindings.values()):
        raise ValueError("scene bindings require a nonempty objects mapping")
    for filename, expected in manifest.get("files", {}).items():
        if Path(filename).name != filename:
            raise ValueError("bundle digest filenames must not traverse directories")
        actual = hashlib.sha256((directory/filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"scene bundle digest mismatch: {filename}")
    for path, expected in manifest.get("source_dependency_hashes", {}).items():
        from .provenance import sha256
        if sha256(path) != expected:
            raise ValueError(f"scene source dependency digest mismatch: {path}")
    return scene, bindings, manifest


def camera_config(path, selected):
    config = load_config(path)
    cameras = config.get("cameras")
    if not isinstance(cameras, list) or not cameras or not all(isinstance(cam, dict) for cam in cameras):
        raise ValueError("Expected nonempty cameras list")
    names = [safe_name(cam.get("id")) for cam in cameras]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate camera id")
    if selected:
        if not isinstance(selected, (list, tuple)) or any(not isinstance(name, str) for name in selected):
            raise ValueError("selected camera ids must be a list of strings")
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate requested camera id")
        requested = set(selected)
        available = set(names)
        if not requested <= available:
            raise ValueError(f"Unknown camera ids: {sorted(requested - available)}")
        config["cameras"] = [cam for cam in config["cameras"] if cam["id"] in requested]
    for cam in config["cameras"]:
        resolution = cam.get("resolution", [640,480])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("camera resolution must be [width,height]")
        for dimension in resolution:
            positive_int(dimension, "camera resolution", 4096)
        near, far = finite_float(cam.get("near", .01), "near"), finite_float(cam.get("far", 10), "far")
        if not 0 < near < far:
            raise ValueError("camera clipping requires 0 < near < far")
        for key, default in (("focal_length_mm",24), ("horizontal_aperture_mm",36), ("vertical_aperture_mm",27)):
            if finite_float(cam.get(key, default), key) <= 0:
                raise ValueError(f"camera {key} must be positive")
        position = np.asarray(cam.get("position_m", [0,0,0]), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("camera position must be finite XYZ")
        parent = cam.get("parent_prim", "/World")
        if not isinstance(parent, str) or not parent.startswith("/") or parent == "/" or "/../" in parent:
            raise ValueError("camera parent must be an absolute non-root prim path")
        prim_path = cam.get("prim_path", parent.rstrip("/")+"/"+cam["id"])
        if not isinstance(prim_path, str) or not prim_path.startswith(parent.rstrip("/")+"/") or "/../" in prim_path:
            raise ValueError("camera prim must be under its parent")
        if "look_at_world_m" in cam:
            if "quaternion_wxyz" in cam or parent != "/World":
                raise ValueError("world look-at requires World parent and no conflicting quaternion")
            from .adapters.isaac_sensors import look_at_usd
            target = np.asarray(cam["look_at_world_m"], dtype=float)
            if target.shape != (3,):
                raise ValueError("camera look-at target must be XYZ")
            look_at_usd(position, target)
        else:
            quaternion = np.asarray(cam.get("quaternion_wxyz", [1,0,0,0]), dtype=float)
            if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-9:
                raise ValueError("camera quaternion must be finite nonzero wxyz")
    return config


def validate_timing(args, task=None):
    physics = positive_int(args.physics_hz, "physics_hz", 10000)
    control = positive_int(args.control_hz, "control_hz", 10000)
    if physics % control:
        raise ValueError("physics_hz must be divisible by control_hz")
    if task is not None and (not np.isclose(task.control_dt, 1/control, rtol=1e-7, atol=1e-10)
                             or task.physics_steps_per_interval != physics//control):
        raise ValueError("Task timing differs from engine command-line timing")
    return physics//control


def validate_task_scene(task, bindings, target_bin_id="target_bin"):
    """Fail closed when the task's success volume differs from authored geometry.

    The current executor uses a world AABB, so rotated non-axis-aligned bins and
    mobile bins need a different containment model and are explicitly rejected.
    This validates configuration only; real post-step poses still decide success.
    """
    if task.object_id not in bindings or target_bin_id not in bindings:
        raise ValueError("Task object_id/target_bin_id must exist in scene bindings")
    target, container = bindings[task.object_id], bindings[target_bin_id]
    if not target.get("rigid_body_prim") or not target.get("dynamic"):
        raise ValueError("Task grasp target must be a dynamic rigid body")
    half = np.asarray(target.get("size_m"), float)/2
    if half.shape != (3,) or not np.allclose(task.object_half_extents, half, rtol=0, atol=1e-7):
        raise ValueError("Task object_half_extents differ from scene-bound object dimensions")
    if container.get("dynamic") or "interior_bounds_local_m" not in container:
        raise ValueError("Task target bin must be a fixed container with an authored usable cavity")
    yaw = finite_float(container.get("initial_yaw_deg", 0), "bin yaw")
    if not np.isclose(yaw/90, round(yaw/90), atol=1e-7, rtol=0):
        raise ValueError("World-AABB containment requires an axis-aligned fixed bin")
    bounds = np.asarray(container["interior_bounds_local_m"], float)
    center = np.asarray(container.get("initial_root_position_m"), float)
    if bounds.shape != (2,3) or center.shape != (3,) or not np.isfinite(bounds).all() or not np.isfinite(center).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError("Invalid bin bounds/position in scene binding")
    angle = np.deg2rad(yaw)
    rotation = np.array([[np.cos(angle),-np.sin(angle),0], [np.sin(angle),np.cos(angle),0], [0,0,1]])
    corners = np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]]) @ rotation.T + center
    if np.any(task.bin_min_W < corners.min(axis=0)-1e-7) or np.any(task.bin_max_W > corners.max(axis=0)+1e-7):
        raise ValueError("Task bin success bounds extend outside the actual container cavity")
    radius = np.abs(task.T_W_O_place[:3,:3]) @ task.object_half_extents
    position = task.T_W_O_place[:3,3]
    if np.any(position-radius < task.bin_min_W-1e-7) or np.any(position+radius > task.bin_max_W+1e-7):
        raise ValueError("Task placement pose does not fit the complete object inside its usable bin volume")


def make_session(scene, args):
    validate_timing(args)
    if not isinstance(args.headless, bool):
        raise ValueError("headless must be a boolean")
    return IsaacSession(scene, headless=args.headless, physics_hz=args.physics_hz, control_hz=args.control_hz,
                        defer_app_shutdown=getattr(args, "_defer_app_shutdown", False))


def settle_scene(session, sampler, manifest, output, before_step=None):
    """Check real dynamic states; authored USD alone is not an accepted sample."""
    from .scene_generation.acceptance import SceneSettleMonitor
    monitor = SceneSettleMonitor.from_manifest(manifest)
    if not monitor.monitored_object_ids:
        result = {"schema_version": 1, "status": "not_applicable", "accepted": None,
                  "reason": "no_independent_dynamic_objects", "physics_stability_verified": False,
                  "excluded_objects": monitor.excluded}
        # Articulation acceptance belongs to the drawer interaction workflow.
        # Still advance an explicit warm-up interval before static rendering.
        session.step(session.substeps, before_step=before_step)
        write_json(Path(output) / "settling.json", result)
        return result
    result = monitor.update(sampler.read_snapshot())
    # The monitor bounds its own physical-time horizon. Additional hard bound
    # protects the host if the engine/sampler stops advancing its clock.
    for _ in range(3 * session.control_hz + 1):
        if result["terminal"]:
            break
        session.step(session.substeps, before_step=before_step)
        result = monitor.update(sampler.read_snapshot())
    result = monitor.finish()
    write_json(Path(output) / "settling.json", result)
    if not result["accepted"]:
        raise RuntimeError(f"scene_rejected:{json.dumps(result['reasons'], ensure_ascii=False)}")
    return result


def _close_runtime(session, *components):
    """Attempt every cleanup; preserve an existing primary host exception."""
    import sys
    active = sys.exc_info()[1]
    errors = []
    actions = [session.world.pause] + [item.close for item in components if item is not None]
    for action in actions:
        try:
            action()
        except BaseException as error:
            errors.append(f"{type(error).__name__}:{error}")
    if errors:
        if active is not None:
            active.add_note("cleanup_errors: " + "; ".join(errors))
        else:
            raise RuntimeError("cleanup_errors: " + "; ".join(errors))


def run_interaction(args):
    from .articulated_objects.schedule import DrawerCommandSchedule
    validate_timing(args)
    scene, bindings, manifest = scene_bundle(args.scene)
    steps = positive_int(args.steps, "steps", 1_000_000)
    binding = bindings[args.object]
    if binding.get("joint_prim") is None:
        raise ValueError("Selected object is not an articulated drawer")
    if args.mode not in {"external_force", "joint_drive"}:
        raise ValueError("Unknown interaction mode")
    force, target = finite_float(args.force, "force"), finite_float(args.target, "target")
    if args.mode == "external_force" and abs(force) > 40:
        raise ValueError("Demo handle force must be within ±40 N")
    if args.mode == "joint_drive" and not binding["limits_m"][0] <= target <= binding["limits_m"][1]:
        raise ValueError("Drawer target exceeds configured joint limits")
    schedule_path = getattr(args, "schedule", None)
    schedule_spec = load_config(schedule_path) if schedule_path else {
        "mode": args.mode, "interpolation": "hold", "after_end": "hold",
        "keyframes": [{"time_s": 0., "value": force if args.mode == "external_force" else target}]}
    schedule_spec = dict(schedule_spec)
    schedule_spec.pop("schema_version", None)
    schedule = DrawerCommandSchedule.from_spec(schedule_spec, limits_m=tuple(binding["limits_m"]))
    if schedule_spec["mode"] != args.mode:
        raise ValueError("Schedule mode and command-line interaction mode differ")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "status": "partial", "interaction_mode": args.mode,
              "physics_hz": args.physics_hz, "physics_backend": "IsaacSim/PhysX",
              "force_n": args.force, "target_m": args.target, "scene": manifest,
              "control_hz": args.control_hz, "schedule": schedule_spec}
    started = time.monotonic()
    try:
        with make_session(scene, args) as session:
            drawer = IsaacDrawer(session, binding, args.mode)
            session.initialize()
            with (args.output / "joint_states.jsonl").open("x", encoding="utf-8") as output:
                output.write(json.dumps(drawer.read_state(reset_interval=True)) + "\n")
                for index in range(steps):
                    # Force is applied each physics tick, not just each logged
                    # control tick. Drive targets use the same explicit clock.
                    session.step(before_step=lambda: drawer.apply_command(schedule.evaluate(session.simulation_time)))
                    record_tick = (index + 1) % session.substeps == 0 or index + 1 == steps
                    state = drawer.read_state(reset_interval=record_tick)
                    lo, hi = binding["limits_m"]
                    if not lo - .002 <= state["position_m"] <= hi + .002:
                        raise RuntimeError("Observed drawer position exceeded configured limit tolerance")
                    if record_tick:
                        output.write(json.dumps(state, allow_nan=False) + "\n")
            result.update(status="complete", observed_steps=session.step_index,
                          final_state=drawer.read_state(), force_is_robot_contact=False)
        from .articulated_objects.curves import export_drawer_curves
        result["curves"] = export_drawer_curves(args.output / "joint_states.jsonl", args.output)
    except BaseException as error:
        result["status"] = "partial"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["wall_time_s"] = time.monotonic() - started
        write_json(args.output / "run_manifest.json", result)
    return {"output": str(args.output), "status": result["status"]}


def run_capture(args):
    from .adapters.isaac_sensors import IsaacSensorRig
    from .adapters.isaac_world import IsaacObjectStateSampler
    from .domain_randomization import StaticSampleWriter
    from .provenance import (configuration_record, archive_configuration, archive_asset_dependencies,
                             load_randomization, apply_pending_cameras)
    validate_timing(args)
    scene, bindings, manifest = scene_bundle(args.scene)
    config = camera_config(args.config, args.camera)
    frames = positive_int(args.frames, "frames", 10000)
    profile = limits = gripper = None
    if getattr(args, "robot_profile", None):
        from .adapters.robot_profile import load_robot_profile
        profile, limits, gripper = load_robot_profile(args.robot_profile)
        # The adapter only needs a dynamic reference object for observations;
        # static capture does not create a cuRobo planner or execute a task.
        targets = [name for name, binding in bindings.items() if binding.get("dynamic") and not binding.get("joint_prim")]
        if not targets:
            raise ValueError("Robot-mounted capture requires a dynamic scene reference object")
    randomization = load_randomization(args.scene)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "status": "partial", "renderer": "IsaacSim/RTX",
              "sample_count": 0, "physics_hz": args.physics_hz}
    started = time.monotonic()
    try:
        archive_configuration(args.output, configuration_record(args, scene, config, profile=profile), randomization)
        with make_session(scene, args) as session:
            rig = robot = sampler = None
            try:
                sensor_bindings = dict(bindings)
                if profile:
                    from .adapters.g2_robot import G2RobotAdapter
                    robot = G2RobotAdapter(session, profile, limits, gripper, bindings, targets[0])
                    sensor_bindings["robot"] = {"prim_path": robot.root}
                rig = IsaacSensorRig(session, config, sensor_bindings)
                archive_asset_dependencies(args.output, scene, profile)
                randomization = apply_pending_cameras(session, randomization)
                write_json(args.output / "randomization.json", randomization)
                sampler = IsaacObjectStateSampler(session, bindings)
                session.initialize()
                if robot:
                    robot.initialize()
                hold = robot.hold if robot else None
                result["settling"] = settle_scene(session, sampler, manifest, args.output, before_step=hold)
                writer = StaticSampleWriter(args.output / "samples")
                for index in range(frames):
                    bundle = rig.capture(index)
                    from .sensors.visibility import check_visibility
                    visibility = check_visibility(bundle, bindings, config.get("visibility"))
                    scene_snapshot = {"scene_manifest": manifest, "bindings": bindings,
                                      "physics_step": session.step_index, "visibility": visibility,
                                      "object_poses": {object_id: session.pose_of(binding.get("rigid_body_prim") or binding["prim_path"]).tolist()
                                                       for object_id,binding in bindings.items()}}
                    writer.write(bundle, scene_snapshot, randomization, sample_id=f"frame_{index:06d}")
                    result["sample_count"] += 1
                    if index + 1 < frames:
                        session.step(session.substeps, before_step=hold)
            finally:
                _close_runtime(session, rig, robot, sampler)
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "partial"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["wall_time_s"] = time.monotonic() - started
        write_json(args.output / "run_manifest.json", result)
    return {"output": str(args.output), "samples": result["sample_count"]}


def run_demonstration(args):
    from .adapters.robot_profile import load_robot_profile
    from .adapters.g2_robot import G2RobotAdapter
    from .adapters.curobo_planner import CuroboPlanner
    from .adapters.isaac_sensors import IsaacSensorRig
    from .manipulation import PickPlaceTask, PickPlaceExecutor, GraspCandidate, ExecutionStopped
    from .recording import EpisodeWriter
    from .adapters.isaac_world import IsaacObjectStateSampler
    from .provenance import (configuration_record, archive_configuration, archive_asset_dependencies,
                             load_randomization, apply_pending_cameras)

    validate_timing(args)
    hold_gui = bool(getattr(args, "hold_gui", False))
    if hold_gui and args.headless:
        raise ValueError("--hold-gui requires --gui")
    scene, bindings, scene_manifest = scene_bundle(args.scene)
    profile, limits, gripper = load_robot_profile(args.robot_profile)
    task_spec = load_config(args.task)
    task_spec.pop("schema_version", None)
    target_bin_id = task_spec.pop("target_bin_id", "target_bin")
    task_spec["T_E_G"] = profile["T_E_G"]
    task_spec["candidates"] = tuple(GraspCandidate(**candidate) for candidate in task_spec["candidates"])
    task = PickPlaceTask(**task_spec)
    validate_timing(args, task)
    validate_task_scene(task, bindings, target_bin_id)
    positive_int(args.max_intervals, "max_intervals", 100000)
    config = camera_config(args.config, args.camera)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Episode output exists: {args.output}")
    writer = EpisodeWriter(args.output, max_actions=args.max_intervals)
    manifest = {"schema_version": 1, "feature_id": "F-GENIESIM-105",
                "execution_mode": "contact_physics", "physics_hz": args.physics_hz,
                "control_hz": args.control_hz, "camera_ids": [cam["id"] for cam in config["cameras"]],
                "robot_profile": profile, "scene": scene_manifest,
                "task_sha256": hashlib.sha256(Path(args.task).read_bytes()).hexdigest(),
                "simulator": "IsaacSim5.1/PhysX", "planner": "cuRobo0.7.6",
                "contact_evidence": "pairwise_net_force_gt_0.01N_both_fingers"}
    started = time.monotonic()
    executor = None
    writer.reserve(manifest)
    try:
        randomization = load_randomization(args.scene)
        archived_task = {**load_config(args.task), "T_E_G": profile["T_E_G"]}
        archive_configuration(args.output, configuration_record(args, scene, config, profile=profile,
                                                                task=archived_task), randomization)
        with make_session(scene, args) as session:
            rig = robot = sampler = None
            try:
                archive_asset_dependencies(args.output, scene, profile)
                robot = G2RobotAdapter(session, profile, limits, gripper, bindings, task.object_id)
                sensor_bindings = dict(bindings, robot={"prim_path": robot.root})
                rig = IsaacSensorRig(session, config, sensor_bindings)
                randomization = apply_pending_cameras(session, randomization)
                write_json(args.output / "randomization.json", randomization)
                sampler = IsaacObjectStateSampler(session, bindings)
                session.initialize()
                robot.initialize()
                settle_scene(session, sampler, scene_manifest, args.output, before_step=robot.hold)
                session.world.pause()
                if hold_gui:
                    from omni.kit.viewport.utility import get_active_viewport
                    import omni.ui as ui
                    camera = config["cameras"][0]
                    get_active_viewport().camera_path = camera.get(
                        "prim_path", camera.get("parent_prim", "/World").rstrip("/") + "/" + camera["id"])
                    status_window = ui.Window("Grasp execution", width=600, height=150)
                    with status_window.frame:
                        status_label = ui.Label("Initializing planner", word_wrap=True)
                planner = CuroboPlanner(session, robot, control_dt=task.control_dt, timeout_s=task.planner_timeout_s)
                executor = PickPlaceExecutor(planner, gripper, limits)
                executor.start(task)
                observation = robot.observe(rig.capture(0))
                initial_sim_time = observation.sim_time
                writer.begin(manifest, observation)
                displayed_stage = None
                for index in range(args.max_intervals):
                    if hold_gui and executor.progress.stage != displayed_stage:
                        displayed_stage = executor.progress.stage
                        status_label.text = f"{displayed_stage} | observation {index}"
                        print(f"GUI_STAGE {displayed_stage} step={observation.step_index}", flush=True)
                        session.world.pause()
                        until = time.monotonic() + 4.0
                        while session.app.is_running() and time.monotonic() < until:
                            session.world.pause()
                            session.app.update()
                    planner.update_observation(observation)
                    try:
                        command = executor.command_for_next_interval(observation)
                    except ExecutionStopped:
                        break
                    robot.apply(command)
                    session.step(task.physics_steps_per_interval, before_step=robot.hold)
                    observation = robot.observe(rig.capture(index + 1))
                    planner.update_observation(observation)
                    writer.append(command, observation)
                    if executor.observe_transition(observation).done:
                        break
                if not executor.progress.done:
                    executor.abort("host_interval_budget_exhausted")
                wall_seconds = time.monotonic() - started
                result = {**asdict(executor.progress), "execution_wall_time_s": wall_seconds,
                          "recorded_simulation_s": observation.sim_time - initial_sim_time,
                          "real_time_factor": (observation.sim_time - initial_sim_time) / wall_seconds,
                          "timing_scope": "wall_clock_includes_startup_settling_planning_rendering_sync_recording"}
                if hold_gui:
                    session.world.pause()
                    contacts = [
                        {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                         for key, value in contact.items()}
                        for contact in observation.contacts]
                    write_json(args.output / "gui_result.json", {**result, "contacts": contacts})
                    detail = "\n".join(
                        f"{c['robot_link']} <-> {c['environment_prim']} "
                        f"{c['force_norm_n']:.3f} N allowed={c['allowed']}" for c in contacts)
                    status_label.text = (
                        f"{'SUCCESS' if result['success'] else 'FAIL'}: {result['stage']}\n"
                        f"{result['failure_reason']}\n{detail}")
                    print("GUI_HELD_RESULT", {**result, "contacts": contacts}, flush=True)
                    import omni.usd
                    selected = [c["robot_link"] for c in contacts if not c["allowed"]]
                    omni.usd.get_context().get_selection().set_selected_prim_paths(selected, False)
                    while session.app.is_running():
                        session.world.pause()
                        session.app.update()
            finally:
                _close_runtime(session, executor, rig, robot, sampler)
        # Physics/sensor cleanup precedes commit; CLI closes Kit after printing.
        writer.finish(result)
        return {"output": str(args.output), **result}
    except BaseException as error:
        if executor is not None:
            try:
                executor.abort(f"host_error:{type(error).__name__}:{error}")
            except Exception as abort_error:
                error.add_note(f"Executor abort failed: {abort_error}")
        try:
            writer.fail(error, details={"execution_wall_time_s": time.monotonic() - started,
                                        "task_progress": asdict(executor.progress) if executor is not None else None})
        except Exception as recording_error:
            error.add_note(f"Could not persist host failure: {recording_error}")
        raise
    finally:
        writer.close()


def run_reexecution(args):
    """Apply recorded controller targets in a NEW real physics session.

    This does not set object transforms, add welds, invoke a fake planner, or
    promise bitwise deterministic rendering. Matching is measured and reported.
    """
    from .recording import EpisodeReader, EpisodeWriter
    from .adapters.robot_profile import load_robot_profile
    from .adapters.g2_robot import G2RobotAdapter
    from .adapters.isaac_world import IsaacObjectStateSampler
    from .adapters.isaac_sensors import IsaacSensorRig
    from .provenance import (sha256, check_reproduction_inputs, archive_configuration, archive_asset_dependencies,
                             load_randomization, apply_pending_cameras)
    from .reexecution import observation_difference, within_replay_tolerance, ReplayCommandValidator

    reader = EpisodeReader(args.episode)
    source = Path(args.episode).resolve(strict=True)
    if reader.manifest.get("execution_mode") != "contact_physics":
        raise ValueError("Action re-execution requires a contact_physics episode")
    configuration = reader.read_json("configuration.json")
    check_reproduction_inputs(configuration)
    dependencies = reader.read_json("asset_dependencies.json")
    for path, expected in dependencies.items():
        if sha256(path) != expected:
            raise ValueError(f"Replay asset dependency changed: {path}")
    for name in ("physics_hz", "control_hz"):
        value = getattr(args, name, None)
        if value is not None and value != configuration[name]:
            raise ValueError(f"Same-configuration replay cannot change {name}")
        setattr(args, name, configuration[name])
    validate_timing(args)
    scene, bindings, manifest = scene_bundle(configuration["scene_directory"])
    robot_path = getattr(args, "robot_profile", None) or configuration["sources"]["robot_profile"]["path"]
    if sha256(robot_path) != configuration["sources"]["robot_profile"]["sha256"]:
        raise ValueError("Same-configuration replay requires the original robot calibration")
    profile, limits, gripper = load_robot_profile(robot_path)
    config = configuration["camera_config"]
    object_id = configuration["task"]["object_id"]
    if reader.action_count < 1:
        raise ValueError("Episode contains no actions to replay")
    # Check ALL target commands before touching the engine, streaming from disk.
    reference = reader.observation(0)
    validator = ReplayCommandValidator(limits, gripper, reference["joint_positions"], 1 / args.control_hz)
    for index in range(reader.action_count):
        validator.validate(reader.control_command(index))
    writer = EpisodeWriter(args.output, max_actions=reader.action_count)
    metadata = {"execution_mode": "contact_physics", "workflow": "action_reexecution",
                "physics_hz": args.physics_hz, "control_hz": args.control_hz,
                "source_episode": str(source), "source_manifest_sha256": sha256(source / "manifest.json"),
                "camera_ids": [cam["id"] for cam in config["cameras"]],
                "match_thresholds": {"joint_rad": .1, "object_m": .03, "object_rad": .25, "gripper_ratio": .15},
                "outcome_semantics": "action_reexecution_matching_not_new_task_success"}
    writer.reserve(metadata)
    started = time.monotonic()
    try:
        randomization = reader.read_json("randomization-request.json")
        archive_configuration(args.output, configuration, randomization)
        with make_session(scene, args) as session:
            rig = robot = sampler = None
            try:
                archive_asset_dependencies(args.output, scene, profile)
                robot = G2RobotAdapter(session, profile, limits, gripper, bindings, object_id)
                rig = IsaacSensorRig(session, config, dict(bindings, robot={"prim_path": robot.root}))
                randomization = apply_pending_cameras(session, randomization)
                write_json(args.output / "randomization.json", randomization)
                sampler = IsaacObjectStateSampler(session, bindings)
                session.initialize()
                robot.initialize()
                settle_scene(session, sampler, manifest, args.output, before_step=robot.hold)
                observed = robot.observe(rig.capture(0))
                difference = observation_difference(observed, reference)
                if not within_replay_tolerance(difference) or observed.unexpected_collision:
                    raise RuntimeError(f"Replay initial state mismatch: {difference}")
                writer.begin(metadata, observed)
                matched, reason = True, None
                maximum = {key: 0. for key in difference}
                with (args.output / "replay_comparison.jsonl").open("x", encoding="utf-8") as stream:
                    for index in range(reader.action_count):
                        command = reader.control_command(index)
                        robot.apply(command)
                        session.step(session.substeps, before_step=robot.hold)
                        observed = robot.observe(rig.capture(index + 1))
                        writer.append(command, observed)
                        difference = observation_difference(observed, reader.observation(index + 1))
                        for key in maximum:
                            maximum[key] = max(maximum[key], difference[key])
                        matched = matched and within_replay_tolerance(difference)
                        stream.write(json.dumps({"action_index": index, **difference}, allow_nan=False) + "\n")
                        stream.flush()
                        if observed.unexpected_collision:
                            reason = "replay_unexpected_collision"
                            break
                success = reason is None and matched
                result = {"done": True, "success": success,
                          "failure_reason": reason or (None if matched else "replay_state_diverged"),
                          "maximum_state_difference": maximum, "execution_wall_time_s": time.monotonic() - started,
                          "source_task_outcome": reader.manifest.get("outcome"),
                          "new_task_success_evaluated": False}
            finally:
                _close_runtime(session, rig, robot, sampler)
        writer.finish(result)
        return {"output": str(args.output), **result}
    except BaseException as error:
        try:
            writer.fail(error, details={"execution_wall_time_s": time.monotonic() - started})
        except Exception as recording_error:
            error.add_note(f"Could not persist replay failure: {recording_error}")
        raise
    finally:
        writer.close()
