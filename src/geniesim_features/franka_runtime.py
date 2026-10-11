"""PhysX Franka pick/place with frame-aligned RGB-D episode recording.

The controller, robot and USD imports are deferred until IsaacSession starts.
Both selectable backends use the same recorded PhysX contact, lift, release and
containment tests: neither controller completion nor a plan is proof of success.
"""

import hashlib
from pathlib import Path

import numpy as np

from .config import finite_float, load_config, positive_int
from .runtime import camera_config, scene_bundle


def _xyz(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite XYZ")
    return result


def _xy(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite XY")
    return result


def _task_spec(path, bindings):
    task = load_config(path)
    required = {"schema_version", "object_id", "target_bin_id", "base_offset_xy_m",
                "grasp_offset_z_m", "settle_steps"}
    extra = set(task) - required - {"backend", "placement_offset_xy_m", "grasp_yaw_offset_rad"}
    if extra or not required <= task.keys() or task["schema_version"] != 1:
        raise ValueError(f"Franka task requires {sorted(required)}; unknown keys: {sorted(extra)}")
    object_id, bin_id = task["object_id"], task["target_bin_id"]
    if object_id not in bindings or bin_id not in bindings or object_id == bin_id:
        raise ValueError("Task object and bin must be distinct scene bindings")
    target, container = bindings[object_id], bindings[bin_id]
    if not target.get("dynamic") or not target.get("rigid_body_prim") or target.get("joint_prim"):
        raise ValueError("Pick object must be a dynamic rigid body")
    if container.get("dynamic") or not container.get("prim_path") or len(container.get("collision_prims", [])) < 5:
        raise ValueError("Target must be a fixed bin with authored collision walls")
    if not np.isclose(finite_float(container.get("initial_yaw_deg", 0), "bin yaw"), 0, atol=1e-7):
        raise ValueError("Physical cavity AABB requires an unrotated bin")
    size = _xyz(target.get("size_m"), "object size_m")
    if np.any(size <= 0):
        raise ValueError("Pick object dimensions must be positive")
    base_xy = _xy(task["base_offset_xy_m"], "base_offset_xy_m")
    placement_xy = _xy(task.get("placement_offset_xy_m", [0, 0]), "placement_offset_xy_m")
    grasp_z = finite_float(task["grasp_offset_z_m"], "grasp_offset_z_m")
    grasp_yaw = finite_float(task.get("grasp_yaw_offset_rad", 0), "grasp_yaw_offset_rad")
    settle_steps = positive_int(task["settle_steps"], "settle_steps", 100000)
    return task, target, container, base_xy, placement_xy, grasp_z, grasp_yaw, settle_steps


def _bounds(stage, path):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"Missing scene prim: {path}")
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    region = box.ComputeWorldBound(prim).ComputeAlignedRange()
    low, high = np.asarray(region.GetMin(), float), np.asarray(region.GetMax(), float)
    if not np.isfinite([low, high]).all() or np.any(low >= high):
        raise ValueError(f"Invalid world bounds for {path}")
    return low, high


def _cavity_bounds(stage, path):
    bottom_top = _bounds(stage, path + "/Bottom")[1][2]
    low = np.array([_bounds(stage, path + "/WallXMin")[1][0],
                    _bounds(stage, path + "/WallYMin")[1][1], bottom_top])
    high = np.array([_bounds(stage, path + "/WallXMax")[0][0],
                     _bounds(stage, path + "/WallYMax")[0][1],
                     min(_bounds(stage, path + "/" + wall)[1][2]
                         for wall in ("WallXMin", "WallXMax", "WallYMin", "WallYMax"))])
    if np.any(low >= high):
        raise ValueError(f"Container cavity has invalid bounds: {low}, {high}")
    return low, high


def _rotation(quaternion):
    w, x, y, z = quaternion
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def _pose(view):
    positions, quaternions = view.get_world_poses()
    position = _xyz(np.asarray(positions, float).reshape(-1, 3)[0], "observed position").copy()
    quaternion = np.asarray(quaternions, float).reshape(-1, 4)[0].copy()
    if not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-3):
        raise RuntimeError("Invalid observed PhysX quaternion")
    return position, quaternion


def _action_record(action, phase, event):
    fields = {}
    for name in ("joint_positions", "joint_velocities", "joint_efforts", "joint_indices"):
        value = getattr(action, name, None)
        fields[name] = None if value is None else np.asarray(value).tolist()
    return {"phase": phase, "controller_event": event, "articulation_action": fields,
            "application": "ArticulationController.apply_action_before_PhysX_step"}


def _failure_reason(result):
    if not result.get("motiongen_sequence_done", result["controller_done"]):
        return "controller_incomplete"
    for condition, name in ((not result["two_finger_contact_observed"], "missing_two_finger_contact"),
                            (result["lift_held_frames"] < result["required_held_frames"], "insufficient_held_lift"),
                            (result["sustained_two_finger_contact_frames"] < result["required_held_frames"],
                             "insufficient_sustained_contact"),
                            (result["max_grip_relative_slip_m"] is None or
                             result["max_grip_relative_slip_m"] > .035, "excessive_grip_slip"),
                            (not result["release_xy_inside_bin"], "release_outside_bin"),
                            (result["release_object_speed_m_s"] is None or
                             result["release_object_speed_m_s"] > .3, "release_too_fast"),
                            (not result["object_inside_bin"], "object_outside_bin"),
                            (not result["object_settled"], "object_not_settled")):
        if condition:
            return name
    return None


def _run_curobo_pick_place(backend, robot, session, advance, initial_observation,
                           initial, place, bin_low, bin_high, half_size, max_intervals,
                           visual, world, hand_quaternion):
    """Execute actual MotionGen arm trajectories and physical gripper actions.

    The hand/object offset and orientation come from the recorded Franka
    PickPlaceController pregrasp pose on the original tabletop (not from a
    free-space cuRobo target). No planner result is a physical success result.
    """
    from isaacsim.core.utils.types import ArticulationAction
    from .execution.curobo_backend import MotionPlanningFailure

    observation = initial_observation
    # Measured panda_hand - settled object at the successful baseline pregrasp.
    hand_offset = np.array([-.00054098, -.00009313, .10085070])
    hand_offset_z = hand_offset[2]
    hand_at_pick = initial + hand_offset
    above_pick = hand_at_pick + np.array([0., 0., .10])
    above_bin = np.array([place[0], place[1], bin_high[2] + half_size[2] + hand_offset_z + .08])
    hand_at_place = place + np.array([0., 0., hand_offset_z])
    phases = (("approach", 0, above_pick, True),
              ("descend", 2, hand_at_pick, False),
              ("clearance", 4, hand_at_pick + np.array([0., 0., .055]), False),
              ("lift", 4, above_pick, False),
              ("transfer", 5, above_bin, False),
              ("lower", 6, hand_at_place, False))
    seen = [False, False]
    held_frames = contact_holding_frames = intervals = 0
    max_lift = max_slip = 0.
    grip_relative = None
    release_xy_ok = False
    release_speed = None
    trajectory_samples = []
    last_action = None
    failure = None

    def step(action, phase, event):
        nonlocal observation, intervals, last_action, max_lift, max_slip
        nonlocal grip_relative, held_frames, contact_holding_frames
        if intervals >= max_intervals:
            return False
        observation = advance(action, phase, event)
        last_action = action
        intervals += 1
        position = observation["object"]["xyz_m"]
        lift = float(position[2] - initial[2])
        max_lift = max(max_lift, lift)
        if 3 <= event <= 6:
            forces = np.linalg.norm(observation["finger_object_force_n"], axis=1)
            for index in range(2):
                seen[index] |= bool(forces[index] > .01)
            if event in (4, 5, 6) and np.all(forces > .01) and lift > .03:
                contact_holding_frames += 1
        if event in (4, 5, 6) and lift > .02:
            relative = _rotation(observation["robot"]["hand_quaternion_wxyz"]).T @ (
                position - observation["robot"]["hand_xyz_m"])
            if grip_relative is None:
                grip_relative = relative
            else:
                max_slip = max(max_slip, float(np.linalg.norm(relative - grip_relative)))
        if event in (4, 5) and lift >= .05:
            held_frames += 1
        return True

    def mark(phase, event):
        trajectory_samples.append({
            "phase": phase, "event": event, "step": intervals,
            "physics_step": session.step_index,
            "object_xyz_m": observation["object"]["xyz_m"].tolist(),
            "ee_xyz_m": observation["robot"]["ee_xyz_m"].tolist(),
            "hand_xyz_m": observation["robot"]["hand_xyz_m"].tolist()})
        if visual is not None:
            visual.capture(world, f"phase_{event:02d}")

    for phase, event, hand_goal, include_pick in phases:
        if phase == "lift":
            forces = np.linalg.norm(observation["finger_object_force_n"], axis=1)
            if not np.all(forces > .01) or observation["object"]["xyz_m"][2] - initial[2] < .03:
                failure = "object_not_physically_held_after_clearance"
                break
            # The block initially touches the table. Attach its collision spheres
            # only after the *physical* clearance move; otherwise the attached
            # geometry collides with the table at the lift plan's start state.
            backend.attach(observation["robot"]["joint_positions"])
        mark(phase, event)
        try:
            path = backend.plan(joint_positions=observation["robot"]["joint_positions"],
                                goal_xyz=hand_goal, goal_quaternion_wxyz=hand_quaternion,
                                phase=phase, include_pick=include_pick)
        except MotionPlanningFailure as error:
            failure = str(error)
            break
        samples = int(np.ceil(path.duration * session.control_hz))
        for index in range(1, samples + 1):
            arm = path.sample(min(index / session.control_hz, path.duration))
            action = ArticulationAction(joint_positions=arm, joint_indices=np.arange(7))
            if not step(action, phase, event):
                failure = "control_interval_budget_exhausted"
                break
        if phase == "approach" and failure is None:
            # A sampled trajectory ends before the position drive converges.
            # Do not descend with the fingers several millimetres off-center.
            for _ in range(session.control_hz):
                error = np.linalg.norm(observation["robot"]["hand_xyz_m"] - hand_goal)
                if error <= .0015:
                    break
                if not step(action, "approach_settle", event):
                    failure = "control_interval_budget_exhausted"
                    break
            else:
                failure = "approach_not_converged"
        if failure:
            break
        if phase == "descend":
            mark("close", 3)
            for _ in range(max(1, session.control_hz // 2)):
                if not step(robot.gripper.forward(action="close"), "gripper_close", 3):
                    failure = "control_interval_budget_exhausted"
                    break
            if failure:
                break
    if failure is None:
        mark("release", 7)
        position = observation["object"]["xyz_m"]
        quat = observation["object"]["quaternion_wxyz"]
        radius = np.abs(_rotation(quat)) @ half_size
        release_xy_ok = bool(np.all(position[:2] - radius[:2] >= bin_low[:2] + .003) and
                             np.all(position[:2] + radius[:2] <= bin_high[:2] - .003))
        for release_index in range(max(1, session.control_hz // 2)):
            if not step(robot.gripper.forward(action="open"), "gripper_open", 7):
                failure = "control_interval_budget_exhausted"
                break
            if release_index == 0:
                release_speed = float(np.linalg.norm(observation["object"]["linear_velocity_m_s"]))
        backend.detach()
    return {"done": failure is None, "failure": failure, "observation": observation,
            "trajectory": trajectory_samples, "intervals": intervals, "last_action": last_action,
            "seen": seen, "held_frames": held_frames, "contact_holding_frames": contact_holding_frames,
            "max_lift": max_lift, "max_slip": max_slip, "grip_relative": grip_relative,
            "release_xy_ok": release_xy_ok, "release_speed": release_speed,
            "place_hand_xyz_m": hand_at_place.tolist(),
            "approach_hand_z_m": float(above_pick[2])}


def run_franka_demonstration(args):
    """Execute the explicitly selected Franka backend in one real-physics episode."""
    from .adapters.isaac_sensors import IsaacSensorRig
    from .adapters.isaac_world import IsaacObjectStateSampler, IsaacSession
    from .recording import EpisodeWriter
    from .provenance import (archive_asset_dependencies, archive_configuration,
                             configuration_record, load_randomization, apply_pending_cameras)

    physics_hz = positive_int(args.physics_hz, "physics_hz", 10000)
    control_hz = positive_int(args.control_hz, "control_hz", 10000)
    if physics_hz % control_hz:
        raise ValueError("physics_hz must be divisible by control_hz")
    if not isinstance(args.headless, bool):
        raise ValueError("headless must be boolean")
    if getattr(args, "hold_gui", False) and args.headless:
        raise ValueError("--hold-gui requires --gui")
    scene, bindings, scene_manifest = scene_bundle(args.scene)
    task, target, container, base_offset, placement_offset, grasp_offset, grasp_yaw_offset, settle_steps = _task_spec(args.task, bindings)
    backend_name = getattr(args, "backend", None) or task.get("backend", "franka_pickplace")
    if backend_name not in ("franka_pickplace", "curobo_motiongen"):
        raise ValueError(f"Unknown Franka task backend: {backend_name}")
    if backend_name == "curobo_motiongen" and abs(grasp_offset) > 1e-8:
        raise ValueError("cuRobo hand/object calibration requires grasp_offset_z_m == 0")
    config = camera_config(args.config, getattr(args, "camera", None))
    max_intervals = positive_int(args.max_intervals, "max_intervals", 100000)
    settle_intervals = (settle_steps + args.physics_hz // args.control_hz - 1) // (args.physics_hz // args.control_hz)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Episode output exists: {output}")
    writer = EpisodeWriter(output, max_actions=max_intervals + 2 * settle_intervals)
    manifest = {"schema_version": 1, "feature_id": "franka_pick_place",
                "execution_mode": "contact_physics", "simulator": "IsaacSim5.1/PhysX",
                "controller": ("IsaacSim_Franka_PickPlaceController" if backend_name == "franka_pickplace"
                               else "cuRobo_v0.7.6_MotionGen.plan_single"), "backend": backend_name,
                "physics_hz": args.physics_hz,
                "control_hz": args.control_hz, "camera_ids": [item["id"] for item in config["cameras"]],
                "scene": scene_manifest, "task_sha256": hashlib.sha256(Path(args.task).read_bytes()).hexdigest(),
                "contact_evidence": "pairwise_finger_object_force_gt_0.01N"}
    writer.reserve(manifest)
    try:
        randomization = load_randomization(args.scene)
        archive_configuration(output, configuration_record(args, scene, config, task=task), randomization)
        with IsaacSession(scene, headless=args.headless, physics_hz=physics_hz,
                          control_hz=control_hz,
                          defer_app_shutdown=getattr(args, "_defer_app_shutdown", False)) as session:
            rig = sampler = visual = robot = obj = hand = fingers = controller = backend = None
            try:
                from isaacsim.core.prims import RigidPrim
                from isaacsim.core.utils.rotations import euler_angles_to_quat
                from isaacsim.core.utils.types import ArticulationAction
                from isaacsim.robot.manipulators.examples.franka.franka import Franka
                # The original PickPlaceController is constructed only for the baseline backend.
                from isaacsim.storage.native import get_assets_root_path
                from pxr import UsdPhysics

                stage, world = session.stage, session.world
                if stage.GetPrimAtPath("/World/G2").IsValid():
                    raise ValueError("Franka run rejects scenes containing G2")
                if stage.GetPrimAtPath("/World/Franka").IsValid():
                    raise ValueError("Input scene already contains Franka")
                pick_path, bin_path = target["rigid_body_prim"], container["prim_path"]
                table_low, table_high = _bounds(stage, "/World/Table/Top")
                object_low, object_high = _bounds(stage, pick_path + "/Geometry")
                half_size = (object_high - object_low) / 2
                bin_low, bin_high = _cavity_bounds(stage, bin_path)
                if not np.allclose(session.pose_of(bin_path)[:3, :3], np.eye(3), atol=1e-7):
                    raise ValueError("World-AABB cavity requires an axis-aligned unrotated bin")
                expected_walls = {bin_path + "/" + name for name in
                                  ("Bottom", "WallXMin", "WallXMax", "WallYMin", "WallYMax")}
                if set(container["collision_prims"]) != expected_walls:
                    raise ValueError("Target bin collision bindings differ from measured cavity")
                for path in ("/World/Table/Top", pick_path + "/Geometry", *sorted(expected_walls)):
                    prim = stage.GetPrimAtPath(path)
                    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI) or not bool(
                            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()):
                        raise ValueError(f"Missing enabled physical collision: {path}")
                base = np.array([(object_low[0] + object_high[0]) / 2 + base_offset[0],
                                 (object_low[1] + object_high[1]) / 2 + base_offset[1], table_high[2]])
                if np.any(base[:2] < table_low[:2] + .16) or np.any(base[:2] > table_high[:2] - .16):
                    raise ValueError(f"Robot base {base} is too close to table edge")
                if np.any(2 * half_size > bin_high - bin_low):
                    raise ValueError("Pick object is too large for the target cavity")
                asset_root = get_assets_root_path()
                if asset_root is None:
                    raise RuntimeError("Isaac Sim Franka assets are unavailable")
                archive_asset_dependencies(output, scene)
                robot = world.scene.add(Franka(
                    prim_path="/World/Franka", name="tabletop_franka",
                    usd_path=asset_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                    position=base, gripper_open_position=np.array([.04, .04]),
                    gripper_closed_position=np.array([0., 0.])))
                obj = world.scene.add(RigidPrim(prim_paths_expr=pick_path, name="pick_object",
                                                reset_xform_properties=False))
                hand = world.scene.add(RigidPrim(prim_paths_expr="/World/Franka/panda_hand",
                                                 name="hand_frame", reset_xform_properties=False))
                finger_paths = ["/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger"]
                for path in finger_paths:
                    if not stage.GetPrimAtPath(path).IsValid():
                        raise ValueError(f"Franka finger prim missing: {path}")
                fingers = world.scene.add(RigidPrim(
                    prim_paths_expr=finger_paths, name="finger_target_contacts",
                    reset_xform_properties=False, contact_filter_prim_paths_expr=[[pick_path], [pick_path]],
                    track_contact_forces=True, max_contact_count=128))
                rig = IsaacSensorRig(session, config, dict(bindings, franka={"prim_path": "/World/Franka"}))
                randomization = apply_pending_cameras(session, randomization)
                from .config import write_json
                write_json(output / "randomization.json", randomization)
                sampler = IsaacObjectStateSampler(session, bindings, object_ids=[task["object_id"]])
                if not args.headless:
                    from visual import VisualEvidence
                    center = np.array([(table_low[0] + table_high[0]) / 2,
                                       (table_low[1] + table_high[1]) / 2, table_high[2]])
                    visual = VisualEvidence(stage, output / "stage_frames", center)
                    from omni.kit.viewport.utility import get_active_viewport
                    viewport = get_active_viewport()
                    if viewport is None:
                        raise RuntimeError("Isaac Sim GUI has no active viewport")
                    viewport.camera_path = visual.path
                robot.gripper.set_default_state(np.array([.04, .04]))
                session.initialize()
                # Match the verified Franka baseline: let the articulation settle
                # for one second before taking the controller's initial joint state.
                session.step(min(60, session.physics_hz))
                action_controller = robot.get_articulation_controller()
                initial_joints = np.asarray(robot.get_joint_positions(), float).copy()
                if not np.isfinite(initial_joints).all():
                    raise RuntimeError("Invalid initial joint positions")

                def observe(frame_index, phase, event):
                    position, quaternion = _pose(obj)
                    hand_position, hand_quaternion = _pose(hand)
                    ee_position = _xyz(robot.end_effector.get_world_pose()[0], "end effector")
                    matrix = np.asarray(fingers.get_contact_force_matrix(dt=1 / session.physics_hz), float).copy()
                    if matrix.shape != (2, 1, 3) or not np.isfinite(matrix).all():
                        raise RuntimeError(f"Invalid pairwise finger contact matrix: {matrix.shape}")
                    joints = np.asarray(robot.get_joint_positions(), float).copy()
                    joint_velocity = np.asarray(robot.get_joint_velocities(), float).copy()
                    if not np.isfinite(joints).all() or not np.isfinite(joint_velocity).all():
                        raise RuntimeError("Invalid observed joint state")
                    physical = sampler.read_snapshot()["objects"][task["object_id"]]
                    return {"step_index": session.step_index, "sim_time": session.simulation_time,
                            "phase": phase, "controller_event": event if backend_name == "franka_pickplace" else None,
                            "robot": {"joint_positions": joints, "joint_velocities": joint_velocity,
                                      "ee_xyz_m": ee_position, "hand_xyz_m": hand_position,
                                      "hand_quaternion_wxyz": hand_quaternion},
                            "object": {"xyz_m": position, "quaternion_wxyz": quaternion,
                                       "linear_velocity_m_s": physical["linear_velocity_m_s"],
                                       "angular_velocity_rad_s": physical["angular_velocity_rad_s"],
                                       "contact_summary": physical["contact_summary"]},
                            "finger_object_force_n": matrix[:, 0, :],
                            "sensors": rig.capture(frame_index)}

                frame_index = 0
                observation = observe(frame_index, "initial_settle", None)
                writer.begin(manifest, observation)

                def advance(action, phase, event):
                    nonlocal frame_index, observation
                    recorded_action = _action_record(action, phase, event)
                    if backend_name == "curobo_motiongen":
                        recorded_action["controller_event"] = None
                        recorded_action["motiongen_phase_event"] = event
                    action_controller.apply_action(action)
                    session.step(session.substeps)
                    frame_index += 1
                    observation = observe(frame_index, phase, event)
                    writer.append(recorded_action, observation)
                    return observation

                for _ in range(settle_intervals):
                    advance(ArticulationAction(joint_positions=initial_joints), "initial_settle", None)
                initial = observation["object"]["xyz_m"].copy()
                object_quat = observation["object"]["quaternion_wxyz"]
                if abs(initial[2] - half_size[2] - table_high[2]) > .012:
                    raise ValueError(f"Object did not settle on table: object={initial}, table_top={table_high[2]}")
                w, x, y, z = object_quat
                pick_yaw = float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
                orientation = euler_angles_to_quat(np.array([0., np.pi, pick_yaw + grasp_yaw_offset]))
                place = np.array([(bin_low[0] + bin_high[0]) / 2 + placement_offset[0],
                                  (bin_low[1] + bin_high[1]) / 2 + placement_offset[1],
                                  bin_high[2] + half_size[2] + .015])
                target_h = max(table_high[2] + .30, bin_high[2] + .12)
                if backend_name == "franka_pickplace":
                    from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
                    controller = PickPlaceController(
                        name="official_franka_pick_place", gripper=robot.gripper, robot_articulation=robot,
                        end_effector_initial_height=target_h,
                        events_dt=[.012, .002, .005, .01, .004, .003, .003, .005, .01, .02])
                else:
                    from .execution import BackendSpec, create_motion_backend
                    backend = create_motion_backend(
                        backend_name, session=session, robot=robot, pick_path=pick_path,
                        spec=BackendSpec(backend_name, control_dt=1 / control_hz))
                seen = [False, False]
                held_frames = contact_holding_frames = 0
                max_lift = max_slip = 0.
                grip_relative = None
                release_xy_ok = False
                release_speed = None
                previous_event = -1
                trajectory = []
                controller_intervals = 0
                last_action = None
                for index in range(max_intervals if backend_name == "franka_pickplace" else 0):
                    event = int(controller.get_current_event())
                    if event != previous_event:
                        position, quaternion = observation["object"]["xyz_m"], observation["object"]["quaternion_wxyz"]
                        ee = observation["robot"]["ee_xyz_m"]
                        trajectory.append({"step": index, "physics_step": session.step_index,
                                           "event": event, "object_xyz_m": position.tolist(),
                                           "ee_xyz_m": ee.tolist(),
                                           "hand_xyz_m": observation["robot"]["hand_xyz_m"].tolist(),
                                           "joint_displacement_rad": float(np.linalg.norm(
                                               observation["robot"]["joint_positions"] - initial_joints)),
                                           "xy_error_to_object_m": float(np.linalg.norm(ee[:2] - initial[:2]))})
                        if event == 7:
                            radius = np.abs(_rotation(quaternion)) @ half_size
                            release_xy_ok = bool(np.all(position[:2] - radius[:2] >= bin_low[:2] + .003)
                                                 and np.all(position[:2] + radius[:2] <= bin_high[:2] - .003))
                            # Read the actual last control interval's measured object displacement.
                            release_speed = float(np.linalg.norm(position - prior_position) * args.control_hz) if controller_intervals else None
                        if visual is not None:
                            visual.capture(world, f"phase_{event:02d}")
                        previous_event = event
                    if controller.is_done():
                        break
                    action = controller.forward(
                        picking_position=initial, placing_position=place,
                        current_joint_positions=observation["robot"]["joint_positions"],
                        end_effector_orientation=orientation,
                        end_effector_offset=np.array([0., 0., grasp_offset]))
                    last_action = action
                    prior_position = observation["object"]["xyz_m"]
                    next_observation = advance(action, "pick_place", event)
                    controller_intervals += 1
                    position = next_observation["object"]["xyz_m"]
                    if 3 <= event <= 6:
                        forces = np.linalg.norm(next_observation["finger_object_force_n"], axis=1)
                        for finger_index in range(2):
                            seen[finger_index] = bool(seen[finger_index] or forces[finger_index] > .01)
                        if event in (4, 5, 6) and np.all(forces > .01) and position[2] - initial[2] > .03:
                            contact_holding_frames += 1
                    lift = position[2] - initial[2]
                    if event in (4, 5, 6) and lift > .02:
                        relative = _rotation(next_observation["robot"]["hand_quaternion_wxyz"]).T @ (
                            position - next_observation["robot"]["hand_xyz_m"])
                        if grip_relative is None:
                            grip_relative = relative
                        else:
                            max_slip = max(max_slip, float(np.linalg.norm(relative - grip_relative)))
                    max_lift = max(max_lift, float(lift))
                    if event in (4, 5) and lift >= .05:
                        held_frames += 1
                if backend is not None:
                    outcome = _run_curobo_pick_place(
                        backend, robot, session, advance, observation, initial, place,
                        bin_low, bin_high, half_size, max_intervals, visual, world,
                        orientation)
                    observation = outcome["observation"]
                    trajectory = outcome["trajectory"]
                    controller_intervals = outcome["intervals"]
                    last_action = outcome["last_action"]
                    seen = outcome["seen"]
                    held_frames = outcome["held_frames"]
                    contact_holding_frames = outcome["contact_holding_frames"]
                    max_lift = outcome["max_lift"]
                    max_slip = outcome["max_slip"]
                    grip_relative = outcome["grip_relative"]
                    release_xy_ok = outcome["release_xy_ok"]
                    release_speed = outcome["release_speed"]
                done = bool(controller.is_done()) if controller is not None else outcome["done"]
                final_positions = []
                if done:
                    for _ in range(settle_intervals):
                        advance(last_action if last_action is not None else
                                ArticulationAction(joint_positions=initial_joints),
                                "post_release_settle",
                                int(controller.get_current_event()) if controller is not None else 7)
                        final_positions.append(observation["object"]["xyz_m"].copy())
                if visual is not None:
                    visual.capture(world, "settled")
                final, quat = observation["object"]["xyz_m"], observation["object"]["quaternion_wxyz"]
                radius = np.abs(_rotation(quat)) @ half_size
                inside = bool(np.all(final - radius >= bin_low - .002) and np.all(final + radius <= bin_high + .002))
                tail = max(1, (30 * args.control_hz + 59) // 60)
                settled = bool(len(final_positions) >= tail and
                               np.max(np.linalg.norm(np.asarray(final_positions[-tail:]) - final, axis=1)) < .005)
                result = {
                    "done": True, "success": False, "failure_reason": None,
                    "controller_done": done if controller is not None else None,
                    "two_finger_contact_observed": bool(all(seen)),
                    "finger_contacts": dict(zip(("left", "right"), seen)), "lift_held_frames": held_frames,
                    "required_held_frames": max(1, (10 * control_hz + 59) // 60),
                    "max_lift_m": float(max_lift), "object_inside_bin": inside, "object_settled": settled,
                    "max_grip_relative_slip_m": max_slip if grip_relative is not None else None,
                    "sustained_two_finger_contact_frames": contact_holding_frames,
                    "release_xy_inside_bin": release_xy_ok, "release_object_speed_m_s": release_speed,
                    "initial_xyz_m": initial.tolist(), "final_xyz_m": final.tolist(),
                    "final_ee_to_place_error_m": float(np.linalg.norm(observation["robot"]["ee_xyz_m"] - place)),
                    "final_ee_xy_to_place_error_m": float(np.linalg.norm(observation["robot"]["ee_xyz_m"][:2] - place[:2])),
                    "table_top_z_m": float(table_high[2]), "franka_base_xyz_m": base.tolist(),
                    "bin_interior_min_m": bin_low.tolist(), "bin_interior_max_m": bin_high.tolist(),
                    "object_half_extents_m": half_size.tolist(), "place_target_xyz_m": place.tolist(),
                    "placement_offset_xy_m": placement_offset.tolist(), "approach_target_z_m": float(target_h),
                    "grasp_offset_z_m": grasp_offset, "gripper_yaw_alignment_rad": pick_yaw,
                    "images": visual.names if visual is not None else [], "event_samples": trajectory,
                    "controller_intervals": controller_intervals, "settle_intervals_each": settle_intervals,
                    "recorded_actions": frame_index, "scene": str(Path(args.scene).resolve())}
                result["final_object_speed_m_s"] = float(np.linalg.norm(observation["object"]["linear_velocity_m_s"]))
                if backend is not None:
                    hand_target = np.asarray(outcome["place_hand_xyz_m"])
                    result.update(planner="cuRobo MotionGen.plan_single", robot_config="franka.yml",
                                  motiongen_plans=backend.plans, motiongen_sequence_done=done,
                                  motiongen_failure=outcome["failure"],
                                  planning_collision_policy=(
                                      "all authored colliders; pick cuboid excluded during intentional "
                                      "grasp and represented by attached spheres during carry"),
                                  place_target_xyz_m=hand_target.tolist(),
                                  approach_target_z_m=outcome["approach_hand_z_m"],
                                  final_ee_to_place_error_m=None,
                                  final_ee_xy_to_place_error_m=None,
                                  hand_place_target_xyz_m=hand_target.tolist(),
                                  final_hand_to_place_error_m=float(np.linalg.norm(
                                      observation["robot"]["hand_xyz_m"] - hand_target)))
                result["failure_reason"] = (outcome["failure"] if backend is not None and outcome["failure"]
                                            else _failure_reason(result))
                if backend is None:
                    result["failure_category"] = (
                        ("timeout" if not done else "physical") if result["failure_reason"] else None)
                else:
                    result["failure_category"] = (
                        "planning" if result["failure_reason"] and result["failure_reason"].startswith("MotionGen ")
                        else "timeout" if result["failure_reason"] == "control_interval_budget_exhausted"
                        else "physical" if done and result["failure_reason"] else "execution"
                        if result["failure_reason"] else None)
                result["success"] = result["failure_reason"] is None
            finally:
                import sys
                active_error = sys.exc_info()[1]
                errors = []
                if backend is not None:
                    try:
                        backend.cleanup()
                    except BaseException as error:
                        errors.append(error)
                for component in (visual, rig, sampler):
                    if component is not None:
                        try:
                            component.close()
                        except BaseException as error:
                            errors.append(error)
                visual = rig = sampler = robot = obj = hand = fingers = controller = action_controller = None
                if errors:
                    if active_error is not None:
                        for error in errors:
                            active_error.add_note(f"Resource cleanup failed: {error}")
                    else:
                        raise RuntimeError("Franka resource cleanup failed") from errors[0]
            writer.finish(result)
            if getattr(args, "hold_gui", False):
                session.world.pause()
                while session.app.is_running():
                    session.world.pause()
                    session.app.update()
        return {"output": str(output), "episode_path": str(output), **result}
    except BaseException as error:
        if writer._state not in ("complete", "failed"):
            try:
                writer.fail(error)
            except Exception as recording_error:
                error.add_note(f"Could not persist host failure: {recording_error}")
        raise
    finally:
        writer.close()
