"""Plan a collision-aware pregrasp in the original tabletop USD; do not execute it.

Run with Isaac Sim's python.sh, cuRobo v0.7.6 on PYTHONPATH, and a CUDA GPU.
The output JSON is evidence of planning only: no grasp, trajectory execution,
contact, lift, or placement is attempted by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np


# Measured from the successful ORIGINAL-scene PhysX episode, not a cuRobo plan.
# Event 2 is the observed pregrasp panda_hand pose relative to the settled block.
CALIBRATION = {
    "source": "verified_original_scene_PhysX_episode_event_2_pregrasp",
    "object_world_xyz_m": [-0.06112306, 0.21515608, 0.78499973],
    "object_yaw_rad": 0.20668536,
    "hand_world_xyz_m": [-0.06166404, 0.21506295, 0.88585043],
    "hand_quaternion_wxyz": [0.00052861, 0.99452007, 0.10454434, 0.00039371],
}
# This joint state was observed at settled event 0; it seeds the actual robot.
RECORDED_START_Q = [0.01525379, -0.56900465, 0.00348463, -2.814394,
                    -0.00024997, 3.02622390, 0.74147195, 0.0399972, 0.0399972]
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
REQUIRED_COLLIDERS = {"/World/Table/Top", "/World/Objects/pick_object/Geometry",
                      *(f"/World/Objects/target_bin/{name}" for name in
                        ("Bottom", "WallXMin", "WallXMax", "WallYMin", "WallYMax"))}


def quaternion(matrix):
    """Unit wxyz quaternion from a right-handed, column-vector rotation matrix."""
    from pxr import Gf

    # USD's Gf matrix convention is row-vector; transpose the NumPy matrix.
    q = Gf.Matrix3d(*matrix.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
    result = np.array([q.GetReal(), *q.GetImaginary()], dtype=float)
    return (result / np.linalg.norm(result)).tolist()


def multiply_quaternions(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], dtype=float)


def transform_of(cache, prim):
    # USD/Gf matrices multiply row vectors; cuRobo and this script use columns.
    return np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T


def scene_cuboids(stage, cache, base_world):
    from pxr import UsdGeom, UsdPhysics
    from curobo.geom.types import Cuboid, WorldConfig

    base_inverse = np.linalg.inv(base_world)
    world_cuboids, measured = [], []
    colliders = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/Franka/") or path == "/World/Franka":
            continue  # Robot is modeled by franka.yml collision spheres, not world cubes.
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is not False:
                colliders[path] = prim
    if not REQUIRED_COLLIDERS <= colliders.keys():
        raise ValueError(f"Original scene collision missing: {sorted(REQUIRED_COLLIDERS - colliders.keys())}")
    if not colliders:
        raise ValueError("No enabled scene collision geometry; refusing empty world")

    for path, prim in sorted(colliders.items()):
        cube = UsdGeom.Cube(prim)
        if not cube:
            raise ValueError(f"Unsupported enabled scene collider, not dropping it: {path}")
        size = cube.GetSizeAttr().Get()
        if size is None or size <= 0:
            raise ValueError(f"Invalid cube size for {path}: {size}")
        world = transform_of(cache, prim)
        scales = np.linalg.norm(world[:3, :3], axis=0)
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError(f"Invalid world scale for {path}")
        rotation = world[:3, :3] / scales
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.999:
            raise ValueError(f"Cannot represent collider as oriented cube without distortion: {path}")
        center_world = world[:3, 3]
        in_base = base_inverse @ world
        base_rotation = in_base[:3, :3] / scales
        pose = [*in_base[:3, 3].tolist(), *quaternion(base_rotation)]
        dims = (scales * size).tolist()
        world_cuboids.append(Cuboid(name=path, pose=pose, dims=dims))
        measured.append({"prim_path": path, "type": "enabled_USD_Cube_collision",
                         "center_world_xyz_m": center_world.tolist(),
                         "quaternion_world_wxyz": quaternion(rotation),
                         "pose_base_xyz_wxyz": pose, "dims_m": dims})
    if len(world_cuboids) != len(colliders):
        raise RuntimeError("Scene collider conversion incomplete")
    return WorldConfig(cuboid=world_cuboids), measured


def compute_goal(object_world, base_world):
    """Rotate the measured hand-to-object offset into the live block yaw frame."""
    measured_obj = np.array(CALIBRATION["object_world_xyz_m"], dtype=float)
    measured_hand = np.array(CALIBRATION["hand_world_xyz_m"], dtype=float)
    old_yaw = CALIBRATION["object_yaw_rad"]
    delta = object_world["yaw_rad"] - old_yaw
    c, s = np.cos(delta), np.sin(delta)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    hand_world = object_world["xyz_m"] + rz @ (measured_hand - measured_obj) + np.array([0., 0., .10])
    qz = [np.cos(delta / 2), 0, 0, np.sin(delta / 2)]
    q_world = multiply_quaternions(qz, CALIBRATION["hand_quaternion_wxyz"])
    q_base_inv = np.array(quaternion(np.linalg.inv(base_world)[:3, :3]))
    q_base = multiply_quaternions(q_base_inv, q_world)
    xyz_base = (np.linalg.inv(base_world) @ np.array([*hand_world, 1.0]))[:3]
    return {"frame": "panda_link0", "ee_link": "panda_hand",
            "world_xyz_m": hand_world.tolist(), "world_quaternion_wxyz": q_world.tolist(),
            "base_xyz_m": xyz_base.tolist(), "base_quaternion_wxyz": (q_base / np.linalg.norm(q_base)).tolist(),
            "approach_clearance_above_recorded_contact_m": 0.10,
            "measured_hand_minus_object_world_xyz_m": (measured_hand - measured_obj).tolist(),
            "live_object_minus_calibration_yaw_rad": float(delta)}


def plan(scene, report, output):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.robot.manipulators.examples.franka.franka import Franka
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Usd, UsdGeom
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        from curobo.util_file import get_robot_configs_path, load_yaml
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo MotionGen requires CUDA; no GPU available")
        config_file = Path(get_robot_configs_path()) / "franka.yml"
        config = load_yaml(str(config_file))["robot_cfg"]["kinematics"]
        if config["base_link"] != "panda_link0" or config["ee_link"] != "panda_hand":
            raise ValueError(f"Unexpected cuRobo Franka base/hand links: {config_file}")
        report["curobo_robot_config"] = {
            "path": str(config_file.resolve(strict=True)),
            "sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
            "base_link": config["base_link"], "ee_link": config["ee_link"]}
        if not open_stage(str(scene)):
            raise RuntimeError(f"Cannot open original scene USD: {scene}")
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
        if stage.GetPrimAtPath("/World/Franka").IsValid() or stage.GetPrimAtPath("/World/G2").IsValid():
            raise ValueError("Expected untouched original tabletop scene, without a robot")
        asset_root = get_assets_root_path()
        if not asset_root:
            raise RuntimeError("Isaac Sim Franka assets are unavailable")
        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
        world.get_physics_context().set_physx_update_transformations_settings(
            update_to_usd=True, update_velocities_to_usd=True)
        object_prim = stage.GetPrimAtPath("/World/Objects/pick_object/Geometry")
        table_prim = stage.GetPrimAtPath("/World/Table/Top")
        if not object_prim.IsValid() or not table_prim.IsValid():
            raise ValueError("Missing table top or block geometry")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        obj_range = bounds.ComputeWorldBound(object_prim).ComputeAlignedRange()
        table_range = bounds.ComputeWorldBound(table_prim).ComputeAlignedRange()
        obj_lo = np.asarray(obj_range.GetMin(), dtype=float)
        obj_hi = np.asarray(obj_range.GetMax(), dtype=float)
        table_lo = np.asarray(table_range.GetMin(), dtype=float)
        table_hi = np.asarray(table_range.GetMax(), dtype=float)
        base = np.array([(obj_lo[0] + obj_hi[0]) / 2,
                         (obj_lo[1] + obj_hi[1]) / 2 - 0.28, table_hi[2]])
        robot = world.scene.add(Franka(
            prim_path="/World/Franka", name="tabletop_franka",
            usd_path=asset_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            position=base, gripper_open_position=np.array([.04, .04]),
            gripper_closed_position=np.array([0., 0.])))
        block = world.scene.add(RigidPrim(prim_paths_expr="/World/Objects/pick_object",
                                          name="pick_object", reset_xform_properties=False))
        hand = world.scene.add(RigidPrim(prim_paths_expr="/World/Franka/panda_hand",
                                         name="hand_frame", reset_xform_properties=False))
        robot.gripper.set_default_state(np.array([.04, .04]))
        world.reset()
        robot.set_joint_positions(np.asarray(RECORDED_START_Q, dtype=float))
        world.play()
        for _ in range(60):  # Replicate the original episode's 1-second physical settle.
            if not app.is_running():
                raise RuntimeError("Isaac Sim stopped during initial settle")
            world.step(render=False)

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        base_prim = stage.GetPrimAtPath("/World/Franka/panda_link0")
        if not base_prim.IsValid():
            raise ValueError("Missing live Franka panda_link0 frame")
        base_world = transform_of(cache, base_prim)
        live_obj_pos, live_obj_quat = block.get_world_poses()
        live_hand_pos, live_hand_quat = hand.get_world_poses()
        live_obj_pos = np.asarray(live_obj_pos, dtype=float).reshape(-1, 3)[0]
        live_obj_quat = np.asarray(live_obj_quat, dtype=float).reshape(-1, 4)[0]
        w, x, y, z = live_obj_quat
        obj_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z)))
        if not np.isfinite([*live_obj_pos, *live_obj_quat]).all():
            raise ValueError("Nonfinite measured block pose")
        if np.linalg.norm(live_obj_pos - np.array(CALIBRATION["object_world_xyz_m"])) > .025:
            raise ValueError("Live block deviates >2.5 cm from calibrated episode: no valid pregrasp calibration")
        world_cfg, obstacles = scene_cuboids(stage, cache, base_world)
        if len(obstacles) < len(REQUIRED_COLLIDERS):
            raise RuntimeError("Not all required collision obstacles imported")
        goal = compute_goal({"xyz_m": live_obj_pos, "yaw_rad": obj_yaw}, base_world)
        q = np.asarray(robot.get_joint_positions(), dtype=float)
        if q.shape != (9,) or not np.isfinite(q).all():
            raise ValueError(f"Invalid live Franka 9-DOF joint state: {q}")
        report.update({
            "robot": {"isaac_usd": asset_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                      "curobo_config": "franka.yml", "base_link": "panda_link0", "ee_link": "panda_hand",
                      "base_world_from_base_matrix": base_world.tolist(),
                      "initial_recorded_joint_positions_9": RECORDED_START_Q,
                      "start_measured_joint_positions_9": q.tolist(), "planned_joint_names": JOINT_NAMES,
                      "start_hand_world_xyz_m": np.asarray(live_hand_pos).reshape(-1, 3)[0].tolist(),
                      "start_hand_world_quaternion_wxyz": np.asarray(live_hand_quat).reshape(-1, 4)[0].tolist()},
            "scene_measurements": {"table_bounds_world_min_m": table_lo.tolist(),
                                   "table_bounds_world_max_m": table_hi.tolist(),
                                   "object_initial_bounds_world_min_m": obj_lo.tolist(),
                                   "object_initial_bounds_world_max_m": obj_hi.tolist(),
                                   "object_settled_world_xyz_m": live_obj_pos.tolist(),
                                   "object_settled_quaternion_wxyz": live_obj_quat.tolist(),
                                   "object_settled_yaw_rad": obj_yaw},
            "calibration": CALIBRATION, "goal": goal, "world_obstacles": obstacles,
            "world_obstacle_count": len(obstacles),
        })
        tensor = TensorDeviceType(device=torch.device("cuda:0"))
        motion = MotionGen(MotionGenConfig.load_from_robot_config(
            "franka.yml", world_cfg, tensor, interpolation_dt=0.02,
            collision_cache={"obb": max(32, len(obstacles) + 4)}))
        motion.warmup()
        start = JointState.from_position(tensor.to_device([q[:7].tolist()]), joint_names=JOINT_NAMES)
        goal_pose = Pose(position=tensor.to_device([goal["base_xyz_m"]]),
                         quaternion=tensor.to_device([goal["base_quaternion_wxyz"]]))
        result = motion.plan_single(start, goal_pose)
        success = bool(result.success.item())
        report.update({"success": success, "status": str(result.status),
                       "waypoints": [], "waypoint_count": 0})
        if success:
            path = result.get_interpolated_plan().position.detach().cpu().numpy()
            path = np.asarray(path).reshape(-1, len(JOINT_NAMES))
            report["waypoints"] = path.tolist()
            report["waypoint_count"] = len(path)
            report["interpolation_dt_s"] = float(result.interpolation_dt)
        return 0 if success else 1
    except Exception as exc:
        report["success"] = False
        report["status"] = "error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        return 1
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"success": report["success"], "status": report["status"],
                          "output": str(output)}), flush=True)
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_scene = Path(__file__).resolve().parents[2] / "scene_original" / "scene.usda"
    parser.add_argument("output", type=Path, help="JSON planning evidence output (overwrites existing file)")
    parser.add_argument("--scene", type=Path, default=default_scene)
    args = parser.parse_args()
    report = {"schema_version": 1, "planner": "cuRobo MotionGen.plan_single",
              "execution_mode": "plan_only_no_physics_grasp_or_execution",
              "success": False, "status": "not_started"}
    exit_code = 1
    try:
        scene = args.scene.resolve(strict=True)
        report["scene_usd"] = str(scene)
        report["scene_sha256"] = hashlib.sha256(scene.read_bytes()).hexdigest()
        report["layout_sha256"] = hashlib.sha256((scene.parent / "layout.usda").read_bytes()).hexdigest()
        exit_code = plan(scene, report, args.output)
    except Exception as exc:
        report["success"] = False
        report["status"] = "error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"success": report["success"], "status": report["status"],
                      "output": str(args.output)}), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
