"""Run Isaac Sim's Franka pick/place controller on an existing tabletop USD bundle.

No project package, G2 profile, cuRobo configuration, or synthetic object is used.
Run with Isaac Sim 5.1's python.sh, not a system Python interpreter.
"""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="Directory containing scene.usda and bindings.json")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON path")
    parser.add_argument("--gui", action="store_true", help="Display the Isaac Sim viewport")
    parser.add_argument("--gui-hold-seconds", type=float, default=20.0,
                        help="Keep the result visible in the GUI after the episode")
    parser.add_argument("--capture", action="store_true", help="Save rendered PNGs at each phase transition")
    parser.add_argument("--grasp-offset-z", type=float, default=0.0,
                        help="World Z offset applied to the official controller's tool target (metres)")
    parser.add_argument("--base-offset-x", type=float, default=0.0)
    parser.add_argument("--base-offset-y", type=float, default=-0.28)
    parser.add_argument("--bin-xy", type=float, nargs=2, metavar=("X", "Y"),
                        help="Move only the bin in the USD session layer for a reachability diagnostic")
    parser.add_argument("--max-steps", type=int, default=3600)
    return parser.parse_args()


def bounds(stage, path):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"Missing scene prim: {path}")
    world_range = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    return np.asarray(world_range.GetMin(), dtype=float), np.asarray(world_range.GetMax(), dtype=float)


def cavity_bounds(stage, path):
    bottom, bottom_max = bounds(stage, path + "/Bottom")
    x_min_max = bounds(stage, path + "/WallXMin")[1]
    x_max_min = bounds(stage, path + "/WallXMax")[0]
    y_min_max = bounds(stage, path + "/WallYMin")[1]
    y_max_min = bounds(stage, path + "/WallYMax")[0]
    wall_tops = [bounds(stage, path + "/" + wall)[1][2] for wall in ("WallXMin", "WallXMax", "WallYMin", "WallYMax")]
    low = np.array([x_min_max[0], y_min_max[1], bottom_max[2]])
    high = np.array([x_max_min[0], y_max_min[1], min(wall_tops)])
    if np.any(low >= high):
        raise ValueError(f"Container cavity has invalid bounds: {low}, {high}")
    return low, high


def quaternion_rotation(quaternion):
    # Isaac poses use wxyz. The complete rotated box, not only its centre,
    # must be inside the container.
    w, x, y, z = quaternion
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return rotation


def half_extent_world(half_size, quaternion):
    # Isaac poses use wxyz. The complete rotated box, not only its centre,
    # must be inside the container.
    return np.abs(quaternion_rotation(quaternion)) @ half_size


def run(args, app):
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.robot.manipulators.examples.franka.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
    import omni.usd
    from pxr import Gf, Usd

    scene = args.scene.resolve(strict=True)
    bindings = json.loads((scene / "bindings.json").read_text(encoding="utf-8"))["objects"]
    pick_path = bindings["pick_object"]["rigid_body_prim"]
    bin_path = bindings["target_bin"]["prim_path"]
    if not open_stage(str(scene / "scene.usda")):
        raise RuntimeError("Cannot open scene.usda")
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/World/G2").IsValid():
        raise ValueError("Load the standalone tabletop scene, not a scene containing G2")
    if stage.GetPrimAtPath("/World/Franka").IsValid():
        raise ValueError("Input scene already contains a Franka robot")
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    if args.bin_xy is not None:
        bin_prim = stage.GetPrimAtPath(bin_path)
        translate = bin_prim.GetAttribute("xformOp:translate")
        if not translate.IsValid():
            raise ValueError("Bin must have a root translation to relocate it")
        old = translate.Get()
        translate.Set(Gf.Vec3d(args.bin_xy[0], args.bin_xy[1], old[2]))

    table_low, table_high = bounds(stage, "/World/Table/Top")
    object_low, object_high = bounds(stage, pick_path + "/Geometry")
    half_size = (object_high - object_low) / 2
    bin_low, bin_high = cavity_bounds(stage, bin_path)
    # These are measured WORLD coordinates; the official RMPflow controller
    # accounts for the robot's nonzero base pose itself.
    base = np.array([(object_low[0] + object_high[0]) / 2 + args.base_offset_x,
                     (object_low[1] + object_high[1]) / 2 + args.base_offset_y,
                     table_high[2]])
    if np.any(base[:2] < table_low[:2] + 0.16) or np.any(base[:2] > table_high[:2] - 0.16):
        raise ValueError(f"Robot base {base} is too close to the table edge")
    if np.any(2 * half_size > bin_high - bin_low):
        raise ValueError("Pick object is too large for the target cavity")
    from isaacsim.storage.native import get_assets_root_path
    asset = get_assets_root_path()
    if asset is None:
        raise RuntimeError("Isaac Sim Franka assets are unavailable")

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60, rendering_dt=1.0 / 60)
    robot = world.scene.add(Franka(prim_path="/World/Franka", name="tabletop_franka",
                                   usd_path=asset + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                                   position=base, gripper_open_position=np.array([0.04, 0.04]),
                                   gripper_closed_position=np.array([0.0, 0.0])))
    obj = world.scene.add(RigidPrim(prim_paths_expr=pick_path, name="pick_object", reset_xform_properties=False))
    hand = world.scene.add(RigidPrim(prim_paths_expr="/World/Franka/panda_hand",
                                    name="hand_frame", reset_xform_properties=False))
    finger_paths = ["/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger"]
    for path in finger_paths:
        if not stage.GetPrimAtPath(path).IsValid():
            raise ValueError(f"Franka finger prim missing: {path}")
    fingers = world.scene.add(RigidPrim(
        prim_paths_expr=finger_paths, name="finger_target_contacts", reset_xform_properties=False,
        contact_filter_prim_paths_expr=[[pick_path], [pick_path]],
        track_contact_forces=True, max_contact_count=128))
    visual = None
    if args.capture:
        from visual import VisualEvidence
        center = np.array([(table_low[0] + table_high[0]) / 2,
                           (table_low[1] + table_high[1]) / 2, table_high[2]])
        visual = VisualEvidence(stage, args.output.with_suffix("").as_posix() + "_frames", center)
        if args.gui:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("Isaac Sim GUI has no active viewport")
            viewport.camera_path = visual.path
    robot.gripper.set_default_state(np.array([0.04, 0.04]))
    world.reset()
    for _ in range(60):
        world.step(render=args.gui)

    def pose():
        p, q = obj.get_world_poses()
        return np.asarray(p, dtype=float).reshape(-1, 3)[0], np.asarray(q, dtype=float).reshape(-1, 4)[0]
    def end_effector_position():
        return np.asarray(robot.end_effector.get_world_pose()[0], dtype=float)

    def hand_position():
        return np.asarray(hand.get_world_poses()[0], dtype=float).reshape(-1, 3)[0]

    initial, object_quat = pose()
    if abs(initial[2] - half_size[2] - table_high[2]) > 0.012:
        raise ValueError(f"Object did not settle on table: object={initial}, table_top={table_high[2]}")
    w, x, y, z = object_quat
    pick_yaw = float(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
    orientation = euler_angles_to_quat(np.array([0.0, np.pi, pick_yaw]))
    # The gripper and wrist must stay ABOVE the bin walls. Open the fingers
    # above the cavity and let the object fall into it under PhysX gravity.
    place = np.array([(bin_low[0] + bin_high[0]) / 2,
                      (bin_low[1] + bin_high[1]) / 2,
                      bin_high[2] + half_size[2] + 0.015])
    controller = PickPlaceController(
        name="official_franka_pick_place", gripper=robot.gripper, robot_articulation=robot,
        end_effector_initial_height=max(table_high[2] + 0.30, bin_high[2] + 0.12),
        events_dt=[0.012, 0.002, 0.005, 0.01, 0.004, 0.003, 0.003, 0.005, 0.01, 0.02])
    action_controller = robot.get_articulation_controller()
    initial_joints = np.asarray(robot.get_joint_positions(), dtype=float).copy()
    target_h = max(table_high[2] + 0.30, bin_high[2] + 0.12)
    print("SCENE", scene, "TABLE_TOP", table_high[2], "BASE", base, "PICK", initial, "PLACE", place, flush=True)
    seen = [False, False]
    held_frames = 0
    max_lift = 0.0
    previous_event = -1
    trajectory = []
    grip_relative = None
    max_slip = 0.0
    contact_holding_frames = 0
    release_xy_ok = False
    release_velocity_m_s = None
    prior_object_position = None
    for step in range(args.max_steps):
        event = controller.get_current_event()
        if event != previous_event:
            position, _ = pose()
            ee = end_effector_position()
            sample = {"step": step, "event": event, "object_xyz_m": position.tolist(),
                      "ee_xyz_m": ee.tolist(), "hand_xyz_m": hand_position().tolist(),
                      "joint_displacement_rad": float(np.linalg.norm(
                          np.asarray(robot.get_joint_positions(), dtype=float) - initial_joints)),
                      "xy_error_to_object_m": float(np.linalg.norm(ee[:2] - initial[:2]))}
            trajectory.append(sample)
            print("EVENT", json.dumps(sample), flush=True)
            if event == 7:
                _, object_quat = pose()
                release_radius = half_extent_world(half_size, object_quat)
                release_xy_ok = bool(np.all(position[:2] - release_radius[:2] >= bin_low[:2] + 0.003)
                                     and np.all(position[:2] + release_radius[:2] <= bin_high[:2] - 0.003))
                release_velocity_m_s = float(np.linalg.norm(position - prior_object_position) * 60) if prior_object_position is not None else None
            if visual is not None:
                visual.capture(world, f"phase_{event:02d}")
            previous_event = event
        if controller.is_done():
            break
        position, _ = pose()
        action = controller.forward(picking_position=initial, placing_position=place,
                                    current_joint_positions=robot.get_joint_positions(),
                                    end_effector_orientation=orientation,
                                    end_effector_offset=np.array([0.0, 0.0, args.grasp_offset_z]))
        action_controller.apply_action(action)
        world.step(render=args.gui)
        if event == 1 and step % 50 == 0:
            print("APPROACH_TRACE", step, "hand", hand_position().tolist(),
                  "object", position.tolist(), flush=True)
        position, _ = pose()
        prior_object_position = position.copy()
        if 3 <= event <= 6:
            matrix = np.asarray(fingers.get_contact_force_matrix(dt=1.0 / 60))
            if matrix.shape != (2, 1, 3):
                raise RuntimeError(f"Unexpected finger contact shape: {matrix.shape}")
            forces = np.linalg.norm(matrix[:, 0], axis=1)
            for i in range(2):
                seen[i] = bool(seen[i] or forces[i] > 0.01)
            if event in (4, 5, 6) and np.all(forces > 0.01) and position[2] - initial[2] > 0.03:
                contact_holding_frames += 1
        if event in (4, 5, 6) and position[2] - initial[2] > 0.02:
            hpos, hquat = hand.get_world_poses()
            relative = quaternion_rotation(np.asarray(hquat, dtype=float).reshape(-1, 4)[0]).T @ (
                position - np.asarray(hpos, dtype=float).reshape(-1, 3)[0])
            if grip_relative is None:
                grip_relative = relative
            else:
                max_slip = max(max_slip, float(np.linalg.norm(relative - grip_relative)))
        lift = position[2] - initial[2]
        max_lift = max(max_lift, lift)
        if event in (4, 5) and lift >= 0.05:
            held_frames += 1
    else:
        print("STEP_BUDGET_EXHAUSTED", args.max_steps, flush=True)

    done = controller.is_done()
    # PhysX continues after release. A controller's is_done() alone is NOT
    # evidence that the object was picked or placed.
    final_positions = []
    if done:
        for _ in range(120):
            world.step(render=args.gui)
            final_positions.append(pose()[0])
    if visual is not None:
        visual.capture(world, "settled")
        visual.close()
    final, quat = pose()
    radius = half_extent_world(half_size, quat)
    inside = bool(np.all(final - radius >= bin_low - 0.002) and np.all(final + radius <= bin_high + 0.002))
    settled = bool(len(final_positions) >= 30 and
                   np.max(np.linalg.norm(np.asarray(final_positions[-30:]) - final, axis=1)) < 0.005)
    lifted = held_frames >= 10
    both_contacts = bool(all(seen))
    success = bool(done and lifted and both_contacts and contact_holding_frames >= 10
                   and grip_relative is not None and max_slip <= 0.035
                   and release_xy_ok and release_velocity_m_s is not None
                   and release_velocity_m_s <= 0.3 and inside and settled)
    result = {
        "success": success, "controller_done": bool(done), "two_finger_contact_observed": both_contacts,
        "finger_contacts": dict(zip(("left", "right"), seen)), "lift_held_frames": held_frames,
        "max_lift_m": float(max_lift), "object_inside_bin": inside, "object_settled": settled,
        "max_grip_relative_slip_m": max_slip,
        "sustained_two_finger_contact_frames": contact_holding_frames,
        "release_xy_inside_bin": release_xy_ok,
        "release_object_speed_m_s": release_velocity_m_s,
        "initial_xyz_m": initial.tolist(), "final_xyz_m": final.tolist(),
        "table_top_z_m": float(table_high[2]), "franka_base_xyz_m": base.tolist(),
        "bin_interior_min_m": bin_low.tolist(), "bin_interior_max_m": bin_high.tolist(),
        "object_half_extents_m": half_size.tolist(), "place_target_xyz_m": place.tolist(),
        "bin_xy_override_m": args.bin_xy,
        "approach_target_z_m": float(target_h),
        "grasp_offset_z_m": args.grasp_offset_z,
        "gripper_yaw_alignment_rad": pick_yaw,
        "images": visual.names if visual is not None else [],
        "event_samples": trajectory, "steps": step + 1, "scene": str(scene),
    }
    print("RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    args = arguments()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.gui_hold_seconds < 0:
        raise ValueError("--gui-hold-seconds must be nonnegative")
    # SimulationApp must be constructed before any Isaac/pxr import.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.gui})
    try:
        try:
            result = run(args, app)
        except Exception as exc:
            traceback.print_exc()
            result = {"success": False, "error": f"{type(exc).__name__}: {exc}", "scene": str(args.scene)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.gui and args.gui_hold_seconds:
            from time import monotonic
            end = monotonic() + args.gui_hold_seconds
            while monotonic() < end:
                app.update()
        return 0 if result["success"] else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
