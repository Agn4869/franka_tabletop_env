"""Isaac Sim dynamic-articulation adapter. No kinematic grasp constraints."""

import hashlib
import json
import time
import numpy as np

from ..manipulation import WorldObservation


class G2RobotAdapter:
    def __init__(self, session, profile, limits, gripper, bindings, object_id):
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaacsim.core.prims import SingleArticulation, RigidPrim
        self.session, self.profile, self.limits, self.gripper = session, profile, limits, gripper
        self.bindings, self.object_id = bindings, object_id
        self.root = profile["root_prim"]
        if session.stage.GetPrimAtPath(self.root).IsValid():
            raise ValueError(f"Robot root already exists: {self.root}")
        prim = UsdGeom.Xform.Define(session.stage, self.root).GetPrim()
        if not prim.GetReferences().AddReference(profile["robot_usd"]):
            raise ValueError("Cannot reference robot asset")
        from pxr import Gf
        if "placement_xyz_m" in profile:
            translation = np.asarray(profile["placement_xyz_m"], dtype=float)
            if translation.shape != (3,) or not np.isfinite(translation).all():
                raise ValueError("placement_xyz_m must be finite XYZ metres")
            UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*translation))
        yaw = float(profile.get("placement_yaw_rad", 0.0))
        if not np.isfinite(yaw):
            raise ValueError("placement_yaw_rad must be finite")
        UsdGeom.XformCommonAPI(prim).SetRotate((0.0, 0.0, np.degrees(yaw)))
        self._link_cache = {}
        self.base = self.resolve_link(profile["base_link"])
        self.ee = self.resolve_link(profile["end_effector_link"])
        self.fingers = [self.resolve_link(path) for path in profile["gripper"]["contact_link_paths"]]
        gripper_joint = session.stage.GetPrimAtPath(self.resolve_link(gripper.joint_name))
        drive = UsdPhysics.DriveAPI(gripper_joint, "angular")
        if not drive or not drive.GetStiffnessAttr().Get():
            raise ValueError("Asset must contain an angular position drive for the gripper")
        drive.CreateMaxForceAttr(gripper.max_effort)
        self.robot = SingleArticulation(prim_path=self.root, name="g2_robot")
        session.world.scene.add(self.robot)
        target = bindings[object_id].get("rigid_body_prim")
        if not target:
            raise ValueError("Grasp target must be a dynamic rigid body")
        self.target = RigidPrim(prim_paths_expr=target, name="grasp_target", reset_xform_properties=False)
        session.world.scene.add(self.target)
        self._drawer_views = {}
        for index, (item_id, binding) in enumerate(bindings.items()):
            if binding.get("joint_prim"):
                joint = UsdPhysics.PrismaticJoint.Get(session.stage, binding["joint_prim"])
                if not joint or not binding.get("fixed_body_prim") or not binding.get("body_prim"):
                    raise ValueError("Scene joint observations currently require a bound prismatic drawer")
                view = RigidPrim(prim_paths_expr=binding["body_prim"], name=f"observed_drawer_{index}",
                                 reset_xform_properties=False)
                session.world.scene.add(view)
                self._drawer_views[item_id] = (joint, view)
        self.body_paths = [str(item.GetPath()) for item in Usd.PrimRange(prim)
                           if item.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not all(path in self.body_paths for path in self.fingers):
            raise ValueError("contact_link_paths must name actual robot rigid bodies, not visual meshes")
        environment_roots = sorted({value.get("prim_path", value.get("root_prim"))
                                   for value in bindings.values()
                                   if value.get("prim_path", value.get("root_prim"))})
        environment_roots += ["/World/Table", "/World/Ground"]
        # PhysX contact filters do not implicitly expand a static Xform subtree.
        # Use dynamic actor paths and individual static collider paths, retaining
        # a separate mapping to logical objects for allowed-pair decisions.
        filter_roots = {}
        for root_path in environment_roots:
            root_prim = session.stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                continue
            for item in Usd.PrimRange(root_prim):
                if not item.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                actor = item
                ancestor = item
                while ancestor and str(ancestor.GetPath()).startswith(root_path):
                    if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                        actor = ancestor
                        break
                    ancestor = ancestor.GetParent()
                filter_roots[str(actor.GetPath())] = root_path
        self.environment = list(filter_roots)
        self.filter_roots = filter_roots
        self.contact_view = RigidPrim(
            prim_paths_expr=self.body_paths, name="robot_contact_pairs", reset_xform_properties=False,
            contact_filter_prim_paths_expr=[list(self.environment) for _ in self.body_paths],
            track_contact_forces=True,
            max_contact_count=4096)
        session.world.scene.add(self.contact_view)
        self.allowed_pairs = {(self.resolve_link(pair[0]), pair[1])
                              for pair in profile.get("allowed_contact_pairs", [])}
        self.target_root = bindings[object_id].get("prim_path", bindings[object_id].get("root_prim"))
        self.attachment_epoch = 0
        self.attached = False
        self.stage = "PREGRASP"

    def resolve_link(self, name):
        from pxr import Usd
        if name in self._link_cache:
            return self._link_cache[name]
        if name.startswith("/"):
            matches = [name] if self.session.stage.GetPrimAtPath(name).IsValid() else []
        else:
            root = self.session.stage.GetPrimAtPath(self.root)
            matches = [str(p.GetPath()) for p in Usd.PrimRange(root) if p.GetName() == name]
        if len(matches) != 1 or not matches[0].startswith(self.root + "/"):
            raise ValueError(f"Robot link must resolve exactly once under root: {name}")
        self._link_cache[name] = matches[0]
        return matches[0]

    def initialize(self):
        """One pre-episode initialization; subsequent motion uses controller targets."""
        if not self.session.initialized:
            raise RuntimeError("Initialize the world before the robot mapping")
        actual_bodies = list(self.contact_view.prim_paths)
        if set(actual_bodies) != set(self.body_paths):
            raise RuntimeError("Contact view resolved a different set of robot bodies")
        self.body_paths = actual_bodies
        actual = tuple(self.robot.dof_names)
        if len(actual) != len(set(actual)):
            raise ValueError("Articulation joint names are not unique")
        wanted = tuple(self.limits.joint_names) + (self.gripper.joint_name,)
        if any(name not in actual for name in wanted):
            raise ValueError("Robot asset is missing configured active/gripper joints")
        inactive = self.profile["inactive_joint_positions"]
        passive = self.profile.get("passive_joint_positions", {})
        if set(actual) != set(wanted) | set(inactive) | set(passive):
            raise ValueError("Specify ALL remaining articulation DOFs as held or passive/mimic")
        self.control_indices = np.array([i for i, name in enumerate(actual) if name not in passive], dtype=int)
        self.indices = np.array([actual.index(name) for name in self.limits.joint_names], dtype=int)
        self.gripper_index = actual.index(self.gripper.joint_name)
        positions = np.zeros(len(actual))
        positions[self.indices] = self.profile["initial_joint_positions"]
        positions[self.gripper_index] = self.gripper.angle(1.0)
        for name, value in inactive.items():
            positions[actual.index(name)] = value
        for name, value in passive.items():
            positions[actual.index(name)] = value
        properties = self.robot.dof_properties
        engine_limits = np.column_stack([properties["lower"], properties["upper"]])
        if np.any(positions < engine_limits[:, 0]) or np.any(positions > engine_limits[:, 1]):
            raise ValueError("Initial posture violates USD articulation limits")
        lower, upper, _, _ = self.limits.ordered(self.limits.joint_names)
        if np.any(lower < engine_limits[self.indices, 0] - 1e-5) or np.any(upper > engine_limits[self.indices, 1] + 1e-5):
            raise ValueError("Profile arm limits exceed USD limits")
        gripper_bounds = engine_limits[self.gripper_index]
        if self.gripper.lower_limit_rad < gripper_bounds[0] - 1e-5 or self.gripper.upper_limit_rad > gripper_bounds[1] + 1e-5:
            raise ValueError("Profile gripper limits exceed USD limits")
        self.robot.set_joint_positions(positions)
        self.robot.set_joint_velocities(np.zeros_like(positions))
        self.hold_positions = positions
        self.hold()

    def hold(self):
        from isaacsim.core.utils.types import ArticulationAction
        self.robot.apply_action(ArticulationAction(joint_positions=self.hold_positions[self.control_indices].copy(),
                                                 joint_indices=self.control_indices))

    def apply(self, command):
        if tuple(command.joint_names) != tuple(self.limits.joint_names):
            raise ValueError("Unexpected active joint order")
        positions = self.hold_positions.copy()
        positions[self.indices] = command.joint_positions
        positions[self.gripper_index] = command.gripper_angle_rad
        self.hold_positions = positions
        self.stage = command.stage
        self.hold()

    def world_version(self):
        # Robot motion is not an environment edit; carried objects are represented
        # by attached geometry. Quantization avoids invalidating for sub-mm jitter.
        objects = {}
        for object_id, binding in self.bindings.items():
            if object_id == self.object_id and self.attached:
                continue
            path = binding.get("prim_path", binding.get("root_prim"))
            if path:
                objects[object_id] = np.round(self.session.pose_of(path), 3)
            body = binding.get("rigid_body_prim")
            if body and body != path:
                # E.g. drawer roots stay fixed while the sliding body's pose changes.
                objects[object_id + ":body"] = np.round(self.session.pose_of(body), 3)
        for key, pose in objects.items():
            # JSON distinguishes -0.0 from 0.0; their hashes must not turn
            # sub-resolution PhysX rotation jitter into an environment edit.
            pose[pose == 0.0] = 0.0
            objects[key] = pose.tolist()
        objects["attachment_epoch"] = self.attachment_epoch
        return hashlib.sha256(json.dumps(objects, sort_keys=True).encode()).hexdigest()

    def locked_positions(self):
        positions = np.asarray(self.robot.get_joint_positions())
        return {name: float(positions[index]) for index, name in enumerate(self.robot.dof_names)
                if name not in self.limits.joint_names}

    def observe(self, sensors=None):
        positions = np.asarray(self.robot.get_joint_positions())
        velocities = np.asarray(self.robot.get_joint_velocities())
        contact = np.asarray(self.contact_view.get_contact_force_matrix(dt=1 / self.session.physics_hz))
        if contact.shape != (len(self.body_paths), len(self.environment), 3):
            raise RuntimeError("Contact matrix shape/order must match configured body and filter paths")
        active = np.linalg.norm(contact, axis=-1) > .01
        target_indices = [index for index, path in enumerate(self.environment)
                          if self.filter_roots[path] == self.target_root]
        if not target_indices:
            raise RuntimeError("No physical contact filter resolved for grasp target")
        both_fingers = all(np.any(active[self.body_paths.index(finger), target_indices]) for finger in self.fingers)
        unexpected = False
        contacts = []
        for body, environment in zip(*np.nonzero(active)):
            pair = (self.body_paths[body], self.filter_roots[self.environment[environment]])
            finger_target = pair[0] in self.fingers and pair[1] == self.target_root
            allowed = pair in self.allowed_pairs or (finger_target and self.stage != "PREGRASP")
            contacts.append({"robot_link": pair[0], "environment_prim": pair[1],
                             "filter_prim": self.environment[environment],
                             "force_world_n": contact[body, environment].copy(),
                             "force_norm_n": float(np.linalg.norm(contact[body, environment])),
                             "allowed": bool(allowed), "stage": self.stage,
                             "measurement": "PhysX_pairwise_net_contact_force"})
            if not allowed:
                unexpected = True
        ratio = (positions[self.gripper_index] - self.gripper.closed_angle_rad) / (
            self.gripper.open_angle_rad - self.gripper.closed_angle_rad)
        # Contact can prevent full closure. Values outside calibration are clamped
        # for the normalized UI/action contract; physical joint limits remain active.
        return WorldObservation(
            self.session.step_index, self.session.simulation_time, tuple(self.limits.joint_names),
            positions[self.indices], velocities[self.indices], self.session.pose_of(self.ee),
            self.session.pose_of(self.bindings[self.object_id]["rigid_body_prim"]),
            self.session.pose_of(self.base), np.asarray(self.target.get_linear_velocities())[0],
            np.asarray(self.target.get_angular_velocities())[0], float(np.clip(ratio, 0, 1)),
            self.world_version(), bool(both_fingers), unexpected, sensors, wall_time=time.time(),
            scene_poses={item_id: self.session.pose_of(binding.get("rigid_body_prim") or binding["prim_path"])
                         for item_id, binding in self.bindings.items()},
            articulated_states=self._observe_drawers(), contacts=tuple(contacts))

    def _observe_drawers(self):
        """Read actual body transforms/velocity; never apply drives or desired q."""
        from pxr import Gf
        states = {}
        for item_id, (joint, view) in self._drawer_views.items():
            binding = self.bindings[item_id]
            frames = []
            for path, pos, rot in (
                (binding["fixed_body_prim"], joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                (binding["body_prim"], joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
            ):
                local = np.eye(4)
                local[:3, :3] = np.asarray(Gf.Matrix3d(Gf.Rotation(rot.Get()))).T
                local[:3, 3] = np.asarray(pos.Get())
                frames.append(self.session.pose_of(path) @ local)
            a, b = frames
            axis = a[:3, {"X": 0, "Y": 1, "Z": 2}[str(joint.GetAxisAttr().Get())]]
            states[item_id] = {"joint_prim": binding["joint_prim"], "joint_type": "prismatic",
                               "position_m": float(axis @ (b[:3, 3] - a[:3, 3])),
                               "velocity_m_s": float(axis @ np.asarray(view.get_linear_velocities())[0]),
                               "measurement": "actual_body_pose_and_PhysX_linear_velocity",
                               "fixed_body_prim": binding["fixed_body_prim"]}
        return states

    def close(self):
        """Release PhysX tensor views before SimulationApp unloads the engine."""
        self.contact_view = self.target = self.robot = None
        self._drawer_views.clear()
