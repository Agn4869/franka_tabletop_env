"""cuRobo 0.7.6 adapter; absence/failure never falls back to straight-line motion.

This adapter changes planning geometry only. PhysX remains responsible for grasp
contact. Real G2 integration requires a calibrated single-arm cuRobo profile,
including the other arm's locked-joint collision geometry and attached spheres.
"""

import numpy as np
from pathlib import Path

from ..config import DependencyUnavailable, load_config
from ..manipulation import JointTrajectory


class CuroboPlanner:
    def __init__(self, session, robot, *, control_dt, timeout_s=30.):
        try:
            import torch
            from curobo.geom.sdf.world import CollisionCheckerType
            from curobo.geom.types import WorldConfig
            from curobo.types.base import TensorDeviceType
            from curobo.types.math import Pose
            from curobo.types.state import JointState
            from curobo.util.usd_helper import UsdHelper
            from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
        except ImportError as error:
            raise DependencyUnavailable("Install the pinned cuRobo 0.7.6/CUDA stack in the Isaac interpreter") from error
        if not torch.cuda.is_available():
            raise DependencyUnavailable("cuRobo adapter requires a working CUDA device")
        self.session, self.robot = session, robot
        self.dt, self.timeout_s = float(control_dt), float(timeout_s)
        self.Pose, self.JointState, self.PlanConfig = Pose, JointState, MotionGenPlanConfig
        self.tensor = TensorDeviceType()
        # Do not rewrite somebody else's planning profile. Joint locking and paths
        # must be authored explicitly for the selected licensed robot asset.
        config = load_config(robot.profile["curobo_config"])
        robot_config = config.get("robot_cfg", config)
        kinematics = robot_config["kinematics"]
        config_directory = Path(robot.profile["curobo_config"]).parent
        for field in ("urdf_path", "asset_root_path", "external_asset_path", "external_robot_configs_path", "collision_spheres"):
            value = kinematics.get(field)
            if isinstance(value, str) and value:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = config_directory / path
                kinematics[field] = str(path.resolve(strict=True))
        if kinematics.get("base_link") != robot.profile["base_link"].split("/")[-1]:
            raise ValueError("cuRobo base link differs from robot profile")
        locks = kinematics.get("lock_joints", {})
        expected_locks = robot.locked_positions()
        passive = set(robot.profile.get("passive_joint_positions", {}))
        planner_locks = set(expected_locks) - passive
        if set(locks) != planner_locks:
            raise ValueError("cuRobo lock_joints must cover all held non-mimic DOFs")
        self.lock_joint_names = tuple(locks)
        for name, position in robot.profile["inactive_joint_positions"].items():
            if not np.isclose(locks[name], position, atol=1e-5):
                raise ValueError(f"cuRobo locked posture differs for {name}")
        for name, position in robot.profile.get("passive_joint_positions", {}).items():
            if name in locks:
                raise ValueError(f"cuRobo lock_joints must omit passive mimic DOF {name}")
        self.robot_config = robot_config
        names = robot_config["kinematics"]["cspace"]["joint_names"]
        if set(names) != set(robot.limits.joint_names):
            raise ValueError("cuRobo active joints must match the single-arm profile")
        if robot_config["kinematics"].get("ee_link") != robot.profile["end_effector_link"].split("/")[-1]:
            raise ValueError("cuRobo end effector differs from calibrated robot profile")
        # G2's short approach move should use a bounded controller interval.
        # Time-optimal retiming can return DT_EXCEPTION at the default 0.15 s
        # cap even when the collision-free pose is reachable.
        motion_config = MotionGenConfig.load_from_robot_config(
            robot_config, WorldConfig(cuboid=[], mesh=[]), self.tensor,
            collision_checker_type=CollisionCheckerType.MESH,
            interpolation_dt=self.dt, trajopt_dt=0.1, js_trajopt_dt=0.1,
            optimize_dt=False, maximum_trajectory_dt=0.1,
            collision_cache={"obb": 128, "mesh": 128})
        self.motion = MotionGen(motion_config)
        self.motion.warmup(enable_graph=True, warmup_js_trajopt=False)
        self.usd = UsdHelper()
        self.usd.load_stage(session.stage)
        self.observation = None
        self.attached_names = []
        self.attachment_link = robot.profile.get("attachment_link", "attached_object")
        self.version = ""

    def update_observation(self, observation):
        self.observation = observation

    def _joint_state(self, names, positions):
        js = self.JointState.from_position(position=self.tensor.to_device(positions), joint_names=list(names))
        return js.get_ordered_joint_state(self.motion.kinematics.joint_names)

    def sync_world(self, observation):
        from pxr import UsdPhysics
        self.update_observation(observation)
        # Exclude robot rendering meshes: its entire articulated collision model
        # belongs to robot_cfg, including locked joints. No blanket obstacle disable.
        ignored = [self.robot.root, "/World/Lighting"]
        # Mimic followers are derived by cuRobo from their URDF driver; only
        # independent inactive joints and the gripper driver may be locked.
        # Updating kinematics can reset attached spheres; reattach below.
        self.motion.update_locked_joints(
            {name: position for name, position in self.robot.locked_positions().items()
             if name in self.lock_joint_names}, self.robot_config)
        collider_paths = [str(prim.GetPath()) for prim in self.session.stage.Traverse()
                          if prim.HasAPI(UsdPhysics.CollisionAPI)
                          and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
        if not collider_paths:
            raise RuntimeError("No collision geometry to synchronize; refusing an empty planning world")
        world = self.usd.get_obstacles_from_stage(
            only_paths=collider_paths, reference_prim_path=self.robot.base,
            ignore_paths=ignored).get_collision_check_world()
        self.motion.update_world(world)
        if self.robot.attached:
            self._attach_geometry()
        self.version = observation.world_version
        return self.version

    def plan(self, joint_names, joint_positions, T_B_E, world_version):
        from pxr import Gf
        if world_version != self.version:
            raise ValueError("Planning requested against an unsynchronized world")
        q = Gf.Matrix3d(*np.asarray(T_B_E[:3, :3]).T.reshape(-1)).ExtractRotation().GetQuat()
        quat = np.array([q.GetReal(), *q.GetImaginary()])
        goal = self.Pose(position=self.tensor.to_device(T_B_E[:3, 3]),
                         quaternion=self.tensor.to_device(quat))
        result = self.motion.plan_single(
            self._joint_state(joint_names, joint_positions).unsqueeze(0), goal,
            self.PlanConfig(enable_graph=False, enable_graph_attempt=2, max_attempts=4,
                            timeout=self.timeout_s, enable_finetune_trajopt=False))
        if not bool(result.success.item()):
            raise ValueError(f"cuRobo candidate not feasible: {result.status}")
        interpolated = result.get_interpolated_plan()
        positions = interpolated.position.detach().cpu().numpy()
        if positions.ndim != 2 or len(positions) < 2:
            raise RuntimeError("Unexpected interpolated trajectory shape")
        actual_dt = float(result.interpolation_dt)
        if not np.isfinite(actual_dt) or actual_dt <= 0:
            raise RuntimeError("cuRobo returned an invalid interpolation interval")
        return JointTrajectory(tuple(interpolated.joint_names), np.arange(len(positions)) * actual_dt,
                               positions, world_version).reordered(joint_names)

    def attach(self, object_id, T_E_O):
        if self.robot.attached or object_id != self.robot.object_id or self.observation is None:
            raise ValueError("Invalid planning attachment lifecycle")
        # Re-extract at actual contact pose; the requested transform is compared
        # against observation instead of teleporting the object to an annotation.
        actual = np.linalg.inv(self.observation.T_W_E) @ self.observation.T_W_O
        if not np.allclose(actual, T_E_O, atol=1e-5):
            raise ValueError("Attachment transform differs from measured contact pose")
        self.sync_world(self.observation)
        self._attach_geometry()
        self.robot.attached = True
        self.robot.attachment_epoch += 1

    def _attach_geometry(self):
        names = [item.name for item in self.motion.world_model.objects
                 if item.name == self.robot.target_root or item.name.startswith(self.robot.target_root + "/")]
        if not names:
            raise RuntimeError("Target collision geometry was not imported into cuRobo")
        state = self._joint_state(self.observation.joint_names, self.observation.joint_positions).unsqueeze(0)
        success = self.motion.attach_objects_to_robot(
            state, names, link_name=self.attachment_link, remove_obstacles_from_world_config=True)
        if not success:
            raise RuntimeError("No usable attachment spheres; configure attached_object in robot_cfg")
        self.attached_names = names

    def detach(self, object_id, T_W_O):
        if object_id != self.robot.object_id or not self.robot.attached:
            raise ValueError("No matching planning attachment")
        self.motion.detach_object_from_robot(link_name=self.attachment_link)
        self.attached_names = []
        self.robot.attached = False
        self.robot.attachment_epoch += 1
        # The next sync imports the measured released pose from USD, not T_W_O's
        # desired target. The executor still verifies settling and containment.

    def cleanup(self):
        import gc
        try:
            if self.robot.attached:
                self.motion.detach_object_from_robot(link_name=self.attachment_link)
                self.robot.attached = False
                self.robot.attachment_epoch += 1
        finally:
            self.attached_names = []
            # Destroy CUDA graphs and their tensor buffers while Kit's GPU
            # context still exists, not during interpreter teardown.
            self.motion = self.robot_config = self.usd = self.observation = None
            gc.collect()
