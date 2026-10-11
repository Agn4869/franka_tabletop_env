"""cuRobo MotionGen for the PhysX Franka, using authored tabletop colliders.

Collision geometry is expressed in the robot base frame. The moving block is an
obstacle up to the intentional contact phase, then attached to the planner for
carry; the block remains an unconstrained PhysX rigid body throughout.
"""
from __future__ import annotations

import numpy as np

from ..config import DependencyUnavailable


class MotionPlanningFailure(RuntimeError):
    """The requested collision-aware pose has no feasible MotionGen plan."""


class CuroboMotionGenBackend:
    name = "curobo_motiongen"

    def __init__(self, session, robot, *, pick_path, control_dt, timeout_s):
        try:
            import torch
            from curobo.geom.types import Cuboid, WorldConfig
            from curobo.types.base import TensorDeviceType
            from curobo.types.math import Pose
            from curobo.types.state import JointState
            from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
        except ImportError as error:
            raise DependencyUnavailable("Franka MotionGen needs cuRobo 0.7.6 and CUDA in Isaac Python") from error
        if not torch.cuda.is_available():
            raise DependencyUnavailable("Franka MotionGen requires a working CUDA device")
        if not np.isfinite(control_dt) or control_dt <= 0 or not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("MotionGen control interval and timeout must be positive")
        self.session, self.robot, self.pick_path = session, robot, pick_path
        self.tensor = TensorDeviceType(device=torch.device("cuda:0"))
        self.Pose, self.JointState, self.PlanConfig = Pose, JointState, MotionGenPlanConfig
        self.Cuboid, self.WorldConfig = Cuboid, WorldConfig
        self.dt, self.timeout_s = float(control_dt), float(timeout_s)
        self.joint_names = tuple(f"panda_joint{i}" for i in range(1, 8))
        actual_names = tuple(robot.dof_names)
        if tuple(actual_names[:7]) != self.joint_names:
            raise ValueError(f"Franka articulation DOFs do not match franka.yml: {actual_names}")
        self.collider_paths = self._collision_paths()
        self.pick_geometry = pick_path + "/Geometry"
        if self.pick_geometry not in self.collider_paths:
            raise ValueError("Pick object's collision geometry is not present in USD")
        world = self._world(include_pick=True)
        self.motion = MotionGen(MotionGenConfig.load_from_robot_config(
            "franka.yml", world, self.tensor, interpolation_dt=self.dt,
            trajopt_dt=0.1, js_trajopt_dt=0.1, optimize_dt=False,
            maximum_trajectory_dt=0.1,
            collision_cache={"obb": max(32, len(self.collider_paths) + 8)}))
        self.motion.warmup()
        self.attached = False
        self.plans = []

    def _collision_paths(self):
        from pxr import UsdPhysics

        paths = []
        for prim in self.session.stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith("/World/Franka/") or path == "/World/Franka":
                continue  # Robot collision spheres come from the official franka.yml.
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = UsdPhysics.CollisionAPI(prim)
                if bool(api.GetCollisionEnabledAttr().Get()):
                    paths.append(path)
        if not paths or "/World/Table/Top" not in paths:
            raise ValueError("No enabled physical tabletop collider in cuRobo world")
        return tuple(paths)

    def _world(self, *, include_pick):
        from pxr import UsdGeom
        from scipy.spatial.transform import Rotation

        base_inverse = np.linalg.inv(self.session.pose_of("/World/Franka"))
        colliders = []
        for path in self.collider_paths:
            if not include_pick and path == self.pick_geometry:
                continue
            cube = UsdGeom.Cube(self.session.stage.GetPrimAtPath(path))
            if not cube:
                raise ValueError(f"Non-cuboid collider needs explicit scene conversion: {path}")
            size = cube.GetSizeAttr().Get()
            if size is None or not np.isfinite(float(size)) or size <= 0:
                raise ValueError(f"Collider has no valid cube size: {path}")
            transform = base_inverse @ self.session.pose_of(path)
            matrix = transform[:3, :3]
            axis_sizes = np.linalg.norm(matrix, axis=0)
            if np.any(axis_sizes <= 0) or not np.isfinite(axis_sizes).all():
                raise ValueError(f"Invalid collider scale: {path}")
            rotation = matrix / axis_sizes
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < 0:
                raise ValueError(f"Unsupported sheared collider: {path}")
            center = transform[:3, 3]
            quat_xyzw = Rotation.from_matrix(rotation).as_quat()
            colliders.append(self.Cuboid(name=path, pose=[*center, quat_xyzw[3], *quat_xyzw[:3]],
                                         dims=(float(size) * axis_sizes).tolist()))
        if not colliders:
            raise ValueError("cuRobo refuses an empty scene collision world")
        return self.WorldConfig(cuboid=colliders)

    def sync_world(self, *, include_pick):
        if self.attached and include_pick:
            raise ValueError("Cannot duplicate the attached object in the collision world")
        self.motion.update_world(self._world(include_pick=include_pick))

    def _state(self, joint_positions):
        positions = np.asarray(joint_positions, dtype=float)
        if positions.shape[0] < 7 or not np.isfinite(positions[:7]).all():
            raise ValueError("Invalid observed Franka arm joint positions")
        return self.JointState.from_position(
            position=self.tensor.to_device(positions[:7][None, :]), joint_names=list(self.joint_names))

    def plan(self, *, joint_positions, goal_xyz, goal_quaternion_wxyz, phase, include_pick):
        from curobo.rollout.cost.pose_cost import PoseCostMetric
        from ..manipulation.trajectory import JointTrajectory
        if not self.attached:
            self.sync_world(include_pick=include_pick)
        position = np.asarray(goal_xyz, dtype=float)
        quaternion = np.asarray(goal_quaternion_wxyz, dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise ValueError("MotionGen goal must be a finite hand pose")
        base_inverse = np.linalg.inv(self.session.pose_of("/World/Franka"))
        from scipy.spatial.transform import Rotation
        world_rotation = Rotation.from_quat([*quaternion[1:], quaternion[0]]).as_matrix()
        goal_rotation = base_inverse[:3, :3] @ world_rotation
        q = Rotation.from_matrix(goal_rotation).as_quat()
        base_position = base_inverse[:3, :3] @ position + base_inverse[:3, 3]
        goal = self.Pose(position=self.tensor.to_device(base_position[None, :]),
                         quaternion=self.tensor.to_device(np.array([[q[3], *q[:3]]])))
        approach_metric = (PoseCostMetric.create_grasp_approach_metric(
            offset_position=.1, linear_axis=2, tstep_fraction=.1,
            project_to_goal_frame=False, tensor_args=self.tensor)
            if phase == "descend" else None)
        result = self.motion.plan_single(
            self._state(joint_positions), goal,
            self.PlanConfig(max_attempts=4, timeout=self.timeout_s,
                            enable_graph=False, enable_graph_attempt=2,
                            enable_finetune_trajopt=False, pose_cost_metric=approach_metric))
        status = str(result.status)
        if not bool(result.success.item()):
            raise MotionPlanningFailure(f"MotionGen {phase} plan failed: {status}")
        interpolation = result.get_interpolated_plan()
        positions = interpolation.position.detach().cpu().numpy()
        dt = float(result.interpolation_dt)
        if positions.ndim != 2 or positions.shape[1] != 7 or len(positions) < 2 or not np.isfinite(positions).all() or not np.isfinite(dt) or dt <= 0:
            raise RuntimeError(f"MotionGen {phase} returned an invalid trajectory")
        max_lateral_deviation = None
        if phase == "descend":
            # Removing the pick block from collision checking is safe only
            # within a narrow straight pinch corridor. Reject detours before
            # allowing a finger to sweep into the physical block.
            fk = self.motion.kinematics.get_state(self.tensor.to_device(positions))
            ee = fk.ee_position.detach().cpu().numpy()
            near_object = ee[:, 2] <= base_position[2] + .105
            if not near_object.any():
                raise MotionPlanningFailure("MotionGen descend trajectory never enters the pinch corridor")
            max_lateral_deviation = float(np.max(
                np.linalg.norm(ee[near_object, :2] - base_position[:2], axis=1)))
            if max_lateral_deviation > .005:
                self.plans.append({"phase": phase, "status": "pinch_corridor_violation",
                                   "maximum_lateral_deviation_m": max_lateral_deviation,
                                   "waypoints": len(positions)})
                raise MotionPlanningFailure(
                    f"MotionGen descend plan failed: pinch corridor drift {max_lateral_deviation:.4f} m")
        trajectory = JointTrajectory(tuple(interpolation.joint_names), np.arange(len(positions)) * dt,
                                     positions, phase).reordered(self.joint_names)
        self.plans.append({"phase": phase, "status": status, "waypoints": len(positions),
                           "interpolation_dt_s": dt, "duration_s": trajectory.duration,
                           "goal_world_xyz_m": position.tolist(),
                           "collision_geometry_count": len(self.collider_paths) - int(not include_pick),
                           "cartesian_approach_metric": phase == "descend",
                           "maximum_lateral_deviation_m": max_lateral_deviation,
                           "pick_object_attached_to_planner": self.attached})
        return trajectory

    def attach(self, joint_positions):
        if self.attached:
            raise ValueError("Pick object is already attached in planning world")
        self.sync_world(include_pick=True)
        attached = self.motion.attach_objects_to_robot(
            self._state(joint_positions), [self.pick_geometry], link_name="attached_object",
            remove_obstacles_from_world_config=True)
        if not attached:
            raise RuntimeError("cuRobo could not attach the measured pick collider to Franka")
        self.attached = True

    def detach(self):
        if not self.attached:
            raise ValueError("No cuRobo planning attachment to detach")
        self.motion.detach_object_from_robot(link_name="attached_object")
        self.attached = False

    def cleanup(self):
        if getattr(self, "motion", None) is not None:
            if self.attached:
                self.detach()
            self.motion = None
