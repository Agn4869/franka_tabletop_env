"""Isaac Sim 5.1 single-owner host; this module itself has no Isaac imports.

All USD/Isaac imports occur AFTER SimulationApp construction. Physics tests are
separate from usd-core authoring tests; an unavailable engine is an error.
"""

from pathlib import Path
from copy import deepcopy
import hashlib
import threading
import numpy as np

from ..config import DependencyUnavailable, positive_int

_ACTIVE_SESSION = None


class IsaacSession:
    def __init__(self, scene_file, *, headless=True, physics_hz=120, control_hz=30, defer_app_shutdown=False):
        self.scene_file = Path(scene_file).resolve(strict=True)
        self.physics_hz = positive_int(physics_hz, "physics_hz", 10000)
        self.control_hz = positive_int(control_hz, "control_hz", 10000)
        if physics_hz % control_hz:
            raise ValueError("physics_hz must be divisible by control_hz")
        self.substeps = physics_hz // control_hz
        self.headless = headless
        self.defer_app_shutdown = defer_app_shutdown
        self.app = self.world = self.stage = None
        self.initialized = False
        self.step_index = 0
        self.owner = threading.get_ident()
        self._closed = False

    def _check_thread(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Only the host thread may access the simulation")
        if self._closed:
            raise RuntimeError("Session has closed")

    def __enter__(self):
        global _ACTIVE_SESSION
        self._check_thread()
        if _ACTIVE_SESSION is not None:
            raise RuntimeError("Only one active SimulationApp is supported per process")
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise DependencyUnavailable(
                "Isaac Sim 5.1 is not installed in this interpreter. Run with its python.sh; "
                "usd-core alone does not simulate physics or render RGB-D."
            ) from exc
        try:
            _ACTIVE_SESSION = self
            # CLI owns the process-level shutdown, after manifests and stdout.
            # In-process callers retain graceful shutdown (e.g. notebooks).
            self.app = SimulationApp({"headless": self.headless, "fast_shutdown": self.defer_app_shutdown})
            from isaacsim.core.utils.stage import open_stage
            from isaacsim.core.api import World
            import omni.usd
            from pxr import Usd

            if not open_stage(str(self.scene_file)):
                raise RuntimeError(f"Could not open stage: {self.scene_file}")
            self.stage = omni.usd.get_context().get_stage()
            # Runtime changes never modify the user's source USD layers.
            self.stage.SetEditTarget(Usd.EditTarget(self.stage.GetSessionLayer()))
            self.world = World(stage_units_in_meters=1.0,
                               physics_dt=1.0 / self.physics_hz,
                               rendering_dt=1.0 / self.control_hz)
            self.world.get_physics_context().set_physx_update_transformations_settings(
                update_to_usd=True, update_velocities_to_usd=True)
            return self
        except BaseException:
            self.close()
            raise

    def initialize(self):
        self._check_thread()
        if self.initialized:
            raise RuntimeError("Simulation already initialized")
        self.world.reset()
        self.initialized = True
        self.step_index = 0

    def step(self, steps=1, before_step=None):
        self._check_thread()
        if not self.initialized:
            raise RuntimeError("Initialize before stepping")
        positive_int(steps, "steps")
        # Rendering capture pauses the timeline. Restart only at an explicit physics step.
        self.world.play()
        for _ in range(steps):
            if not self.app.is_running():
                raise RuntimeError("Simulation application closed during run")
            if before_step is not None:
                before_step()
            self.world.step(render=False)
            self.step_index += 1

    @property
    def simulation_time(self):
        return self.step_index / self.physics_hz

    def pose_of(self, prim_path):
        self._check_thread()
        from pxr import Usd, UsdGeom
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Unknown prim: {prim_path}")
        # Gf matrices use row-vector convention, our contracts use columns.
        return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()), dtype=float).T

    def close(self):
        global _ACTIVE_SESSION
        if self._closed:
            return
        self._check_thread()
        try:
            if self.world is not None:
                self.world.stop()
                self.world.scene.clear(registry_only=True)
                self.world.clear_instance()
                self.world = self.stage = None
                # Engine-owned view callbacks must die before Kit unloads their
                # native libraries, not in interpreter garbage collection.
                import gc
                gc.collect()
        finally:
            try:
                if self.app is not None and not self.defer_app_shutdown:
                    self.app.close()
            finally:
                if _ACTIVE_SESSION is self and not self.defer_app_shutdown:
                    _ACTIVE_SESSION = None
                self._closed = True

    def __exit__(self, *_):
        self.close()


def shutdown_simulation_app(exit_code):
    """Finish the CLI-owned Kit process after its result has been committed."""
    global _ACTIVE_SESSION
    session = _ACTIVE_SESSION
    if session is None or not session.defer_app_shutdown:
        return
    session.close()
    if session.app is None:
        _ACTIVE_SESSION = None
        return
    # Use Kit's supported fast shutdown to avoid omni.ui static Python
    # callbacks acquiring the GIL after interpreter finalization.
    session.app.app.post_quit(exit_code)
    session.app.close()
    _ACTIVE_SESSION = None


def read_contact_summary(view, physics_dt):
    """Read actual net contact force; net zero does NOT prove absence of contact.

    Isaac 5.1 RigidPrim.get_net_contact_forces requires track_contact_forces=True.
    Passing the physical dt converts the backend impulses to newtons.
    https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.prims/docs/index.html
    """
    summary = {"available": False, "source": "RigidPrim.get_net_contact_forces",
               "physics_dt_s": float(physics_dt), "contact_count": None,
               "max_penetration_m": None, "scope": "net_force_only_no_pair_or_penetration_information"}
    try:
        values = view.get_net_contact_forces(dt=physics_dt)
        if values is None:
            raise RuntimeError("Contact force buffer is unavailable")
        values = np.asarray(values, dtype=float)
        if values.shape != (1, 3) or not np.isfinite(values).all():
            raise RuntimeError("Invalid net contact force buffer")
        summary.update(available=True, net_force_world_n=values[0].tolist(),
                       net_force_norm_n=float(np.linalg.norm(values[0])))
    except Exception as error:
        summary["unavailable_reason"] = f"{type(error).__name__}: {error}"
    return summary


class IsaacObjectStateSampler:
    """Read actual dynamic rigid-body state; register before World.reset()."""
    def __init__(self, session, bindings, *, object_ids=None):
        session._check_thread()
        if session.initialized:
            raise RuntimeError("Register state views before session initialization")
        from isaacsim.core.prims import RigidPrim
        from pxr import UsdPhysics
        self.session, self.views, self.paths = session, {}, {}
        selected = set(object_ids) if object_ids is not None else None
        if selected is not None and selected - set(bindings):
            raise ValueError("Requested state sampler object is absent from scene bindings")
        for object_id, binding in bindings.items():
            if selected is not None and object_id not in selected:
                continue
            if not binding.get("dynamic") or binding.get("joint_prim"):
                continue
            path = binding.get("rigid_body_prim") or binding.get("prim_path")
            prim = session.stage.GetPrimAtPath(path)
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"Monitored object is not a rigid body: {object_id}")
            api = UsdPhysics.RigidBodyAPI(prim)
            if api.GetKinematicEnabledAttr().Get() or not api.GetRigidBodyEnabledAttr().Get():
                raise ValueError(f"Monitored object must have active dynamic physics: {object_id}")
            name = "acceptance_" + hashlib.sha256(object_id.encode()).hexdigest()[:16]
            self.views[object_id] = session.world.scene.add(RigidPrim(
                prim_paths_expr=path, name=name, reset_xform_properties=False,
                track_contact_forces=True, prepare_contact_sensors=True, disable_stablization=False))
            self.paths[object_id] = path

    def read_snapshot(self):
        self.session._check_thread()
        if not self.session.initialized:
            raise RuntimeError("Initialize physics before reading dynamic state")
        states = {}
        for object_id, view in self.views.items():
            velocity = np.asarray(view.get_velocities(), dtype=float)
            if velocity.shape != (1, 6) or not np.isfinite(velocity).all():
                raise RuntimeError(f"Invalid measured velocity for {object_id}")
            states[object_id] = {"T_world_object": self.session.pose_of(self.paths[object_id]).tolist(),
                "linear_velocity_m_s": velocity[0, :3].tolist(), "angular_velocity_rad_s": velocity[0, 3:].tolist(),
                "contact_summary": read_contact_summary(view, 1/self.session.physics_hz),
                "state_source": "PhysX_RigidPrim_velocity_and_USD_synced_world_pose"}
        return {"physics_step": self.session.step_index, "simulation_time": self.session.simulation_time, "objects": states}

    def close(self):
        self.views.clear()


class IsaacDrawer:
    """A PhysX force/drive experiment; no transform teleportation during run."""

    def __init__(self, session, binding, mode="external_force"):
        session._check_thread()
        from isaacsim.core.prims import RigidPrim
        from pxr import UsdPhysics
        if session.initialized:
            raise RuntimeError("Bind the drawer before session initialization")
        if mode not in {"external_force", "joint_drive"}:
            raise ValueError("Unknown interaction mode")
        self.session, self.binding, self.mode = session, binding, mode
        self.force_axis_handle = np.asarray(binding.get("force_axis_handle", [1, 0, 0]), float)
        if self.force_axis_handle.shape != (3,) or not np.isfinite(self.force_axis_handle).all() or np.linalg.norm(self.force_axis_handle) < 1e-9:
            raise ValueError("Invalid handle force axis")
        self.force_axis_handle /= np.linalg.norm(self.force_axis_handle)
        self._last_command = None
        self._command_history, self._contact_interval = [], []
        self._last_contact_step = None
        self._interval_start_step = session.step_index
        self.view = session.world.scene.add(RigidPrim(
            prim_paths_expr=binding["body_prim"], name="drawer_force_body",
            reset_xform_properties=False, track_contact_forces=True,
            prepare_contact_sensors=True, disable_stablization=False))
        self.joint = UsdPhysics.PrismaticJoint.Get(session.stage, binding["joint_prim"])
        self.drive = UsdPhysics.DriveAPI.Apply(self.joint.GetPrim(), "linear")
        self.drive.CreateTypeAttr("force")
        self.drive.CreateStiffnessAttr(500.0 if mode == "joint_drive" else 0.0)
        self.drive.CreateDampingAttr(30.0 if mode == "joint_drive" else 0.0)
        self.drive.CreateMaxForceAttr(40.0)
        self.drive.CreateTargetPositionAttr(0.0)

    def _frames(self):
        from pxr import Gf
        # Use actual body transforms; default Isaac writes simulated transforms to USD.
        body0 = self.session.pose_of(self.binding["fixed_body_prim"])
        body1 = self.session.pose_of(self.binding["body_prim"])
        transforms = []
        for body, pos_attr, rot_attr in [
            (body0, self.joint.GetLocalPos0Attr(), self.joint.GetLocalRot0Attr()),
            (body1, self.joint.GetLocalPos1Attr(), self.joint.GetLocalRot1Attr()),
        ]:
            frame = np.eye(4)
            frame[:3, :3] = np.asarray(Gf.Matrix3d(Gf.Rotation(rot_attr.Get()))).T
            frame[:3, 3] = np.asarray(pos_attr.Get())
            transforms.append(body @ frame)
        return transforms

    def read_state(self, *, reset_interval=False):
        self.session._check_thread()
        if type(reset_interval) is not bool:
            raise ValueError("reset_interval must be a boolean")
        if not self.session.initialized:
            raise RuntimeError("Initialize physics before reading drawer state")
        a, b = self._frames()
        axis_index = {"X": 0, "Y": 1, "Z": 2}[str(self.joint.GetAxisAttr().Get())]
        axis = a[:3, axis_index]
        velocity = np.asarray(self.view.get_velocities(), dtype=float)
        if velocity.shape != (1, 6) or not np.isfinite(velocity).all():
            raise RuntimeError("Invalid measured drawer velocity")
        handle = self.session.pose_of(self.binding["handle_prim"])
        contacts = read_contact_summary(self.view, 1/self.session.physics_hz)
        if self._last_contact_step != self.session.step_index:
            if len(self._contact_interval) >= 4096:
                raise RuntimeError("Drawer interval contact buffer full; record and reset the interval")
            self._contact_interval.append({"physics_step": self.session.step_index, **contacts})
            self._last_contact_step = self.session.step_index
        available = [item["net_force_norm_n"] for item in self._contact_interval if item["available"]]
        result = {"step_index": self.session.step_index,
                "simulation_time": self.session.simulation_time,
                "position_m": float(axis @ (b[:3, 3] - a[:3, 3])),
                "velocity_m_s": float(axis @ velocity[0, :3]),
                "mode": self.mode, "handle_T_world": handle.tolist(),
                "body_linear_velocity_world_m_s": velocity[0, :3].tolist(),
                "body_angular_velocity_world_rad_s": velocity[0, 3:].tolist(),
                "contacts": contacts, "applied_command": deepcopy(self._last_command),
                "command_applications": deepcopy(self._command_history),
                "interval": {"start_step": self._interval_start_step, "end_step": self.session.step_index,
                    "contact_samples": len(self._contact_interval), "available_contact_samples": len(available),
                    "max_net_contact_force_norm_n": max(available) if available else None,
                    "mean_net_contact_force_norm_n": float(np.mean(available)) if available else None},
                "contact_samples": deepcopy(self._contact_interval)}
        if reset_interval:
            self._command_history.clear()
            self._contact_interval.clear()
            self._interval_start_step = self.session.step_index
        return result

    def _record_command(self, values):
        self._last_command = {"mode": self.mode, "applied_at_physics_step": self.session.step_index,
                              "simulation_time": self.session.simulation_time,
                              "target_body_prim": self.binding["body_prim"], "handle_prim": self.binding["handle_prim"],
                              "quantity_semantics": "submitted_control_target_not_measured_contact_force", **values}
        self._command_history.append(deepcopy(self._last_command))

    def apply_command(self, command):
        self.session._check_thread()
        if command.get("mode") != self.mode:
            raise ValueError("Schedule and drawer modes differ")
        if self.mode == "joint_drive":
            self.set_drive_target(command["position_m"])
        elif "force_handle_n" in command:
            self.apply_handle_force(command["force_handle_n"])
        else:
            direction = np.asarray(command.get("direction_handle", self.force_axis_handle), float)
            if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-9:
                raise ValueError("Invalid schedule force direction")
            self.apply_handle_force(direction/np.linalg.norm(direction) * float(command["force_n"]))

    def apply_handle_force(self, force_n):
        self.session._check_thread()
        if self.mode != "external_force":
            raise RuntimeError("External force is disabled in drive mode")
        if not self.session.initialized:
            raise RuntimeError("Initialize physics before applying force")
        if len(self._command_history) >= 4096:
            raise RuntimeError("Drawer interval command buffer full; record and reset the interval")
        local_force = np.asarray(force_n, float)
        if local_force.shape == ():
            local_force = self.force_axis_handle * local_force
        if local_force.shape != (3,) or not np.isfinite(local_force).all() or np.linalg.norm(local_force) > 40.0 + 1e-9:
            raise ValueError("Demo handle force must be finite with norm at most 40 N")
        handle = self.session.pose_of(self.binding["handle_prim"])
        force = handle[:3, :3] @ local_force
        self.view.apply_forces_and_torques_at_pos(
            forces=np.asarray([force], dtype=np.float32),
            positions=np.asarray([handle[:3, 3]], dtype=np.float32), is_global=True)
        self._record_command({"force_handle_n": local_force.tolist(), "force_world_n": force.tolist(),
                              "force_norm_n": float(np.linalg.norm(force)), "application_point_world_m": handle[:3, 3].tolist(),
                              "application_method": "RigidPrim.apply_forces_and_torques_at_pos", "force_frame": "world"})

    def set_drive_target(self, position_m):
        self.session._check_thread()
        if self.mode != "joint_drive":
            raise RuntimeError("Position drive is disabled in force mode")
        if not self.session.initialized:
            raise RuntimeError("Initialize physics before submitting drive targets")
        if len(self._command_history) >= 4096:
            raise RuntimeError("Drawer interval command buffer full; record and reset the interval")
        position_m = float(position_m)
        lo, hi = float(self.joint.GetLowerLimitAttr().Get()), float(self.joint.GetUpperLimitAttr().Get())
        if not np.isfinite(position_m) or not lo <= position_m <= hi:
            raise ValueError("Drawer target outside travel limits")
        self.drive.GetTargetPositionAttr().Set(position_m)
        actual = float(self.drive.GetTargetPositionAttr().Get())
        if not np.isclose(actual, position_m, rtol=1e-6, atol=1e-7):
            raise RuntimeError("Drawer drive target did not read back")
        self._record_command({"position_m": actual, "target_joint_prim": self.binding["joint_prim"],
                              "target_readback_source": "USD_drive_target_not_physical_position"})
