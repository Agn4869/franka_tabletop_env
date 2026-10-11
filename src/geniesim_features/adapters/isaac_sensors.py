"""Real Replicator capture at a frozen physics step (Isaac Sim 5.1).

Authoring/math helpers can be tested with usd-core. Those tests do not execute
Replicator, render an image, or validate an installed Isaac engine.
"""
from collections.abc import Mapping
from copy import deepcopy
import numpy as np

from ..config import safe_name


def look_at_usd(position, target):
    """Column rotation for a USD camera looking along local -Z."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    if position.shape != (3,) or target.shape != (3,) or not np.isfinite([position, target]).all():
        raise ValueError("Look-at position and target must be finite length-three vectors")
    forward = target - position
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError("Invalid look-at direction")
    forward /= norm
    up = np.array([0., 0., 1.])
    if abs(forward @ up) > .99:
        up = np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(right, forward), -forward])


def mm_to_usd_camera_units(value_mm, meters_per_unit=1.0):
    """USD lens/filmback attributes use tenths of one scene unit, not always mm."""
    value_mm, meters_per_unit = float(value_mm), float(meters_per_unit)
    if not np.isfinite([value_mm, meters_per_unit]).all() or min(value_mm, meters_per_unit) <= 0:
        raise ValueError("Positive finite optical dimensions and stage units required")
    value = value_mm * 0.001 / (0.1 * meters_per_unit)
    if not np.finfo(np.float32).tiny <= value <= np.finfo(np.float32).max:
        raise ValueError("Camera attribute is outside the supported float32 range")
    return value


def _resolution(resolution):
    resolution = tuple(resolution)
    if len(resolution) != 2 or any(type(v) is not int or not 0 < v <= 4096 for v in resolution):
        raise ValueError("Invalid camera resolution")
    return resolution


def _check_filmback(focal, horizontal, vertical, resolution):
    if not np.isfinite([focal, horizontal, vertical]).all() or min(focal, horizontal, vertical) <= 0:
        raise ValueError("Camera focal length and apertures must be positive and finite")
    width, height = _resolution(resolution)
    if not np.isclose(horizontal / vertical, width / height, rtol=1e-5, atol=1e-8):
        raise ValueError("Filmback aspect must match render resolution; arbitrary conform/crop policies are unsupported")


def author_camera(stage, item):
    """Validate then author one NEW pinhole camera; returns (camera, resolution).

    Position is parent-local meters. look_at_world_m is a world point and is only
    accepted for /World-mounted cameras. Existing prims are never overwritten.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom
    camera_id = safe_name(item["id"])
    parent = item.get("parent_prim", "/World")
    parent_path = Sdf.Path(parent)
    if not parent_path.IsAbsolutePath() or not parent_path.IsPrimPath():
        raise ValueError("Camera parent must be an absolute prim path")
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim or not parent_prim.IsActive():
        raise ValueError(f"Camera {camera_id} parent missing: {parent}. Load robot before wrist camera.")
    path = Sdf.Path(item.get("prim_path", parent.rstrip("/") + "/" + camera_id))
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == parent_path or not path.HasPrefix(parent_path):
        raise ValueError("Camera prim must be an absolute descendant of parent_prim")
    if stage.GetPrimAtPath(path):
        raise ValueError(f"Refusing to overwrite existing camera/prim: {path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isclose(meters_per_unit, 1.0, rtol=0, atol=1e-9):
        raise ValueError("The sensor host requires a meter-scale stage")
    resolution = _resolution(item.get("resolution", [640, 480]))
    focal = mm_to_usd_camera_units(item.get("focal_length_mm", 24), meters_per_unit)
    horizontal = mm_to_usd_camera_units(item.get("horizontal_aperture_mm", 36), meters_per_unit)
    vertical = mm_to_usd_camera_units(item.get("vertical_aperture_mm", 27), meters_per_unit)
    _check_filmback(focal, horizontal, vertical, resolution)
    near, far = float(item.get("near", .01)), float(item.get("far", 10))
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError("Invalid camera clipping range")
    position = np.asarray(item.get("position_m", [0, 0, 0]), float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Invalid camera position")
    if "look_at_world_m" in item:
        if parent != "/World" or "quaternion_wxyz" in item:
            raise ValueError("World look-at requires a world-mounted camera without a second quaternion")
        target_world = np.asarray(item["look_at_world_m"], float)
        if target_world.shape != (3,) or not np.isfinite(target_world).all():
            raise ValueError("Invalid world look-at target")
        parent_transform = np.asarray(UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        target_local = (np.linalg.inv(parent_transform) @ np.r_[target_world, 1])[:3]
        rotation = look_at_usd(position, target_local)
        quaternion = Gf.Matrix3d(*rotation.T.reshape(-1)).ExtractRotation().GetQuat()
    else:
        quat = np.asarray(item.get("quaternion_wxyz", [1, 0, 0, 0]), float)
        if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-9:
            raise ValueError("Invalid camera quaternion")
        quat = quat / np.linalg.norm(quat)
        quaternion = Gf.Quatd(float(quat[0]), Gf.Vec3d(*quat[1:]))
    try:
        cam = UsdGeom.Camera.Define(stage, path)
        xform = UsdGeom.Xformable(cam)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quaternion)
        cam.CreateProjectionAttr("perspective")
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(horizontal)
        cam.CreateVerticalApertureAttr(vertical)
        cam.CreateHorizontalApertureOffsetAttr(0.0)
        cam.CreateVerticalApertureOffsetAttr(0.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(near, far))
    except BaseException:
        stage.RemovePrim(path)
        raise
    return cam, resolution


def camera_intrinsics(cam, resolution):
    """Read actual optics; reject a post-randomization aspect mismatch."""
    resolution = _resolution(resolution)
    if str(cam.GetProjectionAttr().Get()) != "perspective":
        raise ValueError("Only perspective pinhole cameras are supported")
    offsets = [cam.GetHorizontalApertureOffsetAttr().Get(), cam.GetVerticalApertureOffsetAttr().Get()]
    if not np.isfinite(offsets).all() or not np.allclose(offsets, 0, rtol=0, atol=1e-9):
        raise ValueError("Off-center/cropped filmback requires a dedicated calibration adapter")
    focal = float(cam.GetFocalLengthAttr().Get())
    horizontal = float(cam.GetHorizontalApertureAttr().Get())
    vertical = float(cam.GetVerticalApertureAttr().Get())
    _check_filmback(focal, horizontal, vertical, resolution)
    width, height = resolution
    return np.array([[focal / horizontal * width, 0, width / 2],
                     [0, focal / vertical * height, height / 2], [0, 0, 1.]])


def map_instance_labels(raw_labels, bindings, visible_ids):
    """Keep raw IDs; reserved labels are not business objects.

    Supported Replicator contract: idToLabels maps integer/JSON IDs to strings.
    Unknown visible prim paths fail explicitly instead of inventing object IDs.
    """
    if not isinstance(raw_labels, Mapping) or not isinstance(bindings, Mapping):
        raise ValueError("Renderer labels and object bindings must be mappings")
    roots = {}
    for object_id, binding in bindings.items():
        if not isinstance(object_id, str) or not object_id or not isinstance(binding, Mapping):
            raise ValueError("Invalid object binding")
        root = binding.get("prim_path", binding.get("root_prim"))
        if not isinstance(root, str) or not root.startswith("/") or root.endswith("/"):
            raise ValueError(f"Invalid prim binding for {object_id}")
        if root in roots:
            raise ValueError(f"Ambiguous business IDs share prim {root}")
        roots[root] = object_id
    id_to_object, reserved, raw = {}, {}, {}
    for source_id, label in raw_labels.items():
        if isinstance(source_id, bool) or not isinstance(source_id, (str, int, np.integer)):
            raise ValueError("Renderer instance ID must be an integer")
        try:
            instance_id = int(source_id)
        except (TypeError, ValueError) as error:
            raise ValueError("Renderer instance ID must be an integer") from error
        if not 0 <= instance_id <= np.iinfo(np.uint32).max or instance_id in raw:
            raise ValueError("Renderer instance ID is invalid or duplicated after conversion")
        if not isinstance(label, str) or not label:
            raise ValueError("Unsupported renderer label format; expected a raw prim-path string")
        raw[instance_id] = label
        if label in {"BACKGROUND", "UNLABELLED"}:
            reserved[instance_id] = label
            continue
        matches = [root for root in roots if label == root or label.startswith(root + "/")]
        if matches:
            id_to_object[instance_id] = roots[max(matches, key=len)]
    labels = np.asarray(visible_ids)
    if labels.ndim != 2 or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Instance segmentation must be an integer HxW array")
    missing = set(map(int, np.unique(labels))) - set(id_to_object) - set(reserved)
    if missing:
        details = {i: raw.get(i, "<missing renderer label>") for i in sorted(missing)}
        raise ValueError(f"Unmapped visible renderer instances: {details}; add explicit scene/robot bindings")
    return id_to_object, reserved, raw


class IsaacSensorRig:
    def __init__(self, session, spec, bindings):
        session._check_thread()
        import omni.replicator.core as rep
        from isaacsim.core.utils.semantics import add_labels
        self.session, self.rep = session, rep
        self.cameras, self.products, self.annotators = {}, {}, {}
        self.bindings = deepcopy(bindings)
        self._closed = False
        if not isinstance(spec.get("cameras"), list) or not spec["cameras"]:
            raise ValueError("Expected nonempty cameras list")
        try:
            rep.orchestrator.set_capture_on_play(False)
            for object_id, binding in self.bindings.items():
                prim_path = binding.get("prim_path", binding.get("root_prim"))
                if prim_path:
                    prim = session.stage.GetPrimAtPath(prim_path)
                    if not prim.IsValid():
                        raise ValueError(f"Missing semantic object prim: {prim_path}")
                    add_labels(prim, labels=[object_id], instance_name="class")
            for item in spec["cameras"]:
                camera_id = safe_name(item["id"])
                if camera_id in self.cameras:
                    raise ValueError("Duplicate camera id")
                cam, resolution = author_camera(session.stage, item)
                self.cameras[camera_id] = (cam, resolution)
                # Register each resource immediately, before another call can fail.
                product = rep.create.render_product(str(cam.GetPath()), resolution)
                self.products[camera_id] = product
                self.annotators[camera_id] = {}
                for name in ("rgb", "distance_to_image_plane", "instance_segmentation"):
                    annotator = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
                    self.annotators[camera_id][name] = annotator
                    annotator.attach([product])
        except BaseException:
            self._release(suppress_errors=True)
            raise

    def capture(self, frame_index=0):
        from ..sensors import CameraCalibration, SensorFrame, SensorBundle
        self.session._check_thread()
        if self._closed:
            raise RuntimeError("Sensor rig is closed")
        if not self.session.initialized:
            raise RuntimeError("Capture requires initialized physics")
        try:
            self.session.world.pause()
            before_time = self.session.world.current_time
            step, sim_time = self.session.step_index, self.session.simulation_time
            self.rep.orchestrator.step(delta_time=0.0, rt_subframes=4, pause_timeline=True)
            if abs(self.session.world.current_time - before_time) > 1e-9:
                raise RuntimeError("Rendering unexpectedly advanced physics time")
            frames = {}
            for camera_id, (cam, (width, height)) in self.cameras.items():
                anns = self.annotators[camera_id]
                rgb = np.asarray(anns["rgb"].get_data()).copy()
                depth = np.asarray(anns["distance_to_image_plane"].get_data()).copy()
                segmentation = anns["instance_segmentation"].get_data()
                if not isinstance(segmentation, dict) or "data" not in segmentation:
                    raise RuntimeError("No instance segmentation result for capture")
                ids = np.asarray(segmentation["data"]).copy()
                id_to_object, reserved, raw = map_instance_labels(
                    segmentation.get("info", {}).get("idToLabels", {}), self.bindings, ids)
                K = camera_intrinsics(cam, (width, height))
                transform = self.session.pose_of(str(cam.GetPath())) @ np.diag([1., -1., -1., 1.])
                near, far = cam.GetClippingRangeAttr().Get()
                calibration = CameraCalibration(width, height, K, transform, float(near), float(far))
                frames[camera_id] = SensorFrame(camera_id, step, rgb[..., :3], depth, ids, calibration,
                                               id_to_object, reserved, raw)
            if self.session.step_index != step or self.session.simulation_time != sim_time or abs(self.session.world.current_time - before_time) > 1e-9:
                raise RuntimeError("Physics changed while assembling sensor bundle")
            return SensorBundle(step, sim_time, frames, frame_index)
        except BaseException:
            self._release(suppress_errors=True)
            raise

    def _release(self, *, suppress_errors=False):
        errors = []
        for camera_id, product in list(self.products.items()):
            for annotator in self.annotators.get(camera_id, {}).values():
                try:
                    annotator.detach([product.path])
                except Exception as error:
                    errors.append(error)
        # Detach every camera before destroying any product: HydraTexture
        # destruction resets shared SyntheticData graphs used by other cameras.
        for product in self.products.values():
            try:
                product.destroy()
            except Exception as error:
                errors.append(error)
        for cam, _ in self.cameras.values():
            try:
                self.session.stage.RemovePrim(cam.GetPath())
            except Exception as error:
                errors.append(error)
        self.products.clear()
        self.annotators.clear()
        self.cameras.clear()
        self._closed = True
        if errors and not suppress_errors:
            raise RuntimeError(f"Sensor resource cleanup had {len(errors)} failure(s)") from errors[0]

    def close(self):
        if self._closed:
            return
        self.session._check_thread()
        self._release()

    def __enter__(self):
        self.session._check_thread()
        if self._closed:
            raise RuntimeError("Sensor rig is closed")
        return self

    def __exit__(self, exception_type, *_):
        if exception_type is None:
            self.close()
        elif not self._closed:
            self.session._check_thread()
            self._release(suppress_errors=True)
