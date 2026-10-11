"""Explicit cross-view visibility acceptance, leaving optional occluded IDs null."""
import numpy as np
from ..config import positive_int
from .geometry import depth_valid_mask


def check_visibility(bundle, bindings, policy=None):
    policy = {} if policy is None else policy
    if not isinstance(policy, dict) or set(policy) - {"required_objects", "min_visible_pixels"}:
        raise ValueError("visibility requires required_objects/min_visible_pixels")
    defaults = [key for key, value in bindings.items() if value.get("dynamic") and not value.get("joint_prim")]
    required = policy.get("required_objects", defaults or list(bindings))
    if not isinstance(required, list) or not required or any(name not in bindings for name in required) or len(set(required)) != len(required):
        raise ValueError("visibility required_objects must be nonempty distinct bound object IDs")
    minimum = positive_int(policy.get("min_visible_pixels", 1), "min_visible_pixels", 4096 * 4096)
    counts = {name: {} for name in required}
    for camera_id, frame in bundle.frames.items():
        valid = depth_valid_mask(frame.depth, frame.calibration.near, frame.calibration.far)
        for name in required:
            ids = [index for index, object_id in frame.id_to_object.items() if object_id == name]
            counts[name][camera_id] = int(np.count_nonzero(valid & np.isin(frame.instance_ids, ids)))
    missing = [name for name in required if max(counts[name].values(), default=0) < minimum]
    if missing:
        raise RuntimeError(f"scene_rejected:required_objects_not_visible:{missing}; valid_pixels={counts}")
    return {"status": "accepted", "required_objects": required, "min_visible_pixels": minimum,
            "valid_visible_pixels_by_camera": counts, "criterion": "each_required_object_visible_in_at_least_one_camera"}
