"""Independent stable RNG streams and explicit apply/read-back contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Protocol

import numpy as np


FIELDS = {
    "light": {"intensity", "color_temperature", "color", "translation", "rotation_deg"},
    "material": {"base_color", "roughness", "metallic", "base_color_texture"},
    "mass": {"mass_kg"},
    "friction": {"static_friction", "dynamic_friction", "restitution"},
    "camera": {"translation", "rotation_deg", "focal_length_mm", "horizontal_aperture_mm", "vertical_aperture_mm"},
}
VECTORS = {"color", "base_color", "translation", "rotation_deg"}


def validate_texture_assets(registry):
    """Validate supplied local PNG/JPEG candidates, not their legal entitlement.

    license_source records the caller's declared permission/source; this function
    does not download assets or certify that the declaration grants any rights.
    """
    if not isinstance(registry, dict):
        raise ValueError("texture_assets must be a path-to-license/hash mapping")
    result = {}
    for filename, record in registry.items():
        if not isinstance(filename, str) or not Path(filename).is_absolute():
            raise ValueError("texture candidates require absolute local paths")
        path = Path(filename)
        if not path.is_file() or str(path.resolve()) != filename:
            raise ValueError(f"texture candidate is missing or not canonical: {filename}")
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or path.stat().st_size > 64*1024**2:
            raise ValueError("texture must be a local PNG/JPEG of at most 64 MiB")
        if not isinstance(record, dict) or set(record) != {"sha256", "license_source"}:
            raise ValueError("each texture requires sha256 and license_source")
        digest, license_source = record["sha256"], record["license_source"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("texture sha256 must be lowercase hexadecimal")
        if not isinstance(license_source, str) or not license_source.strip():
            raise ValueError("texture requires a nonempty declared license_source")
        actual = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError(f"texture hash mismatch: {filename}")
        from PIL import Image
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG"} or image.mode not in {"RGB", "RGBA", "L"}:
                raise ValueError("texture must be a standard RGB/RGBA/grayscale PNG or JPEG")
            if min(image.size) < 1 or max(image.size) > 8192:
                raise ValueError("texture resolution exceeds 8192-pixel bound")
            image.verify()
        result[filename] = {"sha256": digest, "license_source": license_source}
    return result


def values_equal(actual, requested):
    if isinstance(requested, str):
        return isinstance(actual, str) and actual == requested
    return not isinstance(actual, str) and bool(np.allclose(actual, requested, rtol=1e-5, atol=1e-6))


def _check_target(target):
    if not isinstance(target, str) or not re.fullmatch(r"/[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*", target):
        raise ValueError("target must be an absolute, non-root ASCII USD prim path")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _native(value):
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(_canonical(list(parts)).encode()).digest()[:8], "big")


def validate_values(kind, values):
    if kind not in FIELDS or not isinstance(values, dict) or not values or set(values) - FIELDS[kind]:
        raise ValueError(f"unknown/empty fields for component kind {kind!r}")
    for name, value in values.items():
        if name == "base_color_texture":
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError("base_color_texture must be an absolute local path")
            continue
        array = np.asarray(value)
        if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite numeric")
        expected_shape = (3,) if name in VECTORS else ()
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if name in {"base_color", "color", "roughness", "metallic", "restitution"} and np.any((array < 0) | (array > 1)):
            raise ValueError(f"{name} must be in [0, 1]")
        if name in {"mass_kg", "focal_length_mm", "horizontal_aperture_mm", "vertical_aperture_mm"} and np.any(array <= 0):
            raise ValueError(f"{name} must be strictly positive")
        if name in {"intensity", "static_friction", "dynamic_friction"} and np.any(array < 0):
            raise ValueError(f"{name} must be nonnegative")
        if name == "color_temperature" and not 1000 <= array <= 40000:
            raise ValueError("color_temperature must be in [1000, 40000] kelvin")
    if {"static_friction", "dynamic_friction"} <= set(values) and values["dynamic_friction"] > values["static_friction"]:
        raise ValueError("dynamic_friction must not exceed static_friction")


def _distribution(rng, descriptor):
    if not isinstance(descriptor, dict) or "distribution" not in descriptor:
        raise ValueError("each parameter requires an explicit distribution")
    name = descriptor["distribution"]
    allowed = {"constant": {"distribution", "value"}, "uniform": {"distribution", "low", "high"},
               "normal": {"distribution", "mean", "std", "clip"}, "choice": {"distribution", "values"}}
    if name not in allowed or set(descriptor) - allowed[name]:
        raise ValueError(f"unknown distribution or keys: {name}")
    if name == "constant":
        result = descriptor["value"]
    elif name == "choice":
        values = descriptor["values"]
        if not isinstance(values, list) or not values:
            raise ValueError("choice requires a nonempty list")
        result = deepcopy(values[int(rng.integers(len(values)))])
    elif name == "uniform":
        low, high = np.asarray(descriptor["low"], dtype=float), np.asarray(descriptor["high"], dtype=float)
        if low.shape != high.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
            raise ValueError("uniform bounds must be finite, ordered and same-shaped")
        result = rng.uniform(low, high)
    else:
        mean, std = np.asarray(descriptor["mean"], dtype=float), np.asarray(descriptor["std"], dtype=float)
        if mean.shape != std.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError("normal mean/std must be finite, same-shaped; std >= 0")
        result = rng.normal(mean, std)
        if "clip" in descriptor:
            bounds = descriptor["clip"]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError("normal clip must be [low, high]")
            low, high = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
            if low.shape != mean.shape or high.shape != mean.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
                raise ValueError("invalid normal clipping bounds")
            result = np.clip(result, low, high)
    return _native(result)


class RandomizationAdapter(Protocol):
    """Only the host thread calls these; values are in the schema's SI/local units."""
    def set_component(self, kind: str, target: str, values: dict) -> None: ...
    def read_component(self, kind: str, target: str, fields: list[str]) -> dict: ...


class DomainRandomizer:
    def sample(self, context: dict, spec: dict) -> dict:
        context = deepcopy(context)
        seed = context.get("global_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("context.global_seed must be a nonnegative integer")
        if not isinstance(context.get("scene_id"), str) or not context["scene_id"]:
            raise ValueError("context.scene_id required")
        if isinstance(context.get("episode_id"), bool) or not isinstance(context.get("episode_id"), (str, int)):
            raise ValueError("context.episode_id must be a string or integer")
        if spec.get("schema_version") != 1 or set(spec) - {"schema_version", "components", "texture_assets"}:
            raise ValueError("randomization schema_version must be 1 with components")
        textures = validate_texture_assets(spec.get("texture_assets", {}))
        components = spec.get("components")
        if not isinstance(components, dict) or not components:
            raise ValueError("nonempty components mapping required")
        result = {"schema_version": 1, "context": context, "seed_algorithm": "sha256-json-first64be/PCG64",
                  "sampling_version": 1, "numpy_version": np.__version__,
                  "spec_sha256": hashlib.sha256(_canonical(spec).encode()).hexdigest(), "components": {}}
        if textures:
            result["texture_assets"] = textures
        for component_id in sorted(components):
            entry = components[component_id]
            if not isinstance(component_id, str) or not component_id or not isinstance(entry, dict):
                raise ValueError("component ids and definitions must be nonempty")
            if set(entry) != {"kind", "target", "parameters"}:
                raise ValueError("component requires exactly kind, target, parameters")
            kind, target, parameters = entry["kind"], entry["target"], entry["parameters"]
            _check_target(target)
            if not parameters or not isinstance(parameters, dict):
                raise ValueError("component requires absolute prim target and parameter mapping")
            component_seed = stable_seed(seed, context["scene_id"], context["episode_id"], component_id)
            requested, distributions = {}, {}
            for field in sorted(parameters):
                if field == "base_color_texture":
                    descriptor = parameters[field]
                    if not isinstance(descriptor, dict) or descriptor.get("distribution") not in {"choice", "constant"}:
                        raise ValueError("texture supports only explicit choice/constant distributions")
                    candidates = descriptor.get("values") if descriptor["distribution"] == "choice" else [descriptor.get("value")]
                    if not isinstance(candidates, list) or not candidates or any(not isinstance(path, str) or path not in textures for path in candidates):
                        raise ValueError("every texture candidate needs verified texture_assets metadata")
                parameter_seed = stable_seed(component_seed, field)
                requested[field] = _distribution(np.random.Generator(np.random.PCG64(parameter_seed)), parameters[field])
                distributions[field] = {"seed": parameter_seed, "descriptor": deepcopy(parameters[field])}
            validate_values(kind, requested)
            result["components"][component_id] = {"kind": kind, "target": target, "seed": component_seed,
                                                     "requested": requested, "distributions": distributions}
        return result

    def apply(self, sample: dict, adapter: RandomizationAdapter) -> dict:
        if sample.get("schema_version") != 1 or not sample.get("components"):
            raise ValueError("invalid randomization sample")
        result = deepcopy(sample)
        registry = validate_texture_assets(result.get("texture_assets", {}))
        if registry:
            register = getattr(adapter, "register_texture_assets", None)
            if register is None:
                raise ValueError("adapter does not support licensed texture registration")
            register(registry)
        # Prevalidate every requested value before any writes. An optional adapter
        # preflight also catches absent prims/inertia before applying other entries.
        for entry in result["components"].values():
            _check_target(entry["target"])
            validate_values(entry["kind"], entry["requested"])
            texture = entry["requested"].get("base_color_texture")
            if texture is not None and texture not in registry:
                raise ValueError("selected texture lacks verified asset metadata")
            preflight = getattr(adapter, "validate_component", None)
            if preflight is not None:
                preflight(entry["kind"], entry["target"], entry["requested"])
        order = {"mass": 0, "friction": 1, "camera": 2, "material": 3, "light": 4}
        for component_id in sorted(result["components"], key=lambda key: (order[result["components"][key]["kind"]], key)):
            entry = result["components"][component_id]
            adapter.set_component(entry["kind"], entry["target"], deepcopy(entry["requested"]))
            actual = adapter.read_component(entry["kind"], entry["target"], list(entry["requested"]))
            if set(actual) != set(entry["requested"]):
                raise RuntimeError(f"incomplete read-back for {component_id}")
            validate_values(entry["kind"], actual)
            for field, requested in entry["requested"].items():
                if not values_equal(actual[field], requested):
                    raise RuntimeError(f"read-back mismatch for {component_id}.{field}")
            entry["actual"] = {field: _native(value) for field, value in actual.items()}
            describe = getattr(adapter, "component_metadata", None)
            if describe is not None:
                details = describe(entry["kind"], entry["target"])
                if details:
                    entry["application_details"] = details
        result["applied"] = True
        result["planning_world_invalidated"] = True
        result["calibration_invalidated"] = any(entry["kind"] == "camera" for entry in result["components"].values())
        result["application_phase"] = getattr(adapter, "application_phase", "runtime_adapter")
        return result
