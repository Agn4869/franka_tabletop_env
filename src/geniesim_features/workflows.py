"""Offline scene/variant authoring workflows. These do NOT run Isaac or render."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .config import load_config, positive_int, write_json


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON mapping required: {path}")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


def build_scene(config_path, output_dir, assets_path=None, *, seed=None):
    from .scene_generation import AssetResolver, SceneBuilder

    config_path = Path(config_path).resolve(strict=True)
    spec = load_config(config_path)
    if seed is not None:
        spec["seed"] = seed
    # CLI overrides are explicit caller paths; nested manifest paths in YAML
    # are instead anchored to the YAML directory, not the process cwd.
    resolver = AssetResolver(Path(assets_path).resolve(strict=True)) if assets_path else None
    return SceneBuilder(resolver, base_dir=config_path.parent).build(spec, Path(output_dir))


def randomize_usd(scene_dir, config_path, output_dir, *, seed=42, samples=1):
    """Write replayable USD overrides with actual composed-attribute read-back.

    This is not StaticSampleWriter: no images, sensor frames or fake actions are
    produced. All variants reference the original scene via an absolute path.
    Do not move/delete the source bundle without updating the references.
    """
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
    from .domain_randomization import DomainRandomizer, UsdRandomizationAdapter
    from .domain_randomization.sampling import values_equal

    samples = positive_int(samples, "samples", 10000)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    source = Path(scene_dir).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("--scene must name a scene bundle directory")
    source_scene = source/"scene.usda"
    source_manifest_path = source/"scene_manifest.json"
    base = _json(source_manifest_path)
    base_bindings = _json(source/"bindings.json")
    if base.get("schema_version") != 1 or not isinstance(base_bindings.get("objects"), dict):
        raise ValueError("source bundle must have schema_version 1 and object bindings")
    # Check the committed bundle instead of silently randomizing modified files
    # against stale provenance. Source files are never edited by this workflow.
    for filename, expected in base.get("files", {}).items():
        if Path(filename).name != filename:
            raise ValueError("source file hash entries must be plain filenames")
        if _sha(source/filename) != expected:
            raise ValueError(f"source bundle digest mismatch: {filename}")
    source_names = set(base.get("files", {})) | {"scene.usda", "bindings.json", "scene_manifest.json"}
    source_hashes = {name: _sha(source/name) for name in sorted(source_names)}
    config = load_config(config_path)
    config_base = Path(config_path).resolve(strict=True).parent
    texture_assets = config.get("texture_assets", {})
    if texture_assets:
        def texture_path(value):
            path = Path(value).expanduser()
            return str((path if path.is_absolute() else config_base / path).resolve(strict=True))
        resolved_textures = {texture_path(path): details for path, details in texture_assets.items()}
        if len(resolved_textures) != len(texture_assets):
            raise ValueError("Texture registry contains duplicate canonical paths")
        config["texture_assets"] = resolved_textures
        for component in config.get("components", {}).values():
            descriptor = component.get("parameters", {}).get("base_color_texture")
            if descriptor:
                if descriptor.get("distribution") == "choice":
                    descriptor["values"] = [texture_path(value) for value in descriptor["values"]]
                elif descriptor.get("distribution") == "constant":
                    descriptor["value"] = texture_path(descriptor["value"])
    randomizer = DomainRandomizer()
    context = {"global_seed": seed, "scene_id": str(base.get("scene_id", source.name))}
    # Sample and preflight before claiming the output directory; no fake success
    # directory is created for a missing target or invalid distribution.
    stage = Usd.Stage.Open(str(source_scene), Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError(f"cannot open source USD: {source_scene}")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(source_scene)))
    if unresolved:
        raise ValueError(f"source scene has unresolved references/payloads: {list(unresolved)}")
    dependency_paths = {str(Path(layer.realPath).resolve()) for layer in layers if layer.realPath}
    dependency_paths.update(str(Path(path).resolve(strict=True)) for path in assets)
    dependency_hashes = {path: _sha(path) for path in sorted(dependency_paths)}
    for asset in base.get("assets", {}).values():
        for dependency in asset.get("dependencies") or []:
            if _sha(dependency["path"]) != dependency["sha256"]:
                raise ValueError(f"source asset digest mismatch: {dependency['path']}")
    original_adapter = UsdRandomizationAdapter(stage)
    first = randomizer.sample({**context, "episode_id": 0}, config)
    original_adapter.register_texture_assets(first.get("texture_assets", {}))
    for component in first["components"].values():
        if component["kind"] != "camera":
            original_adapter.validate_component(component["kind"], component["target"], component["requested"])
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing randomization destination: {output}")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for index in range(samples):
        variant = output/f"variant_{index:06d}"
        variant.mkdir(exist_ok=False)
        override_path = variant/"randomization.usda"
        override = Sdf.Layer.CreateNew(str(override_path))
        entry_path = variant/"scene.usda"
        entry = Usd.Stage.CreateNew(str(entry_path))
        entry.GetRootLayer().subLayerPaths = ["randomization.usda", str(source_scene)]
        UsdGeom.SetStageUpAxis(entry, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(entry, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(entry, 1.0)
        entry.SetDefaultPrim(entry.GetPrimAtPath("/World"))
        entry.SetEditTarget(Usd.EditTarget(override))
        requested = first if index == 0 else randomizer.sample({**context, "episode_id": index}, config)
        # Wrist cameras cannot be authored until the robot is loaded. Preserve
        # their exact sampled values, then apply/read them in the engine host
        # after sensor creation. Never label these pending values as applied.
        offline = {key: value for key, value in requested["components"].items() if value["kind"] != "camera"}
        pending = {key: value for key, value in requested["components"].items() if value["kind"] == "camera"}
        applied = (randomizer.apply({**requested, "components": offline}, UsdRandomizationAdapter(entry))
                   if offline else {**deepcopy(requested), "components": {}, "applied": False})
        if pending:
            applied.update(pending_camera_components=pending, applied=False,
                           camera_application_phase="pending_engine_sensor_creation")
        override.Save()
        entry.GetRootLayer().Save()
        reopened = Usd.Stage.Open(str(entry_path), Usd.Stage.LoadAll)
        readback = UsdRandomizationAdapter(reopened)
        for component in applied["components"].values():
            values = readback.read_component(component["kind"], component["target"], list(component["requested"]))
            if any(not values_equal(values[key], wanted) for key,wanted in component["requested"].items()):
                raise RuntimeError(f"persisted USD read-back mismatch: {component['target']}")
            component["persisted_actual"] = values
        applied.update(execution_mode="usd_only", rendered=False, physics_simulated=False,
                       persisted_readback_verified=True, source_scene_sha256=source_hashes["scene.usda"])
        write_json(variant/"randomization.json", applied)
        bindings = deepcopy(base_bindings)
        for binding in bindings["objects"].values():
            body_path = binding.get("rigid_body_prim")
            if body_path:
                body = reopened.GetPrimAtPath(body_path)
                mass = UsdPhysics.MassAPI(body).GetMassAttr().Get()
                if mass is not None:
                    binding["mass_kg"] = float(mass)
            prim = reopened.GetPrimAtPath(binding["prim_path"])
            for purpose, field in (("physics", "physics_material_prim"), ("", "visual_material_prim")):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(purpose)
                if material:
                    binding[field] = str(material.GetPath())
        write_json(variant/"bindings.json", bindings)
        manifest = deepcopy(base)
        manifest.update(scene_file=str(entry_path), bindings_file=str(variant/"bindings.json"),
                        status="authored", variant_id=variant.name, seed=seed, physics_validated=False,
                        bindings=bindings["objects"], source_bundle=str(source),
                        execution_mode="usd_only", rendered=False, physics_simulated=False,
                        randomization_file=str(variant/"randomization.json"),
                        randomization_spec_sha256=applied["spec_sha256"],
                        source_dependency_hashes={**dependency_hashes, **{path: details["sha256"] for path, details in first.get("texture_assets", {}).items()}},
                        validation={**base.get("validation", {}), "usd_composition": "passed", "randomization_readback": "partial_pending_camera" if pending else "passed", "physics_stability": "not_run", "rendering": "not_run"},
                        files={p.name: _sha(p) for p in (entry_path, override_path, variant/"bindings.json", variant/"randomization.json")})
        write_json(variant/"scene_manifest.json", manifest)
        results.append({"variant_id": variant.name, "scene_file": str(entry_path),
                        "manifest_file": str(variant/"scene_manifest.json"),
                        "scene_sha256": _sha(entry_path), "randomization_sha256": _sha(variant/"randomization.json")})
    if any(_sha(source/name) != digest for name,digest in source_hashes.items()):
        raise RuntimeError("source bundle changed during randomization")
    if any(_sha(path) != digest for path,digest in dependency_hashes.items()):
        raise RuntimeError("source USD dependency changed during randomization")
    summary = {"schema_version": 1, "status": "authored", "execution_mode": "usd_only",
               "rendered": False, "physics_simulated": False, "sample_kind": "USD variants, not rendered training samples",
               "seed": seed, "samples_requested": samples, "samples_authored": len(results),
               "source_bundle": str(source), "source_hashes": source_hashes,
               "source_dependency_hashes": dependency_hashes, "variants": results}
    write_json(output/"run_manifest.json", summary)
    return summary
