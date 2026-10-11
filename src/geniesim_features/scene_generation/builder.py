"""F101: deterministic layout and real composed USD output, never fake physics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .assets import AssetResolver, sha256_file
from .layout import sample_layout, validate_object_ids
from .usd_utils import cube, number, open_box, physics_material, rigid_body, semantics, transform, vector


def _write_json(path: Path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _visual_material(stage, object_id, color):
    path = f"/World/Looks/{object_id}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


class SceneBuilder:
    """Generate a fresh bundle without starting or advancing any simulator.

    Pass `base_dir` explicitly when spec/asset-manifest paths are relative. The
    caller that reads YAML should pass its directory, rather than relying on cwd.
    An existing output directory is always rejected (even when empty).
    """

    def __init__(self, asset_resolver: AssetResolver | None = None, *, base_dir: Path | None = None):
        self.asset_resolver = asset_resolver
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else None

    def build(self, spec: dict, output_dir: Path) -> dict:
        from geniesim_features.articulated_objects import author_drawer

        output = Path(output_dir).expanduser().absolute()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing scene destination: {output}")
        if not isinstance(spec, dict) or spec.get("schema_version", 1) != 1:
            raise ValueError("SceneSpec schema_version must be 1")
        world = spec.get("world", {})
        if world.get("up_axis", "Z") != "Z" or world.get("meters_per_unit", 1.0) != 1.0:
            raise ValueError("output world convention must be Z-up and meters_per_unit=1")
        # This scene builder never silently pretends to have loaded a G2 profile.
        if spec.get("robot"):
            raise ValueError("robot profiles must be loaded by the dedicated Isaac robot adapter, not SceneBuilder")
        objects = spec.get("objects", [])
        if not isinstance(objects, list) or not objects:
            raise ValueError("SceneSpec needs a non-empty objects list")
        validate_object_ids(objects)
        resolver = self.asset_resolver or AssetResolver(spec.get("assets_manifest"), base_dir=self.base_dir)
        resolved, asset_selection = {}, {}
        for obj in objects:
            try:
                candidates = obj.get("asset_candidates")
                if candidates is not None:
                    if "asset" in obj:
                        raise ValueError("choose asset OR asset_candidates, not both")
                    if not isinstance(candidates, list) or not candidates or any(not isinstance(alias, str) or not alias for alias in candidates) or len(set(candidates)) != len(candidates):
                        raise ValueError("asset_candidates must be a nonempty distinct list of manifest asset aliases")
                    # Validate the entire explicit whitelist, not only the item
                    # randomly selected on this seed. No implicit downloads.
                    allowed = [resolver.resolve(alias) for alias in candidates]
                    seed_material = json.dumps(["asset_selection_v1", spec.get("seed",42), obj["id"]], separators=(",", ":"))
                    selection_seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
                    rng = np.random.Generator(np.random.PCG64(selection_seed))
                    selected = allowed[int(rng.integers(len(allowed)))]
                    resolved[obj["id"]] = selected
                    asset_selection[obj["id"]] = {"candidates": candidates, "selected_asset_id": selected.asset_id,
                                                   "selection_seed": selection_seed, "seed_algorithm": "sha256-json-first64be/PCG64"}
                else:
                    resolved[obj["id"]] = resolver.resolve(obj["asset"])
                    asset_selection[obj["id"]] = {"selected_asset_id": resolved[obj["id"]].asset_id, "selection_mode": "explicit_asset"}
            except (KeyError, ValueError, OSError) as exc:
                raise ValueError(f"object {obj['id']}: {exc}") from exc
        layout = sample_layout(spec, resolved)
        table_spec = spec.get("table", {})
        table_size = vector(table_spec.get("size_m", [1.2, .8, .05]), 3, "table size", positive=True)
        top = number(table_spec.get("top_z_m", .75), "table top", positive=True)
        if top <= table_size[2]:
            raise ValueError("table top must be higher than slab thickness")
        gap = number(spec.get("layout", {}).get("support_gap_m", .002), "support gap", nonnegative=True)
        normalized = json.loads(json.dumps(spec, sort_keys=True, allow_nan=False))
        config_digest = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        # Exclusive mkdir owns this fresh destination. Any later error leaves
        # only a partial authoring artifact, never a successful manifest.
        output.mkdir(parents=True, exist_ok=False)
        layout_path = output / "layout.usda"
        stage = Usd.Stage.CreateNew(str(layout_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world_prim)
        physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics.CreateGravityMagnitudeAttr(9.81)
        material = physics_material(stage, "/World/PhysicsMaterial", **spec.get("physics_material", {}))
        ground = cube(stage, "/World/Ground", [max(3., table_size[0]+1), max(3., table_size[1]+1), .05], [0, 0, -.025], material=material)
        semantics(ground, "ground", "ground")
        table = UsdGeom.Xform.Define(stage, "/World/Table").GetPrim()
        semantics(table, "table", "table")
        cube(stage, "/World/Table/Top", table_size, [0, 0, top-table_size[2]/2], material=material, color=(.45,.30,.18))
        leg_height = top-table_size[2]
        leg_size = min(.055, min(table_size[:2])/5)
        for i, (sx,sy) in enumerate([(-1,-1), (-1,1), (1,-1), (1,1)]):
            cube(stage, f"/World/Table/Leg{i}", [leg_size,leg_size,leg_height],
                 [sx*(table_size[0]-leg_size)/2,sy*(table_size[1]-leg_size)/2,leg_height/2], material=material)
        dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
        dome.CreateIntensityAttr(300.0)
        key = UsdLux.DistantLight.Define(stage, "/World/Lighting/Key")
        key.CreateIntensityAttr(1200.0)
        key.AddRotateXYZOp().Set(Gf.Vec3f(-35, 25, 0))
        bindings = {}
        by_id = {obj["id"]: obj for obj in objects}
        for placement in layout:
            object_id = placement["object_id"]
            obj, asset = by_id[object_id], resolved[object_id]
            size = placement["size_m"]
            path = f"/World/Objects/{object_id}"
            dynamic = bool(obj.get("dynamic", asset.dynamic))
            mass = number(obj.get("mass_kg", asset.mass_kg), "object mass", positive=True)
            color = vector(obj.get("color", [.75,.22,.12] if dynamic else [.25,.45,.7]), 3, "color")
            if not all(0 <= c <= 1 for c in color):
                raise ValueError("color components must be in [0,1]")
            z = top+(gap if dynamic else 0)+size[2]/2
            binding = {"object_id": object_id, "prim_path": path, "rigid_body_prim": path if dynamic else None,
                       "asset_id": asset.asset_id, "semantic_class": obj.get("semantic_class", asset.semantic_class),
                       "asset_selection": asset_selection[object_id],
                       "support_prim": "/World/Table/Top", "size_m": size,
                       "dynamic": dynamic, "mass_kg": mass if dynamic else None,
                       "physics_material_prim": "/World/PhysicsMaterial", "grasp_annotations": obj.get("grasp_annotations", [])}
            if asset.primitive == "drawer":
                drawer_config = dict(obj.get("drawer", {}))
                drawer_config.update(id=object_id, size_m=size, position_m=placement["xy_m"]+[top], yaw_deg=placement["yaw_deg"])
                binding.update(author_drawer(stage, path, drawer_config))
                binding["rigid_body_prim"] = binding["body_prim"]
                binding["dynamic"] = True
                z = top
                root = stage.GetPrimAtPath(path)
            else:
                root = UsdGeom.Xform.Define(stage, path).GetPrim()
                transform(root, placement["xy_m"]+[z], placement["yaw_deg"])
                if asset.primitive == "box":
                    cube(stage, f"{path}/Geometry", size, material=material, color=color)
                    if dynamic:
                        rigid_body(root, mass, size)
                elif asset.primitive == "open_container":
                    thickness = number(obj.get("wall_thickness_m", .01), "wall thickness", positive=True)
                    binding["collision_prims"] = open_box(stage, path, size, thickness, material=material, color=color)
                    x,y,h = size
                    binding.update(wall_thickness_m=thickness, inner_size_m=[x-2*thickness,y-2*thickness,h-thickness],
                                   interior_bounds_local_m=[[-x/2+thickness,-y/2+thickness,-h/2+thickness], [x/2-thickness,y/2-thickness,h/2]])
                    if dynamic:
                        rigid_body(root, mass)
                else:
                    conversion = UsdGeom.Xform.Define(stage, f"{path}/AssetConversion")
                    conversion.AddTranslateOp().Set(Gf.Vec3d(*[-v for v in asset.source_center_m]))
                    if asset.source_up_axis == "Y":
                        conversion.AddRotateXOp().Set(90.0)
                    conversion.AddScaleOp().Set(Gf.Vec3d(asset.source_meters_per_unit))
                    source = stage.DefinePrim(f"{path}/AssetConversion/Source")
                    source.GetReferences().AddReference(asset.path, asset.prim_path)
                    if dynamic:
                        rigid_body(root, mass)
            semantics(root, object_id, str(binding["semantic_class"]))
            visual = _visual_material(stage, object_id, color)
            UsdShade.MaterialBindingAPI.Apply(root).Bind(visual)
            binding["visual_material_prim"] = str(visual.GetPath())
            binding["visual_shader_prim"] = f"{visual.GetPath()}/Shader"
            placement["position_m"] = placement["xy_m"]+[z]
            binding["initial_position_m"] = placement["position_m"] if asset.primitive != "drawer" else binding["initial_position_m"]
            binding["initial_root_position_m"] = placement["position_m"]
            binding["initial_yaw_deg"] = placement["yaw_deg"]
            bindings[object_id] = binding
        stage.GetRootLayer().Save()
        scene_path = output / "scene.usda"
        entry = Usd.Stage.CreateNew(str(scene_path))
        entry.GetRootLayer().subLayerPaths = ["layout.usda"]
        UsdGeom.SetStageUpAxis(entry, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(entry, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(entry, 1.0)
        entry.SetDefaultPrim(entry.GetPrimAtPath("/World"))
        entry.GetRootLayer().Save()
        # Reopen the actual entry stage and validate composed world positions.
        composed = Usd.Stage.Open(str(scene_path), Usd.Stage.LoadAll)
        if composed is None:
            raise RuntimeError("written USD cannot be reopened")
        cache = UsdGeom.XformCache()
        for placement in layout:
            prim = composed.GetPrimAtPath(bindings[placement["object_id"]]["prim_path"])
            actual = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            if not np.allclose(actual, placement["position_m"], rtol=0, atol=1e-9):
                raise RuntimeError(f"composed USD position mismatch: {placement['object_id']}")
        binding_path = output / "bindings.json"
        _write_json(binding_path, {"schema_version": 1, "objects": bindings})
        manifest = {
            "schema_version": 1, "scene_id": spec.get("scene_id", "tabletop"), "seed": spec.get("seed", 42),
            "status": "authored", "validation": {"usd_composition": "passed", "obb_layout": "passed", "physics_stability": "not_run", "robot_reachability": "not_run"},
            "physics_validated": False, "world": {"up_axis": "Z", "meters_per_unit": 1.0, "quaternion_order": "wxyz"},
            "scene_file": str(scene_path), "layout_file": str(layout_path), "bindings_file": str(binding_path),
            "config_sha256": config_digest, "resolved_config": normalized,
            "assets": {key: value.to_dict() for key,value in resolved.items()}, "asset_selection": asset_selection,
            "layout": layout, "bindings": bindings,
            "files": {p.name: sha256_file(p) for p in (scene_path,layout_path,binding_path)},
        }
        _write_json(output / "scene_manifest.json", manifest)
        # scene_manifest is the commit point, never a physics success claim.
        return manifest
