"""Real USD authoring before physics initialization, NOT a runtime PhysX update.

Author into the host-selected edit target (normally a randomization override
layer). Mass/inertia scale together at fixed geometry. Poses are absolute local
prim poses, with XYZ Euler degrees; the camera host rereads final calibration.
"""
import hashlib
from pathlib import Path
import threading

import numpy as np

from .sampling import validate_values, validate_texture_assets


class UsdRandomizationAdapter:
    application_phase = "usd_pre_initialization"

    def __init__(self, stage, *, initialized=False, texture_assets=None):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
        self.Gf, self.Sdf, self.Usd = Gf, Sdf, Usd
        self.UsdGeom, self.UsdLux, self.UsdPhysics, self.UsdShade = UsdGeom, UsdLux, UsdPhysics, UsdShade
        self.stage = stage
        self.initialized = bool(initialized)
        self.owner_thread = threading.get_ident()
        self.texture_assets = {}
        self.register_texture_assets(texture_assets or {})
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise ValueError("randomization adapter requires a meter-scale stage")

    def register_texture_assets(self, texture_assets):
        self.texture_assets.update(validate_texture_assets(texture_assets))

    def mark_initialized(self):
        """Host calls before World.reset/play; future USD-only writes then fail."""
        self.initialized = True

    def _prim(self, target):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("USD randomization must run on its owning host thread")
        path = self.Sdf.Path(target)
        if not path.IsAbsolutePath() or not path.IsPrimPath() or path == self.Sdf.Path.absoluteRootPath:
            raise ValueError("target must be an absolute non-root prim path")
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive() or prim.IsInstanceProxy():
            raise ValueError(f"target prim missing, inactive or instance proxy: {target}")
        return prim

    def validate_component(self, kind, target, values):
        if self.initialized:
            raise RuntimeError("USD-only adapter cannot update initialized physics; use runtime setters or rebuild")
        validate_values(kind, values)
        prim = self._prim(target)
        if kind == "mass":
            if not prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
                raise ValueError("mass target must be a rigid body")
            api = self.UsdPhysics.MassAPI(prim)
            mass = api.GetMassAttr().Get() if api else None
            inertia = api.GetDiagonalInertiaAttr().Get() if api else None
            if mass is None or not np.isfinite(mass) or mass <= 0 or inertia is None:
                raise ValueError("mass randomization needs explicitly authored positive mass and diagonal inertia")
            inertia = np.asarray(inertia)
            if not np.isfinite(inertia).all() or np.any(inertia <= 0) or np.any(2 * inertia > inertia.sum() + 1e-8):
                raise ValueError("invalid diagonal inertia; infer/author it from geometry first")
        elif kind == "light" and not prim.HasAPI(self.UsdLux.LightAPI):
            raise ValueError("light target needs UsdLux.LightAPI")
        elif kind == "camera" and not prim.IsA(self.UsdGeom.Camera):
            raise ValueError("camera target must be UsdGeom.Camera")
        elif kind in {"material", "friction"} and not self.UsdGeom.Imageable(prim):
            raise ValueError("material target must be an imageable object prim")
        if kind == "material":
            current = self._visual_values(prim)
            self._check_material_subtree(prim)
            texture = values.get("base_color_texture", current.get("base_color_texture"))
            if texture:
                if texture not in self.texture_assets:
                    raise ValueError("texture requires registered sha256/license_source metadata")
                validate_texture_assets({texture: self.texture_assets[texture]})
                self._validate_uvs(prim)
        if kind == "friction":
            old = self._read_friction(prim)
            old.update(values)
            validate_values("friction", old)
        if {"translation", "rotation_deg"} & set(values):
            xform = self.UsdGeom.Xformable(prim)
            if not xform or any(op.GetAttr().GetNumTimeSamples() for op in xform.GetOrderedXformOps()):
                raise ValueError("local pose randomization requires a static xformable prim")
            if "rotation_deg" in values and any(abs(float(v)) >= 89 for v in values["rotation_deg"]):
                raise ValueError("conservative pose randomization requires Euler components in (-89,89) degrees")

    def _material(self, prim, purpose):
        material, _ = self.UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(purpose)
        return material

    def _read_friction(self, prim):
        material = self._material(prim, "physics")
        api = self.UsdPhysics.MaterialAPI(material.GetPrim()) if material else None
        defaults = {"static_friction": 0.5, "dynamic_friction": 0.5, "restitution": 0.0}
        getters = {"static_friction": "GetStaticFrictionAttr", "dynamic_friction": "GetDynamicFrictionAttr", "restitution": "GetRestitutionAttr"}
        for key, getter in getters.items():
            value = getattr(api, getter)().Get() if api else None
            if value is not None:
                defaults[key] = float(value)
        return defaults

    def _owned_material(self, prim, purpose):
        token = hashlib.sha256(f"{purpose}:{prim.GetPath()}".encode()).hexdigest()[:24]
        path = f"/World/RandomizationMaterials/{purpose}_{token}"
        existing = self.stage.GetPrimAtPath(path)
        if existing and existing.GetCustomDataByKey("randomizationTarget") != str(prim.GetPath()):
            raise ValueError(f"refusing to overwrite an unrelated material: {path}")
        material = self.UsdShade.Material.Define(self.stage, path)
        material.GetPrim().SetCustomDataByKey("randomizationTarget", str(prim.GetPath()))
        return material

    def _bind(self, prim, material, purpose):
        self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, self.UsdShade.Tokens.strongerThanDescendants, purpose)

    def _visual_values(self, prim):
        material = self._material(prim, self.UsdShade.Tokens.allPurpose)
        defaults = {"base_color": [0.5, 0.5, 0.5], "roughness": 0.5, "metallic": 0.0}
        if material:
            if any(attribute.GetName().startswith("outputs:") and attribute.GetName() != "outputs:surface"
                   and (attribute.HasAuthoredValueOpinion() or attribute.GetConnections())
                   for attribute in material.GetPrim().GetAttributes()):
                raise ValueError("additional material render contexts require a dedicated adapter")
            source = material.ComputeSurfaceSource()
            shader = source[0] if source else None
            if not shader or source[1] != "surface":
                raise ValueError("material requires a supported connected surface")
            if shader:
                if shader.GetIdAttr().Get() != "UsdPreviewSurface":
                    raise ValueError("visual randomization supports only a simple UsdPreviewSurface material")
                if any(input_.GetBaseName() not in {"diffuseColor", "roughness", "metallic"} for input_ in shader.GetInputs()):
                    raise ValueError("additional PreviewSurface inputs require a dedicated material adapter")
                if any(input_.GetAttr().GetNumTimeSamples() for input_ in shader.GetInputs()):
                    raise ValueError("animated shader inputs require a dedicated material adapter")
                for field, name in (("base_color", "diffuseColor"), ("roughness", "roughness"), ("metallic", "metallic")):
                    input_ = shader.GetInput(name)
                    if input_ and input_.HasConnectedSource():
                        if field != "base_color":
                            raise ValueError("texture/shader networks require a dedicated material adapter")
                        texture, scale = self._texture_value(input_)
                        defaults["base_color_texture"] = texture
                        defaults["base_color"] = scale
                        continue
                    value = input_.Get() if input_ else None
                    if value is not None:
                        defaults[field] = list(value) if field == "base_color" else float(value)
        return defaults

    def _texture_value(self, input_):
        source = input_.GetConnectedSource()
        texture = self.UsdShade.Shader(source[0].GetPrim()) if source else None
        if not texture or texture.GetIdAttr().Get() != "UsdUVTexture" or source[1] != "rgb":
            raise ValueError("only simple UsdUVTexture.rgb diffuseColor connections are supported")
        allowed = {"file", "st", "wrapS", "wrapT", "sourceColorSpace", "scale", "bias", "fallback"}
        for item in texture.GetInputs():
            if item.GetBaseName() not in allowed or (item.GetBaseName() != "st" and item.HasConnectedSource()):
                raise ValueError("complex texture graphs require a dedicated adapter")
            if item.GetAttr().GetNumTimeSamples():
                raise ValueError("animated texture inputs require a dedicated adapter")
        st = texture.GetInput("st")
        uv_source = st.GetConnectedSource() if st else None
        uv = self.UsdShade.Shader(uv_source[0].GetPrim()) if uv_source else None
        if not uv or uv.GetIdAttr().Get() != "UsdPrimvarReader_float2" or uv_source[1] != "result":
            raise ValueError("texture coordinates must come directly from a st primvar reader")
        varname = uv.GetInput("varname")
        if not varname or varname.HasConnectedSource() or varname.Get() != "st":
            raise ValueError("only a direct st primvar is supported")
        if any(item.GetBaseName() != "varname" for item in uv.GetInputs()):
            raise ValueError("custom primvar-reader inputs are not supported")
        for name, expected in (("bias", [0,0,0,0]), ("fallback", [0,0,0,1])):
            item = texture.GetInput(name)
            if item and item.Get() is not None and not np.allclose(item.Get(), expected):
                raise ValueError("nondefault texture bias/fallback would be lost")
        for name, expected in (("wrapS", "repeat"), ("wrapT", "repeat"), ("sourceColorSpace", "sRGB")):
            item = texture.GetInput(name)
            if not item or item.Get() != expected:
                raise ValueError("texture wrap/color-space needs a dedicated adapter")
        file = texture.GetInput("file")
        asset = file.Get() if file else None
        filename = str(asset.resolvedPath or asset.path) if asset else ""
        if not filename or not Path(filename).is_absolute() or not Path(filename).is_file():
            raise ValueError("texture source must resolve to an existing local file")
        scale_input = texture.GetInput("scale")
        scale = np.asarray(scale_input.Get() if scale_input and scale_input.Get() is not None else [1,1,1,1], float)
        if scale.shape != (4,) or not np.isfinite(scale).all() or scale[3] != 1 or np.any((scale[:3] < 0) | (scale[:3] > 1)):
            raise ValueError("texture tint must be finite RGB in [0,1] with alpha scale 1")
        return str(Path(filename).resolve()), scale[:3].tolist()

    def _geometries(self, prim):
        items = list(self.Usd.PrimRange(prim))
        if any(item.IsInstance() or item.IsInstanceProxy() for item in items):
            raise ValueError("texture authoring does not support instanced geometry")
        return [item for item in items if self.UsdGeom.Gprim(item)]

    def _check_material_subtree(self, prim):
        root_material = self._material(prim, self.UsdShade.Tokens.allPurpose)
        for item in self._geometries(prim):
            child_material = self._material(item, self.UsdShade.Tokens.allPurpose)
            if child_material and (not root_material or child_material.GetPath() != root_material.GetPath()):
                raise ValueError("heterogeneous child materials require explicit geometry-level targets")
            self._visual_values(item)

    def _validate_uvs(self, prim):
        geometries = self._geometries(prim)
        if not geometries:
            raise ValueError("texture target must contain supported surface geometry")
        for item in geometries:
            if item.IsA(self.UsdGeom.Cube):
                cube = self.UsdGeom.Cube(item)
                size = cube.GetSizeAttr().Get()
                if not np.isfinite(size) or size <= 0 or cube.GetSizeAttr().GetNumTimeSamples():
                    raise ValueError("texture UV generation requires a positive static Cube size")
                continue
            if not item.IsA(self.UsdGeom.Mesh):
                raise ValueError("texture requires an existing UV Mesh or a simple Cube")
            mesh = self.UsdGeom.Mesh(item)
            st = self.UsdGeom.PrimvarsAPI(item).GetPrimvar("st")
            if not st or st.GetTypeName() not in (self.Sdf.ValueTypeNames.TexCoord2fArray, self.Sdf.ValueTypeNames.Float2Array):
                raise ValueError("mesh texture requires existing float2 st coordinates")
            if st.GetAttr().GetNumTimeSamples():
                raise ValueError("animated UV data requires a dedicated adapter")
            expected = {"constant": 1, "uniform": len(mesh.GetFaceVertexCountsAttr().Get() or []),
                        "faceVarying": len(mesh.GetFaceVertexIndicesAttr().Get() or []),
                        "vertex": len(mesh.GetPointsAttr().Get() or []),
                        "varying": len(mesh.GetPointsAttr().Get() or [])}.get(str(st.GetInterpolation()))
            values = np.asarray(st.ComputeFlattened())
            if not expected or values.shape != (expected, 2) or not np.isfinite(values).all():
                raise ValueError("mesh st coordinates do not match its topology")

    def _author_box_uvs(self, prim):
        """Equivalent box mesh in the current layer; never modify source assets."""
        changes = []
        for item in self._geometries(prim):
            if not item.IsA(self.UsdGeom.Cube):
                continue
            half = float(self.UsdGeom.Cube(item).GetSizeAttr().Get()) / 2
            path = str(item.GetPath())
            mesh = self.UsdGeom.Mesh.Define(self.stage, path)
            points = [(-half,-half,-half), (half,-half,-half), (half,half,-half), (-half,half,-half),
                      (-half,-half,half), (half,-half,half), (half,half,half), (-half,half,half)]
            indices = [0,3,2,1, 4,5,6,7, 0,1,5,4, 1,2,6,5, 2,3,7,6, 3,0,4,7]
            mesh.CreatePointsAttr([self.Gf.Vec3f(*point) for point in points])
            mesh.CreateFaceVertexCountsAttr([4]*6)
            mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateSubdivisionSchemeAttr(self.UsdGeom.Tokens.none)
            mesh.CreateExtentAttr([self.Gf.Vec3f(-half), self.Gf.Vec3f(half)])
            uv = self.UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", self.Sdf.ValueTypeNames.TexCoord2fArray,
                                                          self.UsdGeom.Tokens.faceVarying)
            uv.Set([self.Gf.Vec2f(*value) for value in [(0,0),(1,0),(1,1),(0,1)]*6])
            if item.HasAPI(self.UsdPhysics.CollisionAPI):
                self.UsdPhysics.MeshCollisionAPI.Apply(item).CreateApproximationAttr("convexHull")
            item.SetCustomDataByKey("textureBoxConversion", {"source_schema": "Cube", "source_size": half*2,
                                                            "uv_mapping": "per_face_unit_square", "same_local_extent": True})
            changes.append(path)
        return changes

    def component_metadata(self, kind, target):
        """Additional provenance; records schema conversion, never render success."""
        if kind != "material":
            return {}
        prim = self._prim(target)
        values = self._visual_values(prim)
        texture = values.get("base_color_texture")
        if not texture:
            return {}
        conversions = []
        for item in self._geometries(prim):
            conversion = item.GetCustomDataByKey("textureBoxConversion")
            if conversion:
                conversions.append({"prim_path": str(item.GetPath()), **dict(conversion),
                                    "current_schema": item.GetTypeName()})
        return {"texture_path": texture, "texture_asset": dict(self.texture_assets[texture]),
                "base_color_semantics": "RGB texture tint via UsdUVTexture.scale",
                "geometry_conversions": conversions, "rendering_verified": False}

    def _pose(self, prim, values):
        Gf = self.Gf
        xform = self.UsdGeom.Xformable(prim)
        transform = Gf.Transform(xform.GetLocalTransformation())
        if "translation" in values:
            transform.SetTranslation(Gf.Vec3d(*values["translation"]))
        if "rotation_deg" in values:
            x, y, z = values["rotation_deg"]
            rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), x) * Gf.Rotation(Gf.Vec3d(0, 1, 0), y) * Gf.Rotation(Gf.Vec3d(0, 0, 1), z)
            transform.SetRotation(rotation)
        # Keep the local composed transform/scale, replace the static op stack in
        # the selected override layer. Parent transforms remain untouched.
        reset = xform.GetResetXformStack()
        xform.MakeMatrixXform().Set(transform.GetMatrix())
        xform.SetResetXformStack(reset)

    def set_component(self, kind, target, values):
        self.validate_component(kind, target, values)
        prim, Gf = self._prim(target), self.Gf
        if kind == "mass":
            api = self.UsdPhysics.MassAPI(prim)
            old_mass = float(api.GetMassAttr().Get())
            new_inertia = np.asarray(api.GetDiagonalInertiaAttr().Get()) * (values["mass_kg"] / old_mass)
            api.CreateMassAttr().Set(float(values["mass_kg"]))
            api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, new_inertia)))
        elif kind == "friction":
            combined = self._read_friction(prim)
            combined.update(values)
            material = self._owned_material(prim, "physics")
            api = self.UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr().Set(float(combined["static_friction"]))
            api.CreateDynamicFrictionAttr().Set(float(combined["dynamic_friction"]))
            api.CreateRestitutionAttr().Set(float(combined["restitution"]))
            self._bind(prim, material, "physics")
        elif kind == "material":
            combined = self._visual_values(prim)
            combined.update(values)
            texture = combined.get("base_color_texture")
            conversions = self._author_box_uvs(prim) if texture else []
            material = self._owned_material(prim, "visual")
            shader = self.UsdShade.Shader.Define(self.stage, str(material.GetPath()) + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*combined["base_color"]))
            shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(float(combined["roughness"]))
            shader.CreateInput("metallic", self.Sdf.ValueTypeNames.Float).Set(float(combined["metallic"]))
            if texture:
                uv = self.UsdShade.Shader.Define(self.stage, str(material.GetPath()) + "/PrimvarST")
                uv.CreateIdAttr("UsdPrimvarReader_float2")
                uv.CreateInput("varname", self.Sdf.ValueTypeNames.Token).Set("st")
                uv.CreateOutput("result", self.Sdf.ValueTypeNames.Float2)
                reader = self.UsdShade.Shader.Define(self.stage, str(material.GetPath()) + "/BaseColorTexture")
                reader.CreateIdAttr("UsdUVTexture")
                reader.CreateInput("file", self.Sdf.ValueTypeNames.Asset).Set(self.Sdf.AssetPath(texture))
                reader.CreateInput("st", self.Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
                for name, value in (("wrapS", "repeat"), ("wrapT", "repeat"), ("sourceColorSpace", "sRGB")):
                    reader.CreateInput(name, self.Sdf.ValueTypeNames.Token).Set(value)
                reader.CreateInput("scale", self.Sdf.ValueTypeNames.Float4).Set(self.Gf.Vec4f(*combined["base_color"], 1.))
                reader.CreateOutput("rgb", self.Sdf.ValueTypeNames.Float3)
                shader.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "rgb")
                material.GetPrim().SetCustomDataByKey("textureAsset", self.texture_assets[texture])
                if conversions:
                    from pxr import Vt
                    material.GetPrim().SetCustomDataByKey("textureBoxConversions", Vt.StringArray(conversions))
            shader.CreateOutput("surface", self.Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self._bind(prim, material, self.UsdShade.Tokens.allPurpose)
        elif kind == "light":
            light = self.UsdLux.LightAPI(prim)
            if "intensity" in values:
                light.CreateIntensityAttr().Set(float(values["intensity"]))
            if "color_temperature" in values:
                light.CreateEnableColorTemperatureAttr().Set(True)
                light.CreateColorTemperatureAttr().Set(float(values["color_temperature"]))
            if "color" in values:
                light.CreateColorAttr().Set(Gf.Vec3f(*values["color"]))
        elif kind == "camera":
            camera = self.UsdGeom.Camera(prim)
            for name, method in (("focal_length_mm", "CreateFocalLengthAttr"), ("horizontal_aperture_mm", "CreateHorizontalApertureAttr"),
                                 ("vertical_aperture_mm", "CreateVerticalApertureAttr")):
                # USD camera values are tenths of a scene unit. At metersPerUnit=1,
                # one attribute unit is 100 mm (not one mm).
                if name in values:
                    getattr(camera, method)().Set(float(values[name]) / 100.0)
        if {"translation", "rotation_deg"} & set(values):
            self._pose(prim, values)

    def read_component(self, kind, target, fields):
        prim = self._prim(target)
        result = {}
        if kind == "mass":
            result["mass_kg"] = float(self.UsdPhysics.MassAPI(prim).GetMassAttr().Get())
        elif kind == "friction":
            result.update(self._read_friction(prim))
        elif kind == "material":
            result.update(self._visual_values(prim))
        elif kind == "light":
            light = self.UsdLux.LightAPI(prim)
            result.update(intensity=float(light.GetIntensityAttr().Get()), color_temperature=float(light.GetColorTemperatureAttr().Get()), color=list(light.GetColorAttr().Get()))
        elif kind == "camera":
            camera = self.UsdGeom.Camera(prim)
            result.update(focal_length_mm=float(camera.GetFocalLengthAttr().Get()) * 100.0,
                          horizontal_aperture_mm=float(camera.GetHorizontalApertureAttr().Get()) * 100.0,
                          vertical_aperture_mm=float(camera.GetVerticalApertureAttr().Get()) * 100.0)
        if {"translation", "rotation_deg"} & set(fields):
            transform = self.Gf.Transform(self.UsdGeom.Xformable(prim).GetLocalTransformation())
            result["translation"] = list(transform.GetTranslation())
            result["rotation_deg"] = list(reversed(transform.GetRotation().Decompose(self.Gf.Vec3d(0, 0, 1), self.Gf.Vec3d(0, 1, 0), self.Gf.Vec3d(1, 0, 0))))
        return {field: result[field] for field in fields}
