"""Small OpenUSD authoring helpers, with no simulator or global stage state."""

from __future__ import annotations

import math
from typing import Iterable

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade


def vector(value: Iterable[float], length: int, name: str, *, positive=False) -> list[float]:
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite numbers") from exc
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain {length} finite numbers")
    if positive and not all(v > 0 for v in result):
        raise ValueError(f"{name} must be positive")
    return result


def number(value, name: str, *, positive=False, nonnegative=False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"invalid {name}: {value!r}")
    return result


def transform(prim, translation=(0, 0, 0), yaw_deg=0.0):
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xform.AddRotateZOp().Set(float(yaw_deg))


def semantics(prim, object_id: str, semantic_class: str):
    """Stable business labels; an Isaac adapter maps these to renderer semantics."""
    prim.SetCustomDataByKey("object_id", object_id)
    prim.SetCustomDataByKey("semantic_class", semantic_class)
    prim.CreateAttribute("geniesim:objectId", Sdf.ValueTypeNames.String).Set(object_id)
    prim.CreateAttribute("geniesim:semanticClass", Sdf.ValueTypeNames.String).Set(semantic_class)


def physics_material(stage, path, *, static_friction=0.6, dynamic_friction=0.5, restitution=0.0):
    static_friction = number(static_friction, "static_friction", nonnegative=True)
    dynamic_friction = number(dynamic_friction, "dynamic_friction", nonnegative=True)
    restitution = number(restitution, "restitution", nonnegative=True)
    if dynamic_friction > static_friction or restitution > 1:
        raise ValueError("friction requires dynamic <= static; restitution must be <= 1")
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(static_friction)
    api.CreateDynamicFrictionAttr(dynamic_friction)
    api.CreateRestitutionAttr(restitution)
    return material


def cube(stage, path, size, center=(0, 0, 0), *, color=(0.55, 0.6, 0.65), material=None):
    size = vector(size, 3, "cube size", positive=True)
    geom = UsdGeom.Cube.Define(stage, path)
    geom.CreateSizeAttr(1.0)
    geom.AddTranslateOp().Set(Gf.Vec3d(*center))
    geom.AddScaleOp().Set(Gf.Vec3d(*size))
    geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr(True)
    if material:
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material, materialPurpose="physics")
    return geom.GetPrim()


def rigid_body(prim, mass_kg: float, size=None):
    mass = number(mass_kg, "mass_kg", positive=True)
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(mass)
    if size is not None:
        x, y, z = vector(size, 3, "inertia box size", positive=True)
        api.CreateCenterOfMassAttr(Gf.Vec3f(0))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(mass*(y*y+z*z)/12, mass*(x*x+z*z)/12, mass*(x*x+y*y)/12))
        api.CreatePrincipalAxesAttr(Gf.Quatf(1))
    # Without explicit inertia, the physics implementation integrates the actual
    # compound collider volumes and scales their total mass to mass_kg.
    return api


def open_box(stage, path, size, thickness, *, material=None, color=(0.25, 0.45, 0.7)):
    """Five box colliders. Never wrap the empty cavity in a convex hull."""
    x, y, z = vector(size, 3, "container size", positive=True)
    t = number(thickness, "wall_thickness_m", positive=True)
    if 2*t >= min(x, y, z):
        raise ValueError("container wall thickness leaves no cavity")
    parts = [
        ("Bottom", (x, y, t), (0, 0, -z/2+t/2)),
        ("WallXMin", (t, y, z-t), (-x/2+t/2, 0, t/2)),
        ("WallXMax", (t, y, z-t), (x/2-t/2, 0, t/2)),
        ("WallYMin", (x-2*t, t, z-t), (0, -y/2+t/2, t/2)),
        ("WallYMax", (x-2*t, t, z-t), (0, y/2-t/2, t/2)),
    ]
    for name, dims, center in parts:
        cube(stage, f"{path}/{name}", dims, center, material=material, color=color)
    return [f"{path}/{p[0]}" for p in parts]
