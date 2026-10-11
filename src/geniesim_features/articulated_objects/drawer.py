"""A fixed cabinet and a compound-collider drawer moving on a local-X joint.

Authoring is not a physics simulation. Runtime force/drive application and
contact/limit acceptance require the real Isaac host, never transform animation.
"""

from __future__ import annotations

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from geniesim_features.scene_generation.usd_utils import (
    cube, number, open_box, physics_material, rigid_body, semantics, transform, vector,
)


def author_drawer(stage, parent_path: str, config: dict) -> dict:
    """Create an initially stationary drawer. Root origin is bottom center.

    `size_m` is cabinet depth/width/height; +X is its opening direction.
    `yaw_deg` rotates the whole cabinet, so the rail is not world-X locked.
    Cabinet/drawer collision remains enabled: clearances avoid initial overlap.
    This intentionally does not claim verified contact behavior without PhysX.
    """
    path = Sdf.Path(parent_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError("drawer path must be an absolute non-root prim path")
    if stage.GetPrimAtPath(path):
        raise ValueError(f"drawer prim already exists: {path}")
    d, w, h = vector(config.get("size_m", [0.36, 0.34, 0.24]), 3, "drawer size", positive=True)
    t = number(config.get("wall_thickness_m", 0.012), "wall thickness", positive=True)
    g = number(config.get("clearance_m", 0.006), "drawer clearance", positive=True)
    lower, upper = vector(config.get("limits_m", [0.0, 0.25]), 2, "joint limits")
    initial = number(config.get("initial_position_m", 0.0), "initial_position_m")
    if lower < 0 or upper <= lower or upper > d or not lower <= initial <= upper:
        raise ValueError("drawer limits require 0 <= lower <= initial <= upper <= cabinet depth")
    if config.get("axis", "X") != "X":
        raise ValueError("drawer geometry uses local X; rotate the root with yaw_deg for other world directions")
    inner = [d-2*(t+g), w-2*(t+g), h-2*(t+g)]
    if min(inner) <= 2*t:
        raise ValueError("cabinet is too small for walls and drawer clearances")
    position = vector(config.get("position_m", [0, 0, 0]), 3, "drawer position")
    yaw = number(config.get("yaw_deg", 0), "drawer yaw")
    mass = number(config.get("mass_kg", 1.5), "drawer mass", positive=True)
    cabinet_mass = number(config.get("cabinet_mass_kg", 12.0), "cabinet mass", positive=True)
    drive_config = config.get("drive", {})
    stiffness = number(drive_config.get("stiffness", 0.0), "drive stiffness", nonnegative=True)
    damping = number(drive_config.get("damping", 0.0), "drive damping", nonnegative=True)
    max_force = number(drive_config.get("max_force_n", 100.0), "drive max force", positive=True)
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    transform(root, position, yaw)
    object_id = str(config.get("id", path.name))
    semantics(root, object_id, "drawer_cabinet")
    material = physics_material(stage, f"{path}/PhysicsMaterial", **config.get("physics_material", {}))
    cabinet_path = f"{path}/CabinetBody"
    drawer_path = f"{path}/DrawerBody"
    cabinet = UsdGeom.Xform.Define(stage, cabinet_path).GetPrim()
    drawer = UsdGeom.Xform.Define(stage, drawer_path).GetPrim()
    transform(cabinet, (0, 0, h/2))
    transform(drawer, (initial, 0, h/2))
    rigid_body(cabinet, cabinet_mass)
    rigid_body(drawer, mass)
    semantics(cabinet, object_id, "drawer_cabinet")
    semantics(drawer, object_id, "drawer")
    # Cabinet open at +X. These solids surround, not fill, the drawer cavity.
    cabinet_parts = [
        ("Bottom", (d, w, t), (0, 0, -h/2+t/2)),
        ("Top", (d, w, t), (0, 0, h/2-t/2)),
        ("Back", (t, w, h-2*t), (-d/2+t/2, 0, 0)),
        ("SideLeft", (d-t, t, h-2*t), (t/2, -w/2+t/2, 0)),
        ("SideRight", (d-t, t, h-2*t), (t/2, w/2-t/2, 0)),
    ]
    for name, dims, center in cabinet_parts:
        cube(stage, f"{cabinet_path}/{name}", dims, center, material=material)
    collider_paths = open_box(stage, drawer_path, inner, t, material=material)
    handle_center = (d/2+0.025, 0, 0)
    # A visible solid handle plus two stems, all part of the same rigid body.
    for name, dims, center in [
        ("Handle", (0.014, w*0.38, 0.016), handle_center),
        ("HandleStemLeft", (0.025+t+g, 0.012, 0.012), (d/2-(t+g)/2+0.0125, -w*0.14, 0)),
        ("HandleStemRight", (0.025+t+g, 0.012, 0.012), (d/2-(t+g)/2+0.0125, w*0.14, 0)),
    ]:
        collider_paths.append(f"{drawer_path}/{name}")
        cube(stage, collider_paths[-1], dims, center, material=material, color=(0.2, 0.2, 0.22))
    handle_path = f"{drawer_path}/HandleFrame"
    transform(UsdGeom.Xform.Define(stage, handle_path).GetPrim(), handle_center)
    joint_path = f"{path}/DrawerSlideJoint"
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([cabinet_path])
    joint.CreateBody1Rel().SetTargets([drawer_path])
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    joint.CreateCollisionEnabledAttr(True)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(initial)
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(max_force)
    anchor_path = f"{path}/AnchorJoint"
    anchor = UsdPhysics.FixedJoint.Define(stage, anchor_path)
    # An absent body0 anchors to the world; its frame must be in WORLD space.
    world_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(cabinet)
    world_rotation = world_matrix.ExtractRotationQuat()
    anchor.CreateBody1Rel().SetTargets([cabinet_path])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(world_matrix.ExtractTranslation()))
    anchor.CreateLocalPos1Attr(Gf.Vec3f(0))
    anchor.CreateLocalRot0Attr(Gf.Quatf(world_rotation))
    anchor.CreateLocalRot1Attr(Gf.Quatf(1))
    # Fixed-base articulation discovery; the host still owns initialization.
    UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
    return {
        "object_id": object_id, "root_prim": str(path), "prim_path": str(path),
        "fixed_body_prim": cabinet_path, "body_prim": drawer_path,
        "joint_prim": joint_path, "anchor_joint_prim": anchor_path, "handle_prim": handle_path,
        "axis": "X", "joint_frame_wxyz": [1.0, 0.0, 0.0, 0.0],
        "force_axis_handle": [1.0, 0.0, 0.0], "limits_m": [lower, upper],
        "initial_position_m": initial, "mass_kg": mass,
        "inertia_source": "integrate_compound_box_colliders_in_physics_engine",
        "collision_prims": collider_paths, "physics_material_prim": str(material.GetPath()),
        "runtime_dynamics_verified": False,
    }


def set_drive_target(stage, binding: dict, position_m: float) -> None:
    """Author a bounded target; not a substitute for the host stepping PhysX."""
    target = number(position_m, "drive target")
    lower, upper = binding["limits_m"]
    if not lower <= target <= upper:
        raise ValueError(f"drive target {target} outside [{lower}, {upper}]")
    joint = UsdPhysics.PrismaticJoint.Get(stage, binding["joint_prim"])
    if not joint:
        raise ValueError(f"missing prismatic joint: {binding['joint_prim']}")
    UsdPhysics.DriveAPI(joint.GetPrim(), "linear").GetTargetPositionAttr().Set(target)
