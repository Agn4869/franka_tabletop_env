"""Explicit, local-only asset resolution and provenance (no asset downloads)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
import yaml

from .usd_utils import number, vector


BUILTINS = {
    "graspable_block": {"primitive": "box", "size_m": [0.06, 0.06, 0.06], "mass_kg": 0.15, "dynamic": True, "semantic_class": "graspable_object"},
    "open_container": {"primitive": "open_container", "size_m": [0.24, 0.20, 0.12], "dynamic": False, "semantic_class": "container"},
    "drawer_cabinet": {"primitive": "drawer", "size_m": [0.36, 0.34, 0.24], "dynamic": False, "semantic_class": "drawer_cabinet"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResolvedAsset:
    asset_id: str
    primitive: str
    size_m: list[float]
    dynamic: bool
    mass_kg: float
    semantic_class: str
    path: str | None = None
    prim_path: str | None = None
    source_meters_per_unit: float = 1.0
    source_up_axis: str = "Z"
    source_center_m: list[float] | None = None
    sha256: str | None = None
    dependencies: list[dict] | None = None
    license: str = "self-authored primitive geometry"

    def to_dict(self):
        return asdict(self)


class AssetResolver:
    def __init__(self, manifest: dict | str | Path | None = None, *, base_dir: Path | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else None
        if isinstance(manifest, (str, Path)):
            path = self._absolute_path(manifest)
            self.base_dir = path.parent
            with path.open(encoding="utf-8") as stream:
                manifest = yaml.safe_load(stream)
        self.assets = dict(BUILTINS)
        if manifest is not None:
            if not isinstance(manifest, dict) or not isinstance(manifest.get("assets", {}), dict):
                raise ValueError("asset manifest requires an assets mapping")
            self.assets.update(manifest.get("assets", {}))

    def _absolute_path(self, value) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            if self.base_dir is None:
                raise ValueError(f"relative asset path requires explicit base_dir: {value}")
            path = self.base_dir / path
        return path.resolve(strict=True)

    def resolve(self, asset_id: str) -> ResolvedAsset:
        if asset_id not in self.assets:
            raise ValueError(f"unknown asset alias: {asset_id}")
        entry = self.assets[asset_id]
        if not isinstance(entry, dict):
            raise ValueError(f"asset {asset_id}: entry must be a mapping")
        primitive = entry.get("primitive", "external")
        if primitive not in ("box", "open_container", "drawer", "external"):
            raise ValueError(f"asset {asset_id}: unsupported primitive {primitive}")
        common = dict(asset_id=asset_id, primitive=primitive, dynamic=bool(entry.get("dynamic", primitive == "box")),
                      mass_kg=number(entry.get("mass_kg", 0.15), "asset mass", positive=True),
                      semantic_class=str(entry.get("semantic_class", asset_id)))
        if primitive != "external":
            return ResolvedAsset(**common, size_m=vector(entry["size_m"], 3, "asset size", positive=True))
        for key in ("path", "source_meters_per_unit", "source_up_axis", "license"):
            if key not in entry or entry[key] in (None, ""):
                raise ValueError(f"asset {asset_id}: external asset requires explicit {key}")
        try:
            path = self._absolute_path(entry["path"])
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"asset {asset_id}: cannot resolve {entry['path']}: {exc}") from exc
        if not path.is_file():
            raise ValueError(f"asset {asset_id}: not a file: {path}")
        units = number(entry["source_meters_per_unit"], "source units", positive=True)
        axis = str(entry["source_up_axis"]).upper()
        if axis not in ("Y", "Z"):
            raise ValueError(f"asset {asset_id}: source_up_axis must be Y or Z")
        stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
        if stage is None:
            raise ValueError(f"asset {asset_id}: cannot open USD {path}")
        layers, asset_paths, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
        if unresolved:
            raise ValueError(f"asset {asset_id}: unresolved references/payloads: {list(unresolved)}")
        deps = set()
        for layer in layers:
            if layer.realPath:
                deps.add(Path(layer.realPath).resolve())
        for dependent in asset_paths:
            deps.add(Path(dependent).resolve())
        if any(not p.is_file() for p in deps):
            raise ValueError(f"asset {asset_id}: dependencies must be local readable files")
        root = stage.GetPrimAtPath(entry["prim_path"]) if entry.get("prim_path") else stage.GetDefaultPrim()
        if not root:
            raise ValueError(f"asset {asset_id}: requires valid prim_path or defaultPrim")
        # Avoid nesting arbitrary robot/articulation physics under a scaled rigid
        # wrapper. Such assets require their dedicated robot/object adapter.
        subtree = list(Usd.PrimRange(root))
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Joint) for p in subtree):
            raise ValueError(f"asset {asset_id}: external rigid/jointed asset needs a dedicated unscaled adapter")
        if not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in subtree):
            raise ValueError(f"asset {asset_id}: manipulation asset has no authored collision geometry")
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]).ComputeRelativeBound(root, root.GetParent()).ComputeAlignedRange()
        if bbox.IsEmpty():
            raise ValueError(f"asset {asset_id}: no finite geometry bound")
        low, high = np.array(bbox.GetMin()), np.array(bbox.GetMax())
        points = np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
        if axis == "Y":
            points = points[:, [0,2,1]] * [1,-1,1]  # +90 degrees around X
        points *= units
        low, high = points.min(axis=0), points.max(axis=0)
        return ResolvedAsset(**common, size_m=vector(high-low, 3, "external bound", positive=True),
                             path=str(path), prim_path=str(root.GetPath()), source_meters_per_unit=units,
                             source_up_axis=axis, source_center_m=((high+low)/2).tolist(),
                             sha256=sha256_file(path), license=str(entry["license"]),
                             dependencies=[{"path": str(p), "sha256": sha256_file(p)} for p in sorted(deps)])
