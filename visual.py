"""Isaac Sim RGB evidence for each physical pick/place phase.

Import and construct only after SimulationApp startup. Uses the same frozen
physics step for pose sampling and image capture; does not advance simulation.
"""

from pathlib import Path

import numpy as np


def _look_at(position, target):
    forward = np.asarray(target, dtype=float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(right, forward), -forward))


class VisualEvidence:
    def __init__(self, stage, directory, center):
        from pxr import Gf, UsdGeom
        import omni.replicator.core as rep

        self.rep = rep
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = "/World/FrankaEvidenceCamera"
        position = np.asarray(center, dtype=float) + np.array([0.75, -0.95, 0.8])
        rotation = _look_at(position, np.asarray(center, dtype=float))
        quaternion = Gf.Matrix3d(*rotation.T.reshape(-1)).ExtractRotation().GetQuat()
        cam = UsdGeom.Camera.Define(stage, self.path)
        xform = UsdGeom.Xformable(cam)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quaternion)
        cam.CreateFocalLengthAttr(24.0)
        cam.CreateHorizontalApertureAttr(36.0)
        cam.CreateVerticalApertureAttr(20.25)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 10.0))
        rep.orchestrator.set_capture_on_play(False)
        self.product = rep.create.render_product(self.path, (960, 540))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.rgb.attach([self.product])
        self.names = []

    def capture(self, world, name):
        from PIL import Image

        world.pause()
        before = world.current_time
        try:
            self.rep.orchestrator.step(delta_time=0.0, rt_subframes=4, pause_timeline=True)
            if abs(world.current_time - before) > 1e-9:
                raise RuntimeError("Snapshot changed physics time")
            rgb = np.asarray(self.rgb.get_data())
            if rgb.shape != (540, 960, 4):
                raise RuntimeError(f"Unexpected RGB image shape: {rgb.shape}")
            filename = f"{name}.png"
            Image.fromarray(rgb[..., :3].copy()).save(self.directory / filename)
            self.names.append(filename)
            print("SNAPSHOT", self.directory / filename, flush=True)
        finally:
            world.play()

    def close(self):
        self.rgb.detach([self.product.path])
        self.product.destroy()
