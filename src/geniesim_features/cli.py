"""Honest CLI: offline USD authoring is separate from real Isaac execution."""

import argparse
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys


def _positive_int(value):
    result = int(value)
    if not 0 < result <= 1000000:
        raise argparse.ArgumentTypeError("must be an integer in [1,1000000]")
    return result


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def _seed(value):
    result = int(value)
    if not 0 <= result < 2**64:
        raise argparse.ArgumentTypeError("must be an unsigned 64-bit integer")
    return result


def parser():
    root = argparse.ArgumentParser(prog="geniesim-features", description="USD scene authoring and real Isaac simulation features")
    commands = root.add_subparsers(dest="command", required=True)
    scene = commands.add_parser("scene", help="author real composed USD; no physics simulation")
    scene.add_argument("--config", type=Path, required=True)
    scene.add_argument("--assets", type=Path)
    scene.add_argument("--output", type=Path, required=True)
    scene.add_argument("--seed", type=_seed, help="override scene configuration seed for real layout/asset sampling")
    randomize = commands.add_parser("randomize", help="USD-only variants and parameter read-back; does not render")
    randomize.add_argument("--scene", type=Path, required=True, help="source scene bundle directory")
    randomize.add_argument("--config", type=Path, required=True)
    randomize.add_argument("--output", type=Path, required=True)
    randomize.add_argument("--seed", type=_seed, default=42)
    randomize.add_argument("--samples", type=_positive_int, default=1)
    commands.add_parser("doctor", help="JSON dependency discovery; does not start or validate the simulator")
    cloud = commands.add_parser("pointcloud", help="convert axial metric .npy depth and optical calibration to world PLY")
    cloud.add_argument("--depth", type=Path, required=True)
    cloud.add_argument("--calibration", type=Path, required=True)
    cloud.add_argument("--output", type=Path, required=True)
    batch = commands.add_parser("batch", help="bounded explicit scene/seed episodes; real child-process Franka demonstrations")
    batch.add_argument("--config", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--python", dest="python_executable", type=Path, help="engine Python executable or Isaac python.sh (no shell arguments)")
    batch.add_argument("--timeout-seconds", type=_finite_float, default=600.0, help="default total authoring + engine budget per episode")
    replay = commands.add_parser("replay", help="display recorded observations; does not reexecute physics")
    replay.add_argument("--episode", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    demonstrate = commands.add_parser("franka-demonstrate", help="run a recorded Franka pick-and-place using real Isaac Sim")
    demonstrate.add_argument("--scene", type=Path, required=True, help="scene bundle directory")
    demonstrate.add_argument("--output", type=Path, required=True)
    demonstrate.add_argument("--config", type=Path, required=True, help="sensor configuration YAML")
    demonstrate.add_argument("--camera", action="append", help="camera ID; repeat for multiple cameras")
    display = demonstrate.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--gui", dest="headless", action="store_false")
    demonstrate.set_defaults(headless=True)
    demonstrate.add_argument("--physics-hz", type=_positive_int, default=60)
    demonstrate.add_argument("--control-hz", type=_positive_int, default=60)
    demonstrate.add_argument("--task", type=Path, required=True, help="Franka pick-and-place task YAML")
    demonstrate.add_argument("--backend", choices=("franka_pickplace", "curobo_motiongen"),
                             help="override task YAML backend; defaults to the original Franka controller")
    demonstrate.add_argument("--max-intervals", type=_positive_int, default=3600)
    demonstrate.add_argument("--hold-gui", action="store_true",
                             help="GUI only: retain final physics state until the window closes")
    curobo = commands.add_parser("curobo-demonstrate", help="run the original cuRobo/G2 demonstration backend")
    curobo.add_argument("--scene", type=Path, required=True)
    curobo.add_argument("--output", type=Path, required=True)
    curobo.add_argument("--config", type=Path, required=True, help="sensor configuration YAML")
    curobo.add_argument("--camera", action="append")
    curobo.add_argument("--robot-profile", type=Path, required=True)
    curobo.add_argument("--task", type=Path, required=True)
    curobo.add_argument("--physics-hz", type=_positive_int, default=120)
    curobo.add_argument("--control-hz", type=_positive_int, default=30)
    curobo.add_argument("--max-intervals", type=_positive_int, default=900)
    curobo_display = curobo.add_mutually_exclusive_group()
    curobo_display.add_argument("--headless", dest="headless", action="store_true")
    curobo_display.add_argument("--gui", dest="headless", action="store_false")
    curobo.set_defaults(headless=True)
    curobo.add_argument("--hold-gui", action="store_true")
    return root


def doctor():
    dependencies = {}
    for module, distribution in (("numpy", "numpy"), ("yaml", "PyYAML"), ("PIL", "Pillow"),
                                 ("pxr", "usd-core"), ("isaacsim", "isaacsim")):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, AttributeError):
            available = False
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        dependencies[module] = {"module_discoverable": available, "distribution": distribution, "version": version}
    return {"schema_version": 1, "python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(), "dependencies": dependencies,
            "usd_authoring_dependencies_discoverable": all(dependencies[key]["module_discoverable"] for key in ("pxr", "numpy", "yaml")),
            "simulator_started": False, "runtime_validated": False,
            "note": "Module discovery does not verify GPU/driver, physics, rendering, or robot assets."}


def _pointcloud(args):
    import numpy as np
    from .sensors import CameraCalibration, depth_to_points, write_ply

    with args.calibration.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("depth_semantics") != "distance_to_image_plane_m":
        raise ValueError("calibration must explicitly declare distance_to_image_plane_m depth_semantics")
    if metadata.get("projection_model", "pinhole") != "pinhole":
        raise ValueError("pointcloud supports only undistorted pinhole calibration")
    if metadata.get("optical_axes", "x_right_y_down_z_forward") != "x_right_y_down_z_forward":
        raise ValueError("pointcloud needs optical X-right/Y-down/Z-forward axes")
    calibration = CameraCalibration(**{key: metadata[key] for key in ("width", "height", "K", "T_world_optical", "near", "far")})
    depth = np.load(args.depth, allow_pickle=False)
    points = depth_to_points(depth, calibration)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite pointcloud output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.output, points)
    return {"status": "written", "output": str(args.output.absolute()), "coordinate_frame": "world",
            "points": len(points), "depth_semantics": "distance_to_image_plane_m"}


def main(argv=None):
    try:
        args = parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        exit_code = 0
        if args.command == "scene":
            from .workflows import build_scene
            manifest = build_scene(args.config, args.output, args.assets, seed=args.seed)
            result = {"status": manifest["status"], "scene_file": manifest["scene_file"],
                      "execution_mode": "usd_only", "physics_validated": False}
        elif args.command == "randomize":
            from .workflows import randomize_usd
            manifest = randomize_usd(args.scene, args.config, args.output, seed=args.seed, samples=args.samples)
            result = {"status": manifest["status"], "execution_mode": "usd_only", "rendered": False,
                      "physics_simulated": False, "samples_authored": manifest["samples_authored"],
                      "manifest": str((args.output/"run_manifest.json").absolute())}
        elif args.command == "doctor":
            result = doctor()
        elif args.command == "pointcloud":
            result = _pointcloud(args)
        elif args.command == "batch":
            from .batch import run_batch
            summary = run_batch(args.config,args.output,python_executable=args.python_executable,timeout_seconds=args.timeout_seconds)
            result = {"status":summary["status"],"batch_manifest":str((args.output/"batch_manifest.json").absolute()),
                      "all_episodes_succeeded":summary["all_episodes_succeeded"],"counts":summary["counts"]}
            exit_code = 0 if summary["all_episodes_succeeded"] else 1
        elif args.command == "replay":
            from .recording.replay import EpisodeReader
            reader = EpisodeReader(args.episode)
            html = reader.export_observation_replay(args.output)
            result = {"output":str(html),"action_count":reader.action_count,"observation_count":reader.observation_count,
                      "mode":"recorded_observation_display","physics_reexecuted":False}
        elif args.command == "curobo-demonstrate":
            if args.physics_hz % args.control_hz:
                raise ValueError("physics_hz must be divisible by control_hz")
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(f"Demonstration output exists: {args.output}")
            args._defer_app_shutdown = True
            from .runtime import run_demonstration
            result = run_demonstration(args)

        else:
            if args.physics_hz is not None and args.control_hz is not None and args.physics_hz % args.control_hz:
                raise ValueError("physics_hz must be divisible by control_hz")
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(f"refusing to overwrite runtime output: {args.output}")
            # Never import Isaac-dependent adapters during offline commands.
            args._defer_app_shutdown = True
            from .franka_runtime import run_franka_demonstration
            result = run_franka_demonstration(args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str))
        if args.command in ("franka-demonstrate", "curobo-demonstrate"):
            exit_code = 0 if result is not None and result.get("success") is True else 1
        return exit_code
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        print(f"geniesim-features: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 2
        return 2
    except KeyboardInterrupt:
        exit_code = 130
        print("geniesim-features: interrupted", file=sys.stderr)
        return exit_code
    except Exception:
        import traceback
        exit_code = 1
        traceback.print_exc()
        return exit_code
    finally:
        if getattr(args, "_defer_app_shutdown", False):
            # Kit can terminate Python here. Persisted results and diagnostics
            # must precede shutdown; failures retain a nonzero exit status.
            sys.stdout.flush()
            sys.stderr.flush()
            from .adapters.isaac_world import shutdown_simulation_app
            shutdown_simulation_app(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
