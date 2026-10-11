"""Run a real cuRobo v0.7.6 Franka plan in Isaac Sim's Python interpreter.

This checks planner execution, not grasp physics or alignment with the tabletop USD.
"""
import json
from pathlib import Path
import sys

from isaacsim import SimulationApp


def main(output):
    app = SimulationApp({"headless": True})
    try:
        import torch
        from curobo.geom.types import Cuboid, WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        tensor = TensorDeviceType(device=torch.device("cuda:0"))
        motion = MotionGen(MotionGenConfig.load_from_robot_config(
            "franka.yml", WorldConfig(cuboid=[]), tensor,
            interpolation_dt=0.02, collision_cache={"obb": 32}))
        motion.warmup()
        motion.update_world(WorldConfig(cuboid=[Cuboid(
            name="obstacle", pose=[0.9, 0, 0.5, 1, 0, 0, 0], dims=[0.1, 0.5, 0.5])]))
        joints = [f"panda_joint{i}" for i in range(1, 8)]
        start = JointState.from_position(
            tensor.to_device([[0, -1.3, 0, -2.5, 0, 1, 0]]), joint_names=joints)
        goal = Pose(position=tensor.to_device([[0.5, 0, 0.3]]),
                    quaternion=tensor.to_device([[1, 0, 0, 0]]))
        result = motion.plan_single(start, goal)
        success = bool(result.success.item())
        report = {"planner": "cuRobo MotionGen.plan_single", "robot": "franka.yml",
                  "success": success, "status": str(result.status),
                  "joint_names": joints, "goal_base_xyz_m": [0.5, 0, 0.3],
                  "scene": "official_free_space_smoke_not_tabletop"}
        if success:
            plan = result.get_interpolated_plan()
            report["waypoints"] = int(plan.position.shape[0])
            report["interpolation_dt_s"] = float(result.interpolation_dt)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
        return 0 if success else 1
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
