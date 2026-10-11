# Original-scene cuRobo planner smoke

`scene_motiongen_smoke.py` loads `scene_original/scene.usda` in Isaac Sim, instantiates the same Isaac Franka USD as the verified tabletop PhysX episode, settles for one second, and plans a **panda_hand approach 10 cm above the recorded contact pose** with cuRobo v0.7.6 `MotionGen.plan_single` and official `franka.yml`. It converts all enabled non-robot collision cubes (table, ground, bin, block, etc.) to oriented cuRobo cuboids in the measured `panda_link0` frame. Unknown collision geometry is an error, not silently omitted. The block remains an obstacle; nothing is attached. The goal uses the recorded event-2 hand-to-object transform and the live settled block pose.

On the Isaac Sim server, from the deployed project root (`$PROJECT_ROOT`), with official cuRobo source at `/root/curobo076/src`:

```sh
export PATH=/isaac-sim/kit/python/bin:$PATH
export LD_LIBRARY_PATH=/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/root/curobo076/src${PYTHONPATH:+:$PYTHONPATH}
export TORCH_CUDA_ARCH_LIST=8.9
/isaac-sim/python.sh "$PROJECT_ROOT/experiments/curobo/scene_motiongen_smoke.py" \
  /tmp/scene_motiongen_run.json \
  --scene "$PROJECT_ROOT/scene_original/scene.usda"
```

Use the JSON `success` and `status` fields as the planning verdict: Isaac Sim's `SimulationApp.close()` can exit the interpreter with status 0 even for a failed plan. JSON contains scene hashes, measured frames, collision cuboids, start joints and interpolated waypoints when successful. The archived `scene_motiongen_pregrasp.json` is the successful 10 cm pregrasp; `scene_motiongen_smoke.json` records an earlier contact-height `IK_FAIL`, not a grasp. This smoke does not execute a grasp. For the real cuRobo + PhysX pick-and-place and its verified evidence, use the root README's `franka_motiongen.yaml` command.

若夹爪在闭合前把方块推走，可对完整 EpisodeWriter 回合运行 `python experiments/curobo/inspect_contact_onset.py /path/to/episode`。它报告物体 XY 首次偏移超过 2 mm 前的指尖力和手掌位置；这次失败回合曾显示下降途中一侧手指先接触并施加高力，而成功回合的下降路径最大横向偏差为 1.24 mm。
