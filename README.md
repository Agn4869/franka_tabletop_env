# Isaac Sim Franka Tabletop Pick-and-Place Environment

本目录包含两个层次：`run.py` 是已验证的独立 Franka PickPlaceController 物理基线；`src/geniesim_features/` 是接入原始 GenieSim 场景、观测、记录与评测的独立支线。Isaac Sim 5.1 自带的 Franka Panda 与原始 `scene_original/` USD 不被修改。**Franka + cuRobo v0.7.6 MotionGen 已在原始场景完成一次真实 PhysX 抓放并可视化验证**；这与旧 G2 抓放未成功的历史结果不同。

目录约定：`run.py` 是基线验收；`src/geniesim_features/franka_runtime.py` 是原项目链路后端；`config/tasks/`、`config/sensors/`、`config/randomization/` 和 `config/batches/` 保存可复现实验；`evidence/` 保存结果。Isaac Sim 自带的 Franka USD 从服务器安装目录加载，不随本目录重复打包。

**已验证状态：** 原始 `scene_original/` 已在新服务器上通过集成链路完成一次真实抓放：`success: true`，双指持续接触 849 帧，最大抬升 0.26828 m，物体完整入盒并静置；记录为 2392 个动作、2393 个观测，`data_status: complete`。

GitHub 保存可运行的源码、场景、配置、结果摘要和关键截图。集成基线的完整逐帧回合 `evidence/geniesim_e2e_success/` 与 cuRobo 成功回合压缩档 `evidence/curobo_motiongen_success/full_episode.tar.zst` 保留在本地，不作为仓库附件；服务器原始完整回合亦保留。仓库中的 `summary.json` 和图片仅供核对结果，不能代替完整回合的哈希复验。

## 输入和运行

验收输入是从原项目场景输出原样复制的 `scene_original/`：`scene.usda`、`layout.usda`、`bindings.json`、`scene_manifest.json`；桌面上表面 0.75 m，方块尺寸 60×50×70 mm。脚本从 USD 世界坐标的台面、物体和盒腔计算目标；基座在方块的 Y 负方向 0.28 m，并依据落稳后物体的真实偏航角对齐夹爪。**不移动蓝色盒子**。

在服务器上进入本目录后，也可以直接运行：

```bash
/path/to/isaac-sim/python.sh ./run.py \
  --scene ./scene_original --output ./episode.json --capture
```

`--capture` 在 `episode_frames/` 生成 `phase_00.png` 至 `phase_10.png` 和 `settled.png`；即使无桌面窗口，也渲染真实仿真状态。服务器桌面中加 `--gui --capture` 打开 Isaac Sim 窗口并把视角设为同一证据相机。程序退出码为 0 且 JSON 中 `success: true` 才表示严格验收通过。`--bin-xy X Y` 仅用于**临时诊断**盒子位置，改动保留在 USD session layer，不写回原场景。

用户截图对应的另一个 `tabletop_gui07` 场景**经过修改**：桌面上表面 0.86901 m、方块约 42×35×49 mm。它在先前的实验中抓起过小方块，但没有稳定放入盒内；不要把它的失败结果与此处原始场景的成功结果混淆。

### 从本地再次启动

在 Windows PowerShell 中，从本目录运行（`-Server` 为 SSH 登录目标，`-RemoteDirectory` 为服务器上有写权限的绝对路径；按需调整端口与 Isaac Sim 路径）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch.ps1 -Server 'user@host' -Port 22 -RemoteDirectory '/home/user/franka_tabletop_env' -IsaacPython '/path/to/isaac-sim/python.sh'
```

结果自动保存到本地 `evidence/episode.json` 和 `evidence/episode_frames/`。要在服务器远程桌面里观看实时 Isaac Sim，再加 `-Gui`；结尾会停留 20 秒供检查，画面关闭后仍会拉回图像。启动脚本不会读写或安装 `geniesim-features`，也不记录密码。如远程桌面同时有多个 Isaac Sim 窗口，请看顶部显示 `FrankaEvidenceCamera` 的窗口。

### 原始 GenieSim 链路

服务器上用 Isaac Sim 的 Python 运行集成后端：

```bash
PYTHONPATH=/path/to/franka_tabletop_env/src \
/path/to/isaac-sim/python.sh -m geniesim_features franka-demonstrate \
  --scene ./scene_original \
  --config ./config/sensors/franka_fixed.yaml \
  --task ./config/tasks/franka_pick_place.yaml \
  --output ./evidence/geniesim_e2e_success \
  --headless --physics-hz 60 --control-hz 60
```

这个入口执行真实 PhysX，不使用吸附；每个控制动作都对应下一帧观测。`manifest.json` 必须显示 `data_status: complete`，且 `observation_count = action_count + 1`。批量入口使用 `config/batches/franka_e2e.yaml`；其中 `deliberate_unreachable` 是失败分类对照，不应计入成功率。

Franka 运行时由 `config/tasks/franka_pick_place.yaml` 或 `config/tasks/franka_motiongen.yaml` 明确选择后端；前者保留 PickPlaceController，后者将 cuRobo `MotionGen.plan_single()` 生成的关节轨迹施加到 ArticulationController，并使用真实夹爪与 PhysX 接触、抬升、入盒判定。原始 G2 `CuroboPlanner` 独立保留，不将 G2 profile 当成 Franka profile。`run.py` 仍固定使用 PickPlaceController，不能用其结果证明 cuRobo 成功。

原始 cuRobo 后端入口：

```bash
PYTHONPATH=/path/to/franka_tabletop_env/src \
/path/to/isaac-sim/python.sh -m geniesim_features curobo-demonstrate \
  --scene ./scene_original \
  --config ./config/sensors/franka_fixed.yaml \
  --robot-profile /absolute/path/to/calibrated_g2_profile.yaml \
  --task /absolute/path/to/pick_place.yaml \
  --output ./evidence/curobo_attempt \
  --headless
```

原始 G2 `curobo-demonstrate` 入口仍使用 G2 profile，不能直接传给 Franka。服务器已从 NVIDIA 官方源码检出 cuRobo `v0.7.6`；运行时需让 Isaac Python 可见源码、Torch CUDA 共享库与 Ninja。独立规划检查：`evidence/motiongen_smoke.json` 为自由空间示例（61 路点）；`evidence/scene_motiongen_pregrasp.json` 为原始 USD 的 12 个碰撞体与实测对象位姿上的预抓取规划（40 路点）。这两个检查**不构成物理抓放证据**。

### cuRobo MotionGen 真实抓放（原始场景）

服务器 `/root/franka_tabletop_env` 上运行，远程桌面可见时指定 `DISPLAY=:20`；仅需离线验证时将 `--gui` 改为 `--headless` 并省略 `DISPLAY`：

```bash
cd /root/franka_tabletop_env
export DISPLAY=:20
export PATH=/isaac-sim/kit/python/bin:$PATH
export LD_LIBRARY_PATH=/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/root/franka_tabletop_env/src:/root/franka_tabletop_env:/root/curobo076/src
export TORCH_CUDA_ARCH_LIST=8.9
/isaac-sim/python.sh -m geniesim_features franka-demonstrate \
  --scene scene_original --config config/sensors/franka_fixed.yaml \
  --task config/tasks/franka_motiongen.yaml \
  --output evidence/curobo_new_run --gui --physics-hz 60 --control-hz 60
```

选择 `franka_motiongen.yaml` 后执行 `approach → descend → clearance → lift → transfer → lower` 六次真实 `plan_single()`；下降的 FK 夹取走廊检测拒绝横向绕行。闭合只靠 PhysX 双指接触，方块离桌后才把**规划碰撞几何**加入 cuRobo（不是物理吸附）；盒口上方释放，由真实重力落入盒中。`evidence/curobo_motiongen_success/` 保存阶段图片、`summary.json` 及本地未纳入 Git 的 `full_episode.tar.zst`；完整回合亦在服务器 `evidence/curobo_visual_attempt_06/`。服务器完整回合已由 `EpisodeReader(verify_hashes=True)` 验证；本地压缩档 SHA256 见 `summary.json`。结果：1426 个动作 / 1427 个观测、双指接触 650 帧、最大抬升 0.19523 m、相对滑移 0.000070 m，方块完整入盒并静置。早期失败的原因是下降轨迹横向扫过方块，导致单侧先接触并把方块推走；不是蓝盒位置导致抓取失败。

## 判断标准

- `controller_done` 只说明官方示例状态机走完，**不等于抓放成功**。
- `finger_contacts` / `sustained_two_finger_contact_frames`：PhysX 中 Franka 两指分别接触同一动态方块，且抬升搬运中双指持续夹持至少 10 帧。
- `max_lift_m` / `lift_held_frames`：实际离桌超过 5 cm，连续保持至少 10 帧；`max_grip_relative_slip_m` 须 ≤35 mm，排除搬运途中滑落。
- `release_xy_inside_bin`：释放瞬间整个物体的旋转包围盒在盒口内，四周保留 3 mm；释放速度须 ≤0.3 m/s。
- `object_inside_bin` / `object_settled`：释放后再推进 120 物理帧，整个物体的旋转包围盒位于盒腔，最后 30 帧位移 <5 mm。`controller_done` **单独不构成成功证据**。
- `event_samples` 与逐阶段图片分别校对接近、双指夹取、抬升、越过盒口、松开、沉底；判定基于实时物理接触和刚体位姿，不把物体吸附、瞬移或禁用碰撞。

## 已复现结果及故障定位

本地 `evidence/episode.json` 是执行 `launch.ps1` 后从服务器取回的完整结果，`success: true`：双指接触 **849 帧**，有效搬运抬升 **497 帧**，最高抬升 **0.26449 m**，夹持相对滑移 **0.00251 m**。释放时物体整体位于盒口内，落稳中心 `(0.28125, 0.10628, 0.79500) m`，完全位于盒腔内。`evidence/episode_frames/` 中 `phase_03` 对准方块、`phase_05` 夹起、`phase_06` 搬运至盒上、`phase_07` 降至盒口、`settled` 松开且沉底。图像未显示末端撞击盒壁或物体在搬运途中滑走。带 GUI 的 `evidence/gui_result.json` 也得到 `success: true`；`evidence/original_gui_verified.png` 是服务器远程桌面**实际 Isaac Sim 窗口**，顶部显示证据相机，方块位于盒内。

最关键的测量因素是原始方块落稳后旋转约 **0.20669 rad / 11.84°**。未对齐夹爪的原始场景实验虽出现短暂双指接触，却无法形成稳定夹持；按实际方块偏航对齐后，在**同一个未移动的盒子**里完整放入并静置。因此原始场景不是“盒子摆放导致不可完成”，不能把另一个修改场景的失败直接归咎于所有盒子位置。

如果其它场景失败，依次看 `event_samples` 的到位误差、夹爪两侧接触、离桌后的相对滑移、`phase_06` 至 `phase_07` 的盒壁间隙、释放时整个物体是否在盒口内。仅当具体画面或世界坐标证实目标不可达/边缘碰撞时，再用 `--base-offset-x/y`、`--bin-xy` 做单变量诊断；不要盲调 TCP 偏移或关闭碰撞。

## 从原代码到可用机器人的变化

| 层次 | 原项目承担的工作 | 本实验的增量 |
|---|---|---|
| 场景与目标 | `config/scenes/tabletop.yaml` 和 `scene_generation/` 生成自绘桌面、动态方块、盒腔以及 USD `bindings.json` | 原封不动使用 `scene_original/`；依据 USD 的真实世界边界和落稳位姿选择抓取与放置目标 |
| 机器人与规划 | 原 `adapters/g2_robot.py`、`curobo_planner.py` 和 `manipulation/` 针对 G2 几何与关节映射；G2 实验未通过完整抓放 | Isaac 自带 Franka USD/夹爪；保留官方 PickPlaceController 基线，并新增独立的 Franka cuRobo `MotionGen` 规划与真实关节执行 |
| 坐标与接触 | 场景以 Z 轴朝上、米为单位；方块偏航随机 | 从落稳对象求取抓取目标，校准 `panda_hand` 与物体中心偏移；接近轨迹做 FK 横向走廊检测，避免单侧先撞物体；仅由 PhysX 验证双指接触 |
| 成功条件 | 仅看控制器走完会漏掉未夹住、搬运滑落和碰到盒沿 | 记录接触、相对滑移、连续抬升、释放点整个方块在盒口内，以及 120 帧后的盒腔包容与静止；每个状态配对 RGB 画面 |

**可迁移的做法：**先把“机器人够不够得着”与“夹爪是否真正夹住”分开验证；再分别检查“搬运没滑落”和“松手时目标在盒口内”。坐标与目标来自实测世界位姿和几何边界，而不是从截图凭感觉改固定偏移。原始蓝盒位置已经有成功证据；不同高度和尺寸的 GUI07 场景必须单独校准，不能直接复用这次的成功结论。

旧 G2 USD/URDF 与候选标定数据在原项目独立归档，**不随本仓库发布**。本支线保留原始 G2 适配器、cuRobo planner 和 manipulation 接口源码用于对照，但 Franka 运行使用独立配置和后端，不将未成功的 G2 实验称作成功。G2 profile 校准与 Franka 成功证据是两件事。
