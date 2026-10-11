# Isaac Sim Franka Tabletop Pick-and-Place Environment

本目录独立于生成场景的原项目 `geniesim-features/`：只使用 Isaac Sim 5.1 自带的 Franka Panda、原项目生成的桌面 USD 及其 `bindings.json`；不导入 G2、cuRobo 或 `geniesim_features`。原项目文件与原始场景资产不被修改。需准备有 GPU 的 Isaac Sim 5.1 Linux 服务器和可 SSH 连接的 Windows 客户端。

目录约定：`run.py` 是 Isaac Sim 物理执行和严格验收；`visual.py` 负责证据相机；`launch.ps1` 从本地上传脚本与原始场景、启动服务器、再拉取结果；`scene_original/` 是四个保持原样的 USD 场景文件；`evidence/` 保存成功的 JSON、阶段图片和 GUI 实拍。Isaac Sim 自带的 Franka USD 从服务器安装目录加载，不随本目录重复打包。

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
| 机器人与规划 | 原 `adapters/g2_robot.py`、`curobo_planner.py` 和 `manipulation/` 针对 G2 几何、关节映射及夹爪标定；当时的 G2 实验未通过完整抓放 | Isaac 自带 Franka USD、夹爪驱动和 `PickPlaceController`；把机器人放在可达位置并延长接近阶段，让手掌收敛至目标 |
| 坐标与接触 | 场景以 Z 轴朝上、米为单位；方块随机偏航，不保证平行于机器人手指 | 用落稳后的实际物体四元数提取偏航，构造世界系目标姿态；由 Isaac 官方控制器完成基座/末端变换，PhysX 验证两指真实接触 |
| 成功条件 | 仅看控制器走完会漏掉未夹住、搬运滑落和碰到盒沿 | 记录接触、相对滑移、连续抬升、释放点整个方块在盒口内，以及 120 帧后的盒腔包容与静止；每个状态配对 RGB 画面 |

**可迁移的做法：**先把“机器人够不够得着”与“夹爪是否真正夹住”分开验证；再分别检查“搬运没滑落”和“松手时目标在盒口内”。坐标与目标来自实测世界位姿和几何边界，而不是从截图凭感觉改固定偏移。原始蓝盒位置已经有成功证据；不同高度和尺寸的 GUI07 场景必须单独校准，不能直接复用这次的成功结论。

旧 G2 USD/URDF 和候选标定数据属于原项目的独立历史归档，**不包含在本仓库**。原项目与 G2 耦合的库源码也不随本仓库发布：它和原工程的相机采集、历史回合记录及批处理共享接口，不能为了 Franka 实验误删。保留旧 G2 代码不代表 G2 硬件永远无法抓取，只说明那套旧配置没有获得物理成功证据。
