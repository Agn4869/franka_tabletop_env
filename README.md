# Franka Tabletop Pick-and-Place Environment

Isaac Sim 5.1 / PhysX environment for Franka Panda tabletop pick-and-place, with two verified planning backends:

- **PickPlaceController** (Isaac Sim built-in) — 849 sustained two-finger contact frames, 0.26m lift
- **cuRobo v0.7.6 MotionGen** — 650 sustained contact frames, 0.20m lift, 0.07mm grip slip

Both produce `success: true` on the original scene with real PhysX contact, no suction or teleportation.

## Repository Structure

```
run.py                           Standalone Franka PickPlaceController baseline
visual.py                        GUI visualization runner
launch.ps1                       Windows PowerShell remote launcher
scene_original/                  Original USD scene (table 0.75m, block 60x50x70mm)
config/
  tasks/                         franka_pick_place.yaml, franka_motiongen.yaml
  sensors/                       Camera and observation config
  batches/                       Batch experiment definitions
  randomization/                 Domain randomization config
src/geniesim_features/
  franka_runtime.py              Franka runtime orchestrator
  execution/                     franka_backend.py, curobo_backend.py
  manipulation/                  Trajectory, geometry, executor
  adapters/                      Isaac world, sensors, robot profile, cuRobo planner
  recording/                     Episode recording and replay
  scene_generation/              USD scene builder, layout, assets
  sensors/                       Geometry, visibility, models
  domain_randomization/          Sampling, USD adapter
evidence/                        Verified results, phase images, summaries
experiments/curobo/              cuRobo smoke tests and diagnostics
```

## Quick Start

### Headless (produces episode data)

```bash
/isaac-sim/python.sh run.py \
  --scene scene_original \
  --output episode.json \
  --capture
```

`--capture` generates phase images in `episode_frames/`. Exit code 0 with `success: true` means pass.

### With GUI (Isaac Sim viewport)

```bash
export DISPLAY=:20
/isaac-sim/python.sh run.py \
  --scene scene_original \
  --output episode.json \
  --gui --capture
```

### Remote Launch (from Windows)

```powershell
.\launch.ps1 -Server 'user@host' -Port 22 `
  -RemoteDirectory '/path/to/franka_tabletop_env' `
  -IsaacPython '/isaac-sim/python.sh'
```

Add `-Gui` for remote desktop visualization. Results auto-download to `evidence/`.

### GenieSim Integration

```bash
PYTHONPATH=src /isaac-sim/python.sh -m geniesim_features franka-demonstrate \
  --scene scene_original \
  --config config/sensors/franka_fixed.yaml \
  --task config/tasks/franka_pick_place.yaml \
  --output evidence/output \
  --headless --physics-hz 60 --control-hz 60
```

Switch `--task` to `franka_motiongen.yaml` for cuRobo backend. cuRobo v0.7.6 requires Torch CUDA and Ninja.

## Success Criteria

| Metric | Requirement |
|--------|-------------|
| Two-finger contact | Both fingers contact the block simultaneously |
| Sustained grip | >= 10 consecutive frames of dual contact |
| Lift height | > 50mm above table surface |
| Grip slip | <= 35mm relative displacement during transport |
| Release position | Entire rotated bounding box inside bin opening (3mm margin) |
| Release speed | <= 0.3 m/s |
| Settled in bin | Object inside bin cavity, < 5mm drift over final 30 frames |

`controller_done` alone does NOT constitute success.

## Verified Results

### PickPlaceController Baseline

| Metric | Value |
|--------|-------|
| Two-finger contact frames | 849 |
| Held lift frames | 497 |
| Max lift | 0.265m |
| Grip slip | 2.51mm |
| Object settled in bin | Yes |

### cuRobo MotionGen

| Metric | Value |
|--------|-------|
| Planning phases | approach, descend, clearance, lift, transfer, lower |
| Actions / Observations | 1426 / 1427 |
| Two-finger contact frames | 650 |
| Max lift | 0.195m |
| Grip slip | 0.07mm |
| Object settled in bin | Yes |

Phase images in `evidence/episode_frames/` and `evidence/curobo_motiongen_success/`.

## Requirements

- Isaac Sim 5.x (tested 5.1.0)
- Python 3.11+, NumPy, PyYAML, Pillow
- cuRobo v0.7.6 (for MotionGen backend only)

## Notes

- The original scene block has ~11.8 deg yaw after settling; gripper alignment accounts for this
- `scene_original/` is used as-is from the upstream project; the blue bin is not moved
- G2 robot adapters are preserved for reference but Franka uses independent backends
- Full episode archives (`full_episode.tar.zst`) are kept locally, not in Git
