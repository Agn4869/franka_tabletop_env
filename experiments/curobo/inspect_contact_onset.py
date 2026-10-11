"""Inspect recorded PhysX onset of object motion; never modifies the episode."""
import json
from pathlib import Path
import sys

import numpy as np


def main(root):
    original = None
    previous = []
    print("index phase object_xy hand_xy finger_forces_N deviation_m")
    for row in (root / "observations.jsonl").open(encoding="utf-8"):
        item = json.loads(row)
        obs = item["observation"]
        if not obs["phase"] in ("approach", "approach_settle", "descend", "gripper_close", "clearance"):
            continue
        def vector(group, key):
            return np.load(root / obs[group][key]["path"], allow_pickle=False)
        block = vector("object", "xyz_m")
        hand = vector("robot", "hand_xyz_m")
        force = np.linalg.norm(np.load(root / obs["finger_object_force_n"]["path"], allow_pickle=False), axis=1)
        if original is None:
            original = block.copy()
        shift = float(np.linalg.norm(block[:2] - original[:2]))
        sample = (item["observation_index"], obs["phase"], block[:2].tolist(), hand[:2].tolist(), force.tolist(), shift)
        previous.append(sample)
        if len(previous) > 6:
            previous.pop(0)
        if shift > .002:
            for entry in previous:
                print(*entry)
            return
    print("No XY displacement greater than 2 mm in recorded phases")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(strict=True))
