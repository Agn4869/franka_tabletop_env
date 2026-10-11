"""Export measured drawer JSONL to CSV/SVG; missing channels remain missing."""
import csv
import html
import json
from pathlib import Path

import numpy as np


def _scalar(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric scalar")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def export_drawer_curves(states_jsonl, output_dir):
    """Create drawer_curves.csv and drawer_curves.svg without overwriting files.

    The third trace is the submitted external-force norm or drive position target,
    never a relabelled measured contact force. Net contact force is separate CSV
    data. No values are synthesized for the initial, uncommanded observation.
    """
    source, output_dir = Path(states_jsonl), Path(output_dir)
    states = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("State must be a JSON object")
                time = _scalar(item["simulation_time"], "simulation_time", nonnegative=True)
                position = _scalar(item["position_m"], "position_m")
                velocity = _scalar(item["velocity_m_s"], "velocity_m_s")
                step = item["step_index"]
                if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                    raise ValueError("step_index must be a nonnegative integer")
                if states and time <= states[-1]["time_s"]:
                    raise ValueError("State times must be strictly increasing")
                if states and step <= states[-1]["physics_step"]:
                    raise ValueError("State steps must be strictly increasing")
                mode = item["mode"]
                if mode not in {"external_force", "joint_drive"} or (states and states[-1]["mode"] != mode):
                    raise ValueError("One file must contain one known interaction mode")
                command = item.get("applied_command") or {}
                if not isinstance(command, dict) or command.get("mode", mode) != mode:
                    raise ValueError("Invalid or mismatched applied command")
                force = command.get("force_handle_n")
                if force is not None:
                    force = np.asarray(force, float)
                    if force.shape != (3,) or not np.isfinite(force).all():
                        raise ValueError("Invalid recorded handle-frame force")
                quantity = "submitted_force_norm_n" if mode == "external_force" else "submitted_drive_target_m"
                value = command.get("force_norm_n") if mode == "external_force" else command.get("position_m")
                if value is not None:
                    value = _scalar(value, "recorded command target", nonnegative=mode == "external_force")
                if force is not None and value is not None and mode == "external_force" and not np.isclose(np.linalg.norm(force), value, rtol=1e-5, atol=1e-7):
                    raise ValueError("Recorded force norm disagrees with force vector")
                contacts = item.get("contacts") or {}
                if not isinstance(contacts, dict) or type(contacts.get("available", False)) is not bool:
                    raise ValueError("Invalid contact availability record")
                contact_value = contacts.get("net_force_norm_n") if contacts.get("available") else None
                if contact_value is not None:
                    contact_value = _scalar(contact_value, "measured contact force", nonnegative=True)
                states.append({"time_s": time, "physics_step": step, "position_m": position,
                    "velocity_m_s": velocity, "command_quantity": quantity, "command_value": value,
                    "force_handle_x_n": None if force is None else float(force[0]),
                    "force_handle_y_n": None if force is None else float(force[1]),
                    "force_handle_z_n": None if force is None else float(force[2]),
                    "contact_available": bool(contacts.get("available")), "net_contact_force_norm_n": contact_value,
                    "mode": mode})
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(f"Invalid drawer state at line {line_number}: {error}") from error
    if not states:
        raise ValueError("No observed drawer states to export")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, svg_path = output_dir / "drawer_curves.csv", output_dir / "drawer_curves.svg"
    if any(path.exists() or path.is_symlink() for path in (csv_path, svg_path)):
        raise FileExistsError("Drawer curve outputs already exist")
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(states[0]))
        writer.writeheader()
        writer.writerows(states)
    width, height, left, plot_width, plot_height = 960, 700, 100, 800, 150
    start, stop = states[0]["time_s"], states[-1]["time_s"]
    xspan = stop-start if stop > start else 1.
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="25" y="28" font-family="sans-serif" font-size="16">Observed drawer samples: {html.escape(source.name)}</text>',
              '<text x="25" y="49" font-family="sans-serif" font-size="11">Command target is not measured contact force. Missing commands are left blank.</text>']
    traces = [("position_m", "Observed displacement (m)"), ("velocity_m_s", "Observed velocity (m/s)"),
              ("command_value", "Submitted force norm (N)" if states[0]["mode"] == "external_force" else "Submitted drive position target (m)")]
    for index, (key, title) in enumerate(traces):
        top = 85 + index*200
        available = [float(item[key]) for item in states if item[key] is not None]
        low, high = (min(available), max(available)) if available else (0., 1.)
        if high-low < 1e-12:
            pad = max(.001, abs(high)*.1)
            low, high = low-pad, high+pad
        pieces.extend([f'<text x="{left}" y="{top-12}" font-family="sans-serif" font-size="13">{html.escape(title)}</text>',
            f'<path d="M {left} {top} V {top+plot_height} H {left+plot_width}" fill="none" stroke="#777"/>',
            f'<text x="8" y="{top+8}" font-family="monospace" font-size="11">{high:.5g}</text>',
            f'<text x="8" y="{top+plot_height}" font-family="monospace" font-size="11">{low:.5g}</text>',
            f'<text x="{left}" y="{top+plot_height+17}" font-family="monospace" font-size="11">{start:.4g}s</text>',
            f'<text x="{left+plot_width-50}" y="{top+plot_height+17}" font-family="monospace" font-size="11">{stop:.4g}s</text>'])
        segments, current = [], []
        for item in states:
            if item[key] is None:
                if current:
                    segments.append(current)
                    current = []
                continue
            x = left + (item["time_s"]-start)/xspan*plot_width
            y = top + (high-float(item[key]))/(high-low)*plot_height
            current.append((x, y))
        if current:
            segments.append(current)
        for segment in segments:
            points = " ".join(f"{x:.3f},{y:.3f}" for x, y in segment)
            pieces.append(f'<polyline points="{points}" fill="none" stroke="#1672ac" stroke-width="1.5"/>')
            for x, y in segment:
                pieces.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2" fill="#1672ac"/>')
        if not available:
            pieces.append(f'<text x="{left+25}" y="{top+55}" font-family="sans-serif" font-size="14">No submitted command target in recorded states</text>')
    pieces.append('</svg>')
    with svg_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(pieces) + "\n")
    return {"csv": str(csv_path), "svg": str(svg_path), "rows": len(states),
            "command_quantity": states[0]["command_quantity"], "source": str(source),
            "evidence_scope": "plots_of_input_observations_not_physics_validation"}
