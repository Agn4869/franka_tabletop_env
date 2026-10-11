"""Configuration boundaries shared by the CLI and engine adapters."""

from pathlib import Path
import json
import math
import re
import yaml


class DependencyUnavailable(RuntimeError):
    """A real engine dependency is missing; never substitute synthetic success."""


def load_config(path):
    path = Path(path).resolve(strict=True)
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected configuration mapping: {path}")
    if result.get("schema_version", 1) != 1:
        raise ValueError("Unsupported schema_version")
    return result


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Unsafe identifier: {value!r}")
    return value


def positive_int(value, name, maximum=1000000):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def finite_float(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def write_json(path, value):
    """Caller owns the fresh output directory; never overwrite an existing file."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
