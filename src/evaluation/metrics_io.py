"""Serialize experiment metrics / metadata as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], path: Path, *, indent: int = 2) -> Path:
    """Write ``data`` to ``path`` as UTF-8 JSON; create parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, default=str), encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_metrics_template(
    *,
    experiment_name: str,
    model: str,
    image_size: int,
    epochs_requested: int,
) -> dict[str, Any]:
    """Return a metrics dict with null placeholders until training fills them."""
    return {
        "experiment_name": experiment_name,
        "model": model,
        "image_size": image_size,
        "epochs_requested": epochs_requested,
        "best_epoch": None,
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "checkpoint_size_mb": None,
        "parameter_count": None,
        "avg_latency_ms": None,
        "approx_fps": None,
        "size_bin_recall": None,
        "gt_size_distribution": None,
    }
