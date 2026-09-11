"""Tests for path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.paths import (
    find_project_root,
    metrics_dir,
    project_root,
    resolve_under_project,
    visdrone_processed_dir,
    visdrone_raw_dir,
)


def test_find_project_root_from_nested_path():
    root = project_root()
    nested = root / "src" / "data"
    assert find_project_root(nested) == root


def test_resolve_helpers_are_under_project():
    root = project_root()
    assert visdrone_raw_dir().is_relative_to(root) or str(visdrone_raw_dir()).startswith(
        str(root)
    )
    assert "visdrone" in str(visdrone_raw_dir())
    assert "visdrone_person" in str(visdrone_processed_dir())
    assert metrics_dir().name == "metrics"
    cfg = resolve_under_project("configs", "visdrone_person.yaml")
    assert cfg.exists()


def test_find_project_root_fails_outside(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_project_root(tmp_path)
