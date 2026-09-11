"""Tests for YOLO person dataset sanity checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.evaluation.dataset_checks import validate_person_yolo_dataset


def _make_person_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (64, 64), color=80).save(root / "images" / split / "a.jpg")
        (root / "labels" / split / "a.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    return root


def test_validate_person_dataset_ok(tmp_path: Path):
    ds = _make_person_dataset(tmp_path / "ok")
    report = validate_person_yolo_dataset(ds)
    assert report.splits["train"].num_images == 1
    assert report.splits["val"].num_labels == 1
    assert report.label_class_ids == [0]


def test_validate_rejects_non_person_class(tmp_path: Path):
    ds = _make_person_dataset(tmp_path / "bad")
    (ds / "labels" / "train" / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected class"):
        validate_person_yolo_dataset(ds)


def test_validate_missing_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        validate_person_yolo_dataset(tmp_path / "missing")
