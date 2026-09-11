"""Tests for dataset audit helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data.dataset_audit import audit_dataset, save_audit_report


def test_audit_counts_and_small_boxes(tmp_path: Path):
    root = tmp_path / "ds"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    Image.new("RGB", (100, 100), color=128).save(img_dir / "a.jpg")
    Image.new("RGB", (100, 100), color=64).save(img_dir / "b.jpg")

    # One normal box + one very small box (area 0.0001 < 0.001)
    (lbl_dir / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.01 0.01\n",
        encoding="utf-8",
    )
    (lbl_dir / "b.txt").write_text("", encoding="utf-8")

    report = audit_dataset(root, splits=("train",), small_box_area_threshold=0.001)
    split = report.splits["train"]
    assert split.num_images == 2
    assert split.num_label_files == 2
    assert split.total_person_boxes == 2
    assert split.images_with_zero_boxes == 1
    assert split.very_small_bounding_boxes == 1

    out = tmp_path / "audit.json"
    save_audit_report(report, out)
    assert out.is_file()
