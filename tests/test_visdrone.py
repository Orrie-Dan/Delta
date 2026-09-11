"""Unit tests for VisDrone parsing and YOLO conversion (no downloads)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.data.visdrone import (
    PERSON_SOURCE_CATEGORY_IDS,
    VISDRONE_CATEGORY_NAMES,
    YOLO_PERSON_CLASS_ID,
    discover_split_dirs,
    parse_annotation_file,
    parse_annotation_line,
    visdrone_to_yolo,
)
from src.data.visdrone_convert import convert_visdrone
from src.utils.paths import find_project_root, project_root, resolve_under_project


def test_category_mapping_is_explicit():
    assert VISDRONE_CATEGORY_NAMES[1] == "pedestrian"
    assert VISDRONE_CATEGORY_NAMES[2] == "people"
    assert PERSON_SOURCE_CATEGORY_IDS == frozenset({1, 2})
    assert YOLO_PERSON_CLASS_ID == 0


def test_parse_valid_pedestrian_row():
    # bbox_left,top,width,height,score,category,truncation,occlusion
    line = "10,20,30,40,1,1,0,0"
    box, reason = parse_annotation_line(line, image_width=640, image_height=480)
    assert reason is None
    assert box is not None
    assert box.category_id == 1
    assert box.is_person_source
    assert box.left == 10
    assert box.width == 30


def test_parse_people_maps_as_person_source():
    line = "5,5,10,20,1,2,0,1"
    box, reason = parse_annotation_line(line)
    assert reason is None
    assert box is not None
    assert box.category_name == "people"
    yolo = visdrone_to_yolo(box, 100, 100)
    assert yolo is not None
    assert yolo.class_id == 0


def test_ignore_unrelated_class_car():
    line = "10,10,50,50,1,4,0,0"  # car = 4
    box, reason = parse_annotation_line(line)
    assert box is None
    assert reason == "ignored"


def test_invalid_bounding_box_rejected():
    line = "10,10,0,50,1,1,0,0"  # zero width
    box, reason = parse_annotation_line(line)
    assert box is None
    assert reason == "invalid"


def test_malformed_line():
    box, reason = parse_annotation_line("not,enough")
    assert box is None
    assert reason == "malformed"


def test_yolo_coordinate_conversion():
    line = "100,50,50,100,1,1,0,0"
    box, _ = parse_annotation_line(line)
    assert box is not None
    yolo = visdrone_to_yolo(box, image_width=200, image_height=200)
    assert yolo is not None
    assert yolo.class_id == 0
    assert yolo.x_center == pytest.approx(0.625)  # (100+25)/200
    assert yolo.y_center == pytest.approx(0.5)  # (50+50)/200
    assert yolo.width == pytest.approx(0.25)
    assert yolo.height == pytest.approx(0.5)


def test_yolo_rejects_non_person_even_if_called_directly():
    line = "10,10,20,20,1,4,0,0"
    box, reason = parse_annotation_line(line)
    assert reason == "ignored"
    # Constructing via parse won't return a box; ensure converter guards category.
    from src.data.visdrone import VisDroneBox

    car = VisDroneBox(10, 10, 20, 20, 1, 4, 0, 0)
    assert visdrone_to_yolo(car, 100, 100) is None


def test_project_root_helpers():
    root = project_root()
    assert (root / "requirements.txt").is_file()
    assert (root / "src").is_dir()
    assert find_project_root(root / "src" / "data") == root
    resolved = resolve_under_project("configs", "visdrone_person.yaml")
    assert resolved.is_file()


def _write_mini_visdrone(root: Path) -> Path:
    """Create a tiny VisDrone-style layout for converter integration tests."""
    for split, pkg in (
        ("train", "VisDrone2019-DET-train"),
        ("val", "VisDrone2019-DET-val"),
        ("test", "VisDrone2019-DET-test-dev"),
    ):
        img_dir = root / pkg / "images"
        ann_dir = root / pkg / "annotations"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        image_path = img_dir / f"{split}_000.jpg"
        Image.new("RGB", (200, 100), color=(40, 40, 40)).save(image_path)

        if split == "test":
            # No usable GT annotations for test-dev in this fixture.
            continue

        # pedestrian + people + car (+ invalid)
        (ann_dir / f"{split}_000.txt").write_text(
            "\n".join(
                [
                    "10,10,20,30,1,1,0,0",  # pedestrian
                    "50,20,15,25,1,2,0,0",  # people
                    "80,10,40,40,1,4,0,0",  # car (ignored)
                    "0,0,0,10,1,1,0,0",  # invalid
                    "bad-line",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return root


def test_discover_and_convert_synthetic_visdrone(tmp_path: Path):
    source = _write_mini_visdrone(tmp_path / "raw")
    output = tmp_path / "processed"
    discovered = discover_split_dirs(source)
    assert set(discovered) >= {"train", "val", "test"}

    stats = convert_visdrone(source=source, output=output, overwrite=True)
    assert stats["train"].person_boxes == 2
    assert stats["val"].person_boxes == 2
    assert stats["test"].test_without_gt is True

    train_label = (output / "labels" / "train" / "train_000.txt").read_text(encoding="utf-8")
    lines = [ln for ln in train_label.splitlines() if ln.strip()]
    assert len(lines) == 2
    for line in lines:
        assert line.startswith("0 ")

    assert (output / "images" / "train" / "train_000.jpg").is_file()
    assert (output / "images" / "test" / "test_000.jpg").is_file()


def test_parse_annotation_file_stats(tmp_path: Path):
    ann = tmp_path / "a.txt"
    ann.write_text("10,10,5,5,1,1,0,0\n1,1,2,2,1,4,0,0\nbad\n", encoding="utf-8")
    boxes, stats = parse_annotation_file(ann, image_width=100, image_height=100)
    assert len(boxes) == 1
    assert stats.retained_person_boxes == 1
    assert stats.ignored_non_person == 1
    assert stats.malformed_lines == 1
