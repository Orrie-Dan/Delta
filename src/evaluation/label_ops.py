"""Helpers to load YOLO labels and run size-bin recall over a dataset split."""

from __future__ import annotations

from pathlib import Path

from src.data.visdrone import IMAGE_EXTENSIONS
from src.evaluation.size_analysis import (
    Box,
    compute_gt_size_distribution,
    compute_size_bin_recall,
)


def parse_yolo_label_file(path: Path) -> list[Box]:
    boxes: list[Box] = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            _cid = int(float(parts[0]))
            xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        boxes.append(Box(xc, yc, w, h))
    return boxes


def iter_split_gt_boxes(dataset_root: Path, split: str = "val") -> list[Box]:
    labels_dir = dataset_root / "labels" / split
    boxes: list[Box] = []
    if not labels_dir.is_dir():
        return boxes
    for path in sorted(labels_dir.iterdir()):
        if path.suffix.lower() == ".txt":
            boxes.extend(parse_yolo_label_file(path))
    return boxes


def list_split_images(dataset_root: Path, split: str = "val") -> list[Path]:
    images_dir = dataset_root / "images" / split
    if not images_dir.is_dir():
        return []
    return [
        p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def gt_size_distribution_for_split(
    dataset_root: Path,
    split: str = "val",
    *,
    small_area_threshold: float = 0.001,
    medium_area_threshold: float = 0.01,
) -> dict:
    boxes = iter_split_gt_boxes(dataset_root, split)
    return compute_gt_size_distribution(
        boxes,
        small_area_threshold=small_area_threshold,
        medium_area_threshold=medium_area_threshold,
    )


def size_bin_recall_from_predictions(
    image_predictions: list[tuple[Path, list[Box]]],
    dataset_root: Path,
    split: str = "val",
    *,
    iou_threshold: float = 0.5,
    small_area_threshold: float = 0.001,
    medium_area_threshold: float = 0.01,
) -> dict:
    """Compute size-bin recall given ``(image_path, pred_boxes)`` pairs."""
    labels_dir = dataset_root / "labels" / split
    pairs: list[tuple[list[Box], list[Box]]] = []
    for image_path, preds in image_predictions:
        gt = parse_yolo_label_file(labels_dir / f"{image_path.stem}.txt")
        pairs.append((gt, preds))
    return compute_size_bin_recall(
        pairs,
        iou_threshold=iou_threshold,
        small_area_threshold=small_area_threshold,
        medium_area_threshold=medium_area_threshold,
    )
