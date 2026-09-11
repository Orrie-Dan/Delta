"""Visualize YOLO person annotations (save to disk; no GUI required).

CLI example::

    python -m src.data.visualize_annotations \\
        --dataset data/processed/visdrone_person \\
        --split val \\
        --count 12
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from src.data.visdrone import CANONICAL_SPLITS, IMAGE_EXTENSIONS, YOLO_PERSON_CLASS_NAME
from src.utils.paths import project_root, visdrone_processed_dir, visualizations_dir


def _list_split_images(dataset_root: Path, split: str) -> list[Path]:
    images_dir = dataset_root / "images" / split
    if not images_dir.is_dir():
        return []
    return [
        p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _load_yolo_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            continue
        boxes.append((xc, yc, w, h))
    return boxes


def draw_person_boxes(
    image_bgr: np.ndarray,
    boxes_xywhn: list[tuple[float, float, float, float]],
    *,
    label: str = YOLO_PERSON_CLASS_NAME,
) -> np.ndarray:
    """Draw normalized YOLO boxes on a BGR image; returns a new image."""
    out = image_bgr.copy()
    ih, iw = out.shape[:2]
    color = (0, 200, 0)
    for xc, yc, w, h in boxes_xywhn:
        x1 = int(round((xc - w / 2.0) * iw))
        y1 = int(round((yc - h / 2.0) * ih))
        x2 = int(round((xc + w / 2.0) * iw))
        y2 = int(round((yc + h / 2.0) * ih))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def visualize_split(
    dataset_root: Path,
    split: str,
    output_dir: Path,
    *,
    count: int = 12,
    seed: int = 42,
) -> list[Path]:
    """Sample images, draw person boxes, save under ``output_dir``."""
    dataset_root = dataset_root.resolve()
    images = _list_split_images(dataset_root, split)
    if not images:
        raise FileNotFoundError(
            f"No images found for split '{split}' under {dataset_root / 'images' / split}"
        )

    rng = random.Random(seed)
    sample = images if len(images) <= count else rng.sample(images, count)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    labels_dir = dataset_root / "labels" / split

    for image_path in sample:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"WARNING: could not read {image_path}", file=sys.stderr)
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = _load_yolo_boxes(label_path)
        vis = draw_person_boxes(bgr, boxes)
        out_path = output_dir / f"{split}_{image_path.stem}_vis.jpg"
        cv2.imwrite(str(out_path), vis)
        saved.append(out_path)

    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO person annotations (saves images to disk)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Processed dataset root (default: data/processed/visdrone_person)",
    )
    parser.add_argument(
        "--split",
        choices=list(CANONICAL_SPLITS),
        default="val",
        help="Dataset split to sample",
    )
    parser.add_argument("--count", type=int, default=12, help="Number of images to visualize")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: outputs/visualizations/visdrone_person)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = project_root()
    dataset = args.dataset or visdrone_processed_dir(root)
    output = args.output or (visualizations_dir(root) / "visdrone_person")
    if not dataset.is_absolute():
        dataset = (root / dataset).resolve()
    if not output.is_absolute():
        output = (root / output).resolve()

    try:
        saved = visualize_split(
            dataset,
            args.split,
            output,
            count=args.count,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(saved)} visualization(s) to {output}")
    for path in saved:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
