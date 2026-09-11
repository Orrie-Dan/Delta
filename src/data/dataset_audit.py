"""Audit a YOLO person-detection dataset (e.g. processed VisDrone).

CLI example::

    python -m src.data.dataset_audit \\
        --dataset data/processed/visdrone_person \\
        --report outputs/metrics/visdrone_person_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from src.data.visdrone import CANONICAL_SPLITS, IMAGE_EXTENSIONS
from src.utils.paths import metrics_dir, project_root, visdrone_processed_dir


@dataclass
class SplitAudit:
    split: str
    num_images: int = 0
    num_label_files: int = 0
    total_person_boxes: int = 0
    avg_boxes_per_image: float = 0.0
    images_with_zero_boxes: int = 0
    missing_label_files: list[str] = field(default_factory=list)
    missing_image_files: list[str] = field(default_factory=list)
    malformed_annotation_lines: int = 0
    invalid_bounding_boxes: int = 0
    very_small_bounding_boxes: int = 0
    bbox_width_stats: dict[str, float] = field(default_factory=dict)
    bbox_height_stats: dict[str, float] = field(default_factory=dict)
    bbox_area_stats: dict[str, float] = field(default_factory=dict)
    image_width_stats: dict[str, float] = field(default_factory=dict)
    image_height_stats: dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetAuditReport:
    dataset: str
    small_box_area_threshold: float
    splits: dict[str, SplitAudit] = field(default_factory=dict)
    totals: dict[str, float | int] = field(default_factory=dict)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p05": 0.0,
            "p95": 0.0,
        }
    ordered = sorted(values)
    n = len(ordered)

    def percentile(p: float) -> float:
        if n == 1:
            return ordered[0]
        idx = (n - 1) * p
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)

    return {
        "count": n,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / n),
        "median": float(percentile(0.5)),
        "p05": float(percentile(0.05)),
        "p95": float(percentile(0.95)),
    }


def _parse_yolo_line(
    line: str,
) -> tuple[tuple[int, float, float, float, float] | None, str | None]:
    text = line.strip()
    if not text:
        return None, "empty"
    parts = text.split()
    if len(parts) != 5:
        return None, "malformed"
    try:
        class_id = int(float(parts[0]))
        xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except ValueError:
        return None, "malformed"
    if w <= 0 or h <= 0:
        return None, "invalid"
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
        # Allow slight floating noise outside [0,1] as invalid for audit purposes
        if xc < -0.01 or xc > 1.01 or yc < -0.01 or yc > 1.01 or w > 1.01 or h > 1.01:
            return None, "invalid"
    return (class_id, xc, yc, w, h), None


def audit_split(
    dataset_root: Path,
    split: str,
    *,
    small_box_area_threshold: float = 0.001,
) -> SplitAudit:
    """Audit one YOLO split under ``dataset_root``."""
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    report = SplitAudit(split=split)

    if not images_dir.is_dir():
        return report

    images = [
        p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    label_files = {
        p.stem: p
        for p in labels_dir.iterdir()
        if labels_dir.is_dir() and p.is_file() and p.suffix.lower() == ".txt"
    } if labels_dir.is_dir() else {}

    report.num_images = len(images)
    report.num_label_files = len(label_files)

    image_stems = {p.stem for p in images}
    for stem, label_path in label_files.items():
        if stem not in image_stems:
            report.missing_image_files.append(label_path.name)

    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    img_ws: list[float] = []
    img_hs: list[float] = []
    box_counts: list[int] = []

    for image_path in images:
        label_path = label_files.get(image_path.stem)
        if label_path is None:
            report.missing_label_files.append(image_path.name)
            box_counts.append(0)
            report.images_with_zero_boxes += 1
            continue

        try:
            with Image.open(image_path) as im:
                iw, ih = im.size
            img_ws.append(float(iw))
            img_hs.append(float(ih))
        except OSError:
            iw, ih = 0, 0

        n_boxes = 0
        text = label_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if not line.strip():
                continue
            parsed, reason = _parse_yolo_line(line)
            if reason == "malformed":
                report.malformed_annotation_lines += 1
                continue
            if reason == "invalid":
                report.invalid_bounding_boxes += 1
                continue
            assert parsed is not None
            _cid, _xc, _yc, w, h = parsed
            n_boxes += 1
            area = w * h
            widths.append(w)
            heights.append(h)
            areas.append(area)
            if area < small_box_area_threshold:
                report.very_small_bounding_boxes += 1

        box_counts.append(n_boxes)
        report.total_person_boxes += n_boxes
        if n_boxes == 0:
            report.images_with_zero_boxes += 1

    report.avg_boxes_per_image = (
        float(sum(box_counts) / len(box_counts)) if box_counts else 0.0
    )
    report.bbox_width_stats = _stats(widths)
    report.bbox_height_stats = _stats(heights)
    report.bbox_area_stats = _stats(areas)
    report.image_width_stats = _stats(img_ws)
    report.image_height_stats = _stats(img_hs)
    return report


def audit_dataset(
    dataset_root: Path,
    *,
    splits: tuple[str, ...] = CANONICAL_SPLITS,
    small_box_area_threshold: float = 0.001,
) -> DatasetAuditReport:
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    report = DatasetAuditReport(
        dataset=str(dataset_root),
        small_box_area_threshold=small_box_area_threshold,
    )
    for split in splits:
        report.splits[split] = audit_split(
            dataset_root,
            split,
            small_box_area_threshold=small_box_area_threshold,
        )

    report.totals = {
        "num_images": sum(s.num_images for s in report.splits.values()),
        "num_label_files": sum(s.num_label_files for s in report.splits.values()),
        "total_person_boxes": sum(s.total_person_boxes for s in report.splits.values()),
        "images_with_zero_boxes": sum(
            s.images_with_zero_boxes for s in report.splits.values()
        ),
        "very_small_bounding_boxes": sum(
            s.very_small_bounding_boxes for s in report.splits.values()
        ),
        "malformed_annotation_lines": sum(
            s.malformed_annotation_lines for s in report.splits.values()
        ),
        "invalid_bounding_boxes": sum(
            s.invalid_bounding_boxes for s in report.splits.values()
        ),
    }
    return report


def save_audit_report(report: DatasetAuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": report.dataset,
        "small_box_area_threshold": report.small_box_area_threshold,
        "totals": report.totals,
        "splits": {name: asdict(split) for name, split in report.splits.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_audit_summary(report: DatasetAuditReport) -> None:
    print("=" * 60)
    print(f"Dataset audit: {report.dataset}")
    print(f"Small-box area threshold (normalized): {report.small_box_area_threshold}")
    print("-" * 60)
    for name, split in report.splits.items():
        print(
            f"[{name}] images={split.num_images} labels={split.num_label_files} "
            f"boxes={split.total_person_boxes} avg/img={split.avg_boxes_per_image:.2f} "
            f"zero-box={split.images_with_zero_boxes} "
            f"tiny={split.very_small_bounding_boxes} "
            f"missing_lbl={len(split.missing_label_files)} "
            f"missing_img={len(split.missing_image_files)} "
            f"malformed={split.malformed_annotation_lines} "
            f"invalid={split.invalid_bounding_boxes}"
        )
        if split.bbox_area_stats.get("count", 0):
            a = split.bbox_area_stats
            print(
                f"      bbox area: min={a['min']:.6f} median={a['median']:.6f} "
                f"p05={a['p05']:.6f} mean={a['mean']:.6f} max={a['max']:.6f}"
            )
        if split.image_width_stats.get("count", 0):
            w = split.image_width_stats
            h = split.image_height_stats
            print(
                f"      image size: "
                f"{w['min']:.0f}x{h['min']:.0f} .. {w['max']:.0f}x{h['max']:.0f} "
                f"(median {w['median']:.0f}x{h['median']:.0f})"
            )
    print("-" * 60)
    print(f"TOTALS: {report.totals}")
    print("=" * 60)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a YOLO person dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Processed dataset root (default: data/processed/visdrone_person)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: outputs/metrics/visdrone_person_audit.json)",
    )
    parser.add_argument(
        "--small-box-area",
        type=float,
        default=0.001,
        help="Normalized area threshold for 'very small' boxes (default: 0.001)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = project_root()
    dataset = args.dataset or visdrone_processed_dir(root)
    report_path = args.report or (metrics_dir(root) / "visdrone_person_audit.json")
    if not dataset.is_absolute():
        dataset = (root / dataset).resolve()
    if not report_path.is_absolute():
        report_path = (root / report_path).resolve()

    try:
        report = audit_dataset(
            dataset, small_box_area_threshold=args.small_box_area
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    save_audit_report(report, report_path)
    print_audit_summary(report)
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
