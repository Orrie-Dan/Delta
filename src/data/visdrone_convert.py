"""Convert VisDrone DET annotations to YOLO person-only labels.

CLI example::

    python -m src.data.visdrone_convert \\
        --source data/raw/visdrone \\
        --output data/processed/visdrone_person
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from src.data.visdrone import (
    CANONICAL_SPLITS,
    SPLIT_TEST,
    annotation_path_for_image,
    list_images,
    parse_annotation_file,
    validate_source_layout,
    visdrone_to_yolo,
    yolo_to_label_text,
)
from src.utils.paths import project_root, visdrone_processed_dir, visdrone_raw_dir


@dataclass
class ConvertStats:
    images_copied: int = 0
    labels_written: int = 0
    empty_labels: int = 0
    missing_annotations: int = 0
    person_boxes: int = 0
    malformed_lines: int = 0
    invalid_boxes: int = 0
    ignored_non_person: int = 0
    skipped_existing: int = 0
    test_without_gt: bool = False
    notes: list[str] = field(default_factory=list)


def _ensure_output_dirs(output_root: Path) -> None:
    for split in CANONICAL_SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _copy_or_link_image(src: Path, dst: Path, *, use_symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            # Fall back to copy on platforms / mounts that block symlinks.
            pass
    shutil.copy2(src, dst)


def convert_split(
    *,
    split_name: str,
    images_dir: Path,
    annotations_dir: Path | None,
    output_root: Path,
    copy_images: bool = True,
    use_symlink: bool = False,
    overwrite: bool = False,
    limit: int | None = None,
) -> ConvertStats:
    """Convert one split to YOLO person-only format."""
    stats = ConvertStats()
    out_img = output_root / "images" / split_name
    out_lbl = output_root / "labels" / split_name
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    if limit is not None:
        images = images[: max(0, limit)]

    has_any_ann = annotations_dir is not None and any(annotations_dir.glob("*.txt"))
    if split_name == SPLIT_TEST and not has_any_ann:
        stats.test_without_gt = True
        stats.notes.append(
            "test split has no usable ground-truth annotations; "
            "images will be copied and empty label files will NOT be invented "
            "as detections - empty placeholder labels are written only so YOLO "
            "folder layout stays consistent (0 person boxes)."
        )

    for image_path in images:
        dst_image = out_img / image_path.name
        dst_label = out_lbl / f"{image_path.stem}.txt"

        if dst_image.exists() and dst_label.exists() and not overwrite:
            stats.skipped_existing += 1
            continue

        if copy_images:
            _copy_or_link_image(image_path, dst_image, use_symlink=use_symlink)
            stats.images_copied += 1
        elif not dst_image.exists():
            _copy_or_link_image(image_path, dst_image, use_symlink=use_symlink)
            stats.images_copied += 1

        ann_path = annotation_path_for_image(image_path, annotations_dir)
        if ann_path is None:
            stats.missing_annotations += 1
            # Empty label file — image has no usable GT for person boxes.
            dst_label.write_text("", encoding="utf-8")
            stats.labels_written += 1
            stats.empty_labels += 1
            continue

        try:
            with Image.open(image_path) as im:
                image_width, image_height = im.size
        except OSError as exc:
            stats.notes.append(f"Failed to read image size for {image_path.name}: {exc}")
            continue

        boxes, parse_stats = parse_annotation_file(
            ann_path, image_width=image_width, image_height=image_height
        )
        stats.malformed_lines += parse_stats.malformed_lines
        stats.invalid_boxes += parse_stats.invalid_boxes
        stats.ignored_non_person += parse_stats.ignored_non_person

        yolo_boxes = []
        for box in boxes:
            yolo = visdrone_to_yolo(box, image_width, image_height)
            if yolo is None:
                stats.invalid_boxes += 1
                continue
            yolo_boxes.append(yolo)

        stats.person_boxes += len(yolo_boxes)
        dst_label.write_text(yolo_to_label_text(yolo_boxes), encoding="utf-8")
        stats.labels_written += 1
        if not yolo_boxes:
            stats.empty_labels += 1

    return stats


def convert_visdrone(
    source: Path,
    output: Path,
    *,
    splits: list[str] | None = None,
    copy_images: bool = True,
    use_symlink: bool = False,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, ConvertStats]:
    """Convert discovered VisDrone splits to YOLO person-only dataset."""
    source = source.resolve()
    output = output.resolve()
    discovered = validate_source_layout(source)
    _ensure_output_dirs(output)

    selected = splits or list(discovered.keys())
    results: dict[str, ConvertStats] = {}

    for split_name in selected:
        if split_name not in discovered:
            raise FileNotFoundError(
                f"Requested split '{split_name}' not found under {source}. "
                f"Available: {sorted(discovered)}"
            )
        info = discovered[split_name]
        print(f"Converting split '{split_name}' from {info.source_label} ...")
        stats = convert_split(
            split_name=split_name,
            images_dir=info.images_dir,
            annotations_dir=info.annotations_dir,
            output_root=output,
            copy_images=copy_images,
            use_symlink=use_symlink,
            overwrite=overwrite,
            limit=limit,
        )
        results[split_name] = stats
        print(
            f"  images={stats.images_copied} labels={stats.labels_written} "
            f"person_boxes={stats.person_boxes} empty_labels={stats.empty_labels} "
            f"missing_ann={stats.missing_annotations}"
        )
        for note in stats.notes:
            print(f"  NOTE: {note}")

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert VisDrone DET (pedestrian+people → person) to YOLO format. "
            "Does not download data. Does not create uniform-class labels."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Raw VisDrone root (default: data/raw/visdrone under project root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YOLO dataset root (default: data/processed/visdrone_person)",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        choices=list(CANONICAL_SPLITS),
        help="Convert only this split (repeatable). Default: all discovered splits.",
    )
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy images into the processed tree (default: true)",
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Prefer symlinks for images (falls back to copy if unsupported)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted images/labels",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit images per split (development/testing)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = project_root()

    source = args.source or visdrone_raw_dir(root)
    output = args.output or visdrone_processed_dir(root)
    if not source.is_absolute():
        source = (root / source).resolve()
    if not output.is_absolute():
        output = (root / output).resolve()

    try:
        convert_visdrone(
            source=source,
            output=output,
            splits=args.splits,
            copy_images=args.copy_images,
            use_symlink=args.symlink_images,
            overwrite=args.overwrite,
            limit=args.limit,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: conversion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Done. YOLO person dataset written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
