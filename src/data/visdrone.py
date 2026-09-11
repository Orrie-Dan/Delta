"""VisDrone DET dataset discovery and annotation parsing.

VisDrone is used ONLY for person detection training data preparation.
It is NOT a uniform-recognition dataset. Do not derive our_team /
known_friendly / unknown labels from VisDrone.

Official VisDrone2019-DET object categories (object_category field)::

    0  ignored regions
    1  pedestrian
    2  people
    3  bicycle
    4  car
    5  van
    6  truck
    7  tricycle
    8  awning-tricycle
    9  bus
    10 motor
    11 others

This project retains only pedestrian (1) and people (2), and maps both
to a single YOLO class::

    0 = person

Annotation row fields (comma-separated)::

    bbox_left, bbox_top, bbox_width, bbox_height,
    score, object_category, truncation, occlusion
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Explicit VisDrone category IDs — do not silently assume IDs elsewhere.
VISDRONE_CATEGORY_NAMES: dict[int, str] = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

# Categories retained for person detection.
PERSON_SOURCE_CATEGORY_IDS: frozenset[int] = frozenset({1, 2})  # pedestrian, people

# YOLO / project output class.
YOLO_PERSON_CLASS_ID: int = 0
YOLO_PERSON_CLASS_NAME: str = "person"

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

# Canonical split names used in this project.
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
CANONICAL_SPLITS: tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)


@dataclass(frozen=True)
class VisDroneBox:
    """One VisDrone detection box after parsing (pixel coordinates)."""

    left: float
    top: float
    width: float
    height: float
    score: int
    category_id: int
    truncation: int
    occlusion: int

    @property
    def category_name(self) -> str:
        return VISDRONE_CATEGORY_NAMES.get(self.category_id, f"unknown_{self.category_id}")

    @property
    def is_person_source(self) -> bool:
        return self.category_id in PERSON_SOURCE_CATEGORY_IDS

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height


@dataclass(frozen=True)
class YoloBox:
    """Normalized YOLO detection box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_label_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


@dataclass(frozen=True)
class SplitPaths:
    """Resolved image and annotation directories for one VisDrone split."""

    canonical_name: str
    images_dir: Path
    annotations_dir: Path | None
    source_label: str


@dataclass
class ParseStats:
    """Counters collected while parsing one annotation file."""

    lines_total: int = 0
    malformed_lines: int = 0
    ignored_non_person: int = 0
    invalid_boxes: int = 0
    retained_person_boxes: int = 0


def is_valid_box(
    left: float,
    top: float,
    width: float,
    height: float,
    image_width: int | None = None,
    image_height: int | None = None,
) -> bool:
    """Return True if the box has positive size and (optionally) overlaps the image."""
    if width <= 0 or height <= 0:
        return False
    if left != left or top != top:  # NaN
        return False
    if image_width is not None and image_height is not None:
        if image_width <= 0 or image_height <= 0:
            return False
        if left >= image_width or top >= image_height:
            return False
        if left + width <= 0 or top + height <= 0:
            return False
    return True


def parse_annotation_line(
    line: str,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[VisDroneBox | None, str | None]:
    """Parse a single VisDrone annotation line.

    Returns:
        (box, None) on success,
        (None, "malformed") if the line cannot be parsed,
        (None, "invalid") if geometry is invalid,
        (None, "ignored") if the category is not pedestrian/people.
    """
    text = line.strip()
    if not text:
        return None, "malformed"

    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 8:
        return None, "malformed"

    try:
        left = float(parts[0])
        top = float(parts[1])
        width = float(parts[2])
        height = float(parts[3])
        score = int(float(parts[4]))
        category_id = int(float(parts[5]))
        truncation = int(float(parts[6]))
        occlusion = int(float(parts[7]))
    except ValueError:
        return None, "malformed"

    if not is_valid_box(left, top, width, height, image_width, image_height):
        return None, "invalid"

    box = VisDroneBox(
        left=left,
        top=top,
        width=width,
        height=height,
        score=score,
        category_id=category_id,
        truncation=truncation,
        occlusion=occlusion,
    )

    if not box.is_person_source:
        return None, "ignored"

    return box, None


def parse_annotation_file(
    path: Path,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[list[VisDroneBox], ParseStats]:
    """Parse a VisDrone ``.txt`` annotation file, keeping person-source boxes only."""
    stats = ParseStats()
    boxes: list[VisDroneBox] = []

    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.strip():
            continue
        stats.lines_total += 1
        box, reason = parse_annotation_line(
            line, image_width=image_width, image_height=image_height
        )
        if reason == "malformed":
            stats.malformed_lines += 1
            continue
        if reason == "invalid":
            stats.invalid_boxes += 1
            continue
        if reason == "ignored":
            stats.ignored_non_person += 1
            continue
        assert box is not None
        boxes.append(box)
        stats.retained_person_boxes += 1

    return boxes, stats


def visdrone_to_yolo(
    box: VisDroneBox,
    image_width: int,
    image_height: int,
    *,
    clip: bool = True,
) -> YoloBox | None:
    """Convert a VisDrone person-source box to a normalized YOLO box.

    Returns None if the box becomes invalid after clipping / normalization.
    """
    if image_width <= 0 or image_height <= 0:
        return None
    if not box.is_person_source:
        return None

    left, top, width, height = box.left, box.top, box.width, box.height
    if clip:
        right = min(left + width, float(image_width))
        bottom = min(top + height, float(image_height))
        left = max(left, 0.0)
        top = max(top, 0.0)
        width = right - left
        height = bottom - top

    if not is_valid_box(left, top, width, height, image_width, image_height):
        return None

    x_center = (left + width / 2.0) / image_width
    y_center = (top + height / 2.0) / image_height
    w_norm = width / image_width
    h_norm = height / image_height

    # Clamp numerical noise to [0, 1]
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    w_norm = min(max(w_norm, 0.0), 1.0)
    h_norm = min(max(h_norm, 0.0), 1.0)

    if w_norm <= 0.0 or h_norm <= 0.0:
        return None

    return YoloBox(
        class_id=YOLO_PERSON_CLASS_ID,
        x_center=x_center,
        y_center=y_center,
        width=w_norm,
        height=h_norm,
    )


def yolo_to_label_text(boxes: Iterable[YoloBox]) -> str:
    lines = [b.to_label_line() for b in boxes]
    return ("\n".join(lines) + "\n") if lines else ""


def _looks_like_images_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False


def _find_named_subdir(root: Path, names: tuple[str, ...]) -> Path | None:
    """Find a subdirectory whose name matches (case-insensitive) one of ``names``."""
    if not root.is_dir():
        return None
    lower_map = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for name in names:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def discover_split_dirs(source_root: Path) -> dict[str, SplitPaths]:
    """Detect common VisDrone2019-DET layouts under ``source_root``.

    Supported patterns (non-exhaustive)::

        source_root/
          VisDrone2019-DET-train/{images,annotations}/
          VisDrone2019-DET-val/{images,annotations}/
          VisDrone2019-DET-test-dev/{images,annotations?}/

        source_root/
          train/{images,annotations}/
          val/{images,annotations}/
          test-dev/ or test/{images,annotations?}/

    Returns a mapping of canonical split name -> SplitPaths.
    """
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"VisDrone source root does not exist or is not a directory: {source_root}\n"
            "Place the extracted VisDrone2019-DET folders under data/raw/visdrone/ "
            "(or pass --source)."
        )

    found: dict[str, SplitPaths] = {}

    # Pattern A: VisDrone2019-DET-* packages
    package_map = {
        SPLIT_TRAIN: ("VisDrone2019-DET-train", "VisDrone2019-DET-train"),
        SPLIT_VAL: ("VisDrone2019-DET-val", "VisDrone2019-DET-val"),
        SPLIT_TEST: (
            "VisDrone2019-DET-test-dev",
            "VisDrone2019-DET-test-challenge",
            "VisDrone2019-DET-test",
        ),
    }

    for canonical, package_names in package_map.items():
        for pkg_name in package_names:
            pkg = _find_named_subdir(source_root, (pkg_name,))
            if pkg is None:
                # Also allow package sitting one level deeper
                for child in source_root.iterdir() if source_root.is_dir() else []:
                    if child.is_dir():
                        pkg = _find_named_subdir(child, (pkg_name,))
                        if pkg is not None:
                            break
            if pkg is None:
                continue
            images = _find_named_subdir(pkg, ("images", "img", "image"))
            ann = _find_named_subdir(pkg, ("annotations", "labels", "annot"))
            if images is None or not _looks_like_images_dir(images):
                # Some dumps put images directly in the package root
                if _looks_like_images_dir(pkg):
                    images = pkg
                else:
                    continue
            found[canonical] = SplitPaths(
                canonical_name=canonical,
                images_dir=images,
                annotations_dir=ann if ann and ann.is_dir() else None,
                source_label=str(pkg.relative_to(source_root)),
            )
            break

    # Pattern B: flat train/val/test-dev under root (fill missing only)
    flat_map = {
        SPLIT_TRAIN: ("train",),
        SPLIT_VAL: ("val", "validation"),
        SPLIT_TEST: ("test-dev", "testdev", "test"),
    }
    for canonical, names in flat_map.items():
        if canonical in found:
            continue
        split_dir = _find_named_subdir(source_root, names)
        if split_dir is None:
            continue
        images = _find_named_subdir(split_dir, ("images", "img", "image"))
        ann = _find_named_subdir(split_dir, ("annotations", "labels", "annot"))
        if images is None:
            if _looks_like_images_dir(split_dir):
                images = split_dir
            else:
                continue
        if not _looks_like_images_dir(images):
            continue
        found[canonical] = SplitPaths(
            canonical_name=canonical,
            images_dir=images,
            annotations_dir=ann if ann and ann.is_dir() else None,
            source_label=str(split_dir.relative_to(source_root)),
        )

    if not found:
        raise FileNotFoundError(
            f"Could not detect VisDrone DET splits under: {source_root}\n"
            "Expected something like:\n"
            "  VisDrone2019-DET-train/images + annotations\n"
            "  VisDrone2019-DET-val/images + annotations\n"
            "  VisDrone2019-DET-test-dev/images (+ optional annotations)\n"
            "or train|val|test-dev folders with images/ and annotations/."
        )

    return found


def list_images(images_dir: Path) -> list[Path]:
    files = [
        p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return files


def annotation_path_for_image(image_path: Path, annotations_dir: Path | None) -> Path | None:
    if annotations_dir is None:
        return None
    candidate = annotations_dir / f"{image_path.stem}.txt"
    return candidate if candidate.is_file() else None


def validate_source_layout(source_root: Path) -> dict[str, SplitPaths]:
    """Discover splits and raise a clear error if train images are missing."""
    splits = discover_split_dirs(source_root)
    if SPLIT_TRAIN not in splits:
        raise FileNotFoundError(
            f"Train split not found under {source_root}. "
            "VisDrone train images are required for conversion."
        )
    train = splits[SPLIT_TRAIN]
    if train.annotations_dir is None:
        raise FileNotFoundError(
            f"Train annotations directory not found for {train.source_label}. "
            "Person-label conversion requires VisDrone ground-truth .txt files."
        )
    return splits
