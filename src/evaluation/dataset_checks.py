"""Dataset sanity checks for YOLO person datasets (MVP-02)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.visdrone import IMAGE_EXTENSIONS


@dataclass
class SplitCounts:
    split: str
    num_images: int
    num_labels: int


@dataclass
class DatasetSanityReport:
    dataset_root: str
    splits: dict[str, SplitCounts]
    label_class_ids: list[int]
    invalid_coordinate_lines: int
    sample_labels_checked: int

    def to_dict(self) -> dict:
        return {
            "dataset_root": self.dataset_root,
            "splits": {k: asdict(v) for k, v in self.splits.items()},
            "label_class_ids": self.label_class_ids,
            "invalid_coordinate_lines": self.invalid_coordinate_lines,
            "sample_labels_checked": self.sample_labels_checked,
        }


def count_split(dataset_root: Path, split: str) -> SplitCounts:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    n_img = 0
    n_lbl = 0
    if images_dir.is_dir():
        n_img = sum(
            1
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if labels_dir.is_dir():
        n_lbl = sum(1 for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    return SplitCounts(split=split, num_images=n_img, num_labels=n_lbl)


def validate_person_yolo_dataset(
    dataset_root: Path,
    *,
    required_splits: tuple[str, ...] = ("train", "val"),
    sample_labels: int = 50,
    allowed_class_ids: frozenset[int] = frozenset({0}),
) -> DatasetSanityReport:
    """Validate YOLO person dataset structure and sample label content.

    Raises:
        FileNotFoundError / ValueError with actionable messages on failure.
    """
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            "Upload visdrone_person.zip to Drive and extract it (notebook 02) to\n"
            "  /content/datasets/visdrone_person/\n"
            "Expected layout: {{images,labels}}/{{train,val}}/"
        )

    splits: dict[str, SplitCounts] = {}
    for split in required_splits:
        counts = count_split(dataset_root, split)
        splits[split] = counts
        if counts.num_images <= 0:
            raise FileNotFoundError(
                f"No images found for split '{split}' under {dataset_root / 'images' / split}"
            )
        if counts.num_labels <= 0:
            raise FileNotFoundError(
                f"No labels found for split '{split}' under {dataset_root / 'labels' / split}"
            )

    labels_dir = dataset_root / "labels" / "train"
    label_files = sorted(p for p in labels_dir.iterdir() if p.suffix == ".txt")
    class_ids: set[int] = set()
    invalid_coords = 0
    checked = 0
    for path in label_files[: max(0, sample_labels)]:
        checked += 1
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                invalid_coords += 1
                continue
            try:
                cid = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"Malformed label in {path.name}: {line!r}") from exc
            class_ids.add(cid)
            if any(v < 0.0 or v > 1.0 for v in vals):
                invalid_coords += 1

    unexpected = class_ids - set(allowed_class_ids)
    if unexpected:
        raise ValueError(
            f"Found unexpected class IDs {sorted(unexpected)} in sampled labels. "
            f"Person-only dataset must use only {sorted(allowed_class_ids)}."
        )

    return DatasetSanityReport(
        dataset_root=str(dataset_root),
        splits=splits,
        label_class_ids=sorted(class_ids),
        invalid_coordinate_lines=invalid_coords,
        sample_labels_checked=checked,
    )
