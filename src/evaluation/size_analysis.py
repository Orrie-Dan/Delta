"""Project-specific object-size analysis for person detection.

These size bins are **not** COCO area thresholds. They use *normalized*
bounding-box area (width_norm * height_norm) in the YOLO label space.

Default bins (configurable)::

    small  : area < SMALL_AREA_THRESHOLD          (default 0.001)
    medium : SMALL <= area < MEDIUM_AREA_THRESHOLD (default 0.01)
    large  : area >= MEDIUM_AREA_THRESHOLD

Matching algorithm for size-bin recall
--------------------------------------
For each image:

1. Collect GT boxes and prediction boxes (xywh normalized).
2. Compute pairwise IoU for all GT–pred pairs.
3. Greedy one-to-one matching: repeatedly take the highest-IoU pair
   with IoU >= ``iou_threshold`` (default 0.5), remove that GT and
   prediction from further consideration.
4. A GT box is a true positive for recall if it was matched.
5. Aggregate TP / GT counts **within each size bin** (bin assigned from
   GT box area only).

This reports **recall @ IoU**, not mAP. Do not treat it as size-specific mAP.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum


class SizeBin(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True)
class Box:
    """Normalized YOLO-style box: center-x, center-y, width, height in [0, 1]."""

    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return box_area_normalized(self.width, self.height)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        x1 = self.x_center - self.width / 2.0
        y1 = self.y_center - self.height / 2.0
        x2 = self.x_center + self.width / 2.0
        y2 = self.y_center + self.height / 2.0
        return x1, y1, x2, y2


def box_area_normalized(width: float, height: float) -> float:
    """Return normalized area ``w * h`` (clamped at >= 0)."""
    return max(0.0, float(width)) * max(0.0, float(height))


def assign_size_bin(
    area: float,
    *,
    small_area_threshold: float = 0.001,
    medium_area_threshold: float = 0.01,
) -> SizeBin:
    """Assign a project-specific normalized-area size bin."""
    if small_area_threshold <= 0 or medium_area_threshold <= small_area_threshold:
        raise ValueError(
            "Require 0 < small_area_threshold < medium_area_threshold "
            f"(got small={small_area_threshold}, medium={medium_area_threshold})"
        )
    if area < small_area_threshold:
        return SizeBin.SMALL
    if area < medium_area_threshold:
        return SizeBin.MEDIUM
    return SizeBin.LARGE


def iou_xywhn(a: Box, b: Box) -> float:
    """Intersection-over-union for two normalized xywh boxes."""
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def greedy_match_predictions_to_gt(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    *,
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one GT↔prediction matches by descending IoU.

    Returns a list of ``(gt_index, pred_index, iou)`` for matches that
    meet ``iou_threshold``. Each GT and each prediction is used at most once.
    """
    if not gt_boxes or not pred_boxes:
        return []

    pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            score = iou_xywhn(gt, pred)
            if score >= iou_threshold:
                pairs.append((score, gi, pi))

    pairs.sort(key=lambda t: t[0], reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, score))
    return matches


def compute_gt_size_distribution(
    gt_boxes: Iterable[Box],
    *,
    small_area_threshold: float = 0.001,
    medium_area_threshold: float = 0.01,
) -> dict[str, float | int]:
    """Count GT boxes per size bin and return counts + percentages."""
    counts = {SizeBin.SMALL.value: 0, SizeBin.MEDIUM.value: 0, SizeBin.LARGE.value: 0}
    total = 0
    for box in gt_boxes:
        total += 1
        bin_name = assign_size_bin(
            box.area,
            small_area_threshold=small_area_threshold,
            medium_area_threshold=medium_area_threshold,
        ).value
        counts[bin_name] += 1

    result: dict[str, float | int] = {
        "total": total,
        "small_count": counts[SizeBin.SMALL.value],
        "medium_count": counts[SizeBin.MEDIUM.value],
        "large_count": counts[SizeBin.LARGE.value],
        "small_pct": (100.0 * counts[SizeBin.SMALL.value] / total) if total else 0.0,
        "medium_pct": (100.0 * counts[SizeBin.MEDIUM.value] / total) if total else 0.0,
        "large_pct": (100.0 * counts[SizeBin.LARGE.value] / total) if total else 0.0,
        "small_area_threshold": small_area_threshold,
        "medium_area_threshold": medium_area_threshold,
        "note": "Project-specific normalized-area bins (not COCO thresholds).",
    }
    return result


def compute_size_bin_recall(
    image_gt_pred: Sequence[tuple[Sequence[Box], Sequence[Box]]],
    *,
    iou_threshold: float = 0.5,
    small_area_threshold: float = 0.001,
    medium_area_threshold: float = 0.01,
) -> dict[str, dict[str, float | int]]:
    """Compute per-bin recall @ IoU using greedy one-to-one matching.

    Parameters
    ----------
    image_gt_pred:
        Iterable of ``(gt_boxes, pred_boxes)`` per image.
    """
    stats = {
        SizeBin.SMALL.value: {"tp": 0, "gt": 0},
        SizeBin.MEDIUM.value: {"tp": 0, "gt": 0},
        SizeBin.LARGE.value: {"tp": 0, "gt": 0},
    }

    for gt_boxes, pred_boxes in image_gt_pred:
        gt_list = list(gt_boxes)
        pred_list = list(pred_boxes)
        matches = greedy_match_predictions_to_gt(
            gt_list, pred_list, iou_threshold=iou_threshold
        )
        matched_gt = {gi for gi, _pi, _iou in matches}

        for gi, gt in enumerate(gt_list):
            bin_name = assign_size_bin(
                gt.area,
                small_area_threshold=small_area_threshold,
                medium_area_threshold=medium_area_threshold,
            ).value
            stats[bin_name]["gt"] += 1
            if gi in matched_gt:
                stats[bin_name]["tp"] += 1

    output: dict[str, dict[str, float | int]] = {}
    for bin_name, values in stats.items():
        gt = int(values["gt"])
        tp = int(values["tp"])
        recall = (tp / gt) if gt else 0.0
        output[bin_name] = {
            "tp": tp,
            "gt": gt,
            "recall": recall,
            "iou_threshold": iou_threshold,
        }
    output["meta"] = {
        "metric": "recall_at_iou_greedy_one_to_one",
        "not_map": 1,
        "small_area_threshold": small_area_threshold,
        "medium_area_threshold": medium_area_threshold,
        "note": (
            "Size bins are project-specific normalized areas; "
            "this is recall@IoU, not size-specific mAP."
        ),
    }
    return output
