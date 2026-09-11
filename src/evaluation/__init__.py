"""Evaluation helpers for person-detection experiments (MVP-02+)."""

from src.evaluation.size_analysis import (
    Box,
    SizeBin,
    assign_size_bin,
    box_area_normalized,
    compute_gt_size_distribution,
    compute_size_bin_recall,
    greedy_match_predictions_to_gt,
    iou_xywhn,
)

__all__ = [
    "Box",
    "SizeBin",
    "assign_size_bin",
    "box_area_normalized",
    "compute_gt_size_distribution",
    "compute_size_bin_recall",
    "greedy_match_predictions_to_gt",
    "iou_xywhn",
]
