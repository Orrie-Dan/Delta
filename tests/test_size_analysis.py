"""Unit tests for size-bin analysis and IoU matching (no GPU / no YOLO train)."""

from __future__ import annotations

import pytest

from src.evaluation.metrics_io import build_metrics_template, load_json, save_json
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


def test_box_area_normalized():
    assert box_area_normalized(0.1, 0.2) == pytest.approx(0.02)
    assert box_area_normalized(-1, 0.5) == 0.0


def test_assign_size_bin_boundaries():
    assert assign_size_bin(0.0005) == SizeBin.SMALL
    assert assign_size_bin(0.001) == SizeBin.MEDIUM
    assert assign_size_bin(0.0099) == SizeBin.MEDIUM
    assert assign_size_bin(0.01) == SizeBin.LARGE
    assert assign_size_bin(0.5) == SizeBin.LARGE


def test_assign_size_bin_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        assign_size_bin(0.1, small_area_threshold=0.01, medium_area_threshold=0.001)


def test_iou_identical_boxes():
    a = Box(0.5, 0.5, 0.2, 0.2)
    assert iou_xywhn(a, a) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    a = Box(0.2, 0.2, 0.1, 0.1)
    b = Box(0.8, 0.8, 0.1, 0.1)
    assert iou_xywhn(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    a = Box(0.5, 0.5, 0.4, 0.4)
    b = Box(0.6, 0.5, 0.4, 0.4)
    score = iou_xywhn(a, b)
    assert 0.0 < score < 1.0


def test_greedy_one_to_one_matching():
    gt = [Box(0.2, 0.2, 0.1, 0.1), Box(0.8, 0.8, 0.1, 0.1)]
    pred = [Box(0.21, 0.21, 0.1, 0.1), Box(0.5, 0.5, 0.1, 0.1)]
    matches = greedy_match_predictions_to_gt(gt, pred, iou_threshold=0.5)
    assert len(matches) == 1
    assert matches[0][0] == 0  # first GT matched
    assert matches[0][1] == 0


def test_greedy_does_not_double_assign():
    gt = [Box(0.5, 0.5, 0.2, 0.2)]
    pred = [Box(0.5, 0.5, 0.2, 0.2), Box(0.51, 0.5, 0.2, 0.2)]
    matches = greedy_match_predictions_to_gt(gt, pred, iou_threshold=0.5)
    assert len(matches) == 1


def test_gt_size_distribution():
    boxes = [
        Box(0.5, 0.5, 0.02, 0.02),  # 0.0004 small
        Box(0.5, 0.5, 0.05, 0.05),  # 0.0025 medium
        Box(0.5, 0.5, 0.2, 0.2),  # 0.04 large
        Box(0.5, 0.5, 0.01, 0.01),  # 0.0001 small
    ]
    dist = compute_gt_size_distribution(boxes)
    assert dist["small_count"] == 2
    assert dist["medium_count"] == 1
    assert dist["large_count"] == 1
    assert dist["total"] == 4
    assert dist["small_pct"] == pytest.approx(50.0)


def test_size_bin_recall():
    # Image 1: small GT matched; large GT missed
    gt1 = [Box(0.2, 0.2, 0.02, 0.02), Box(0.7, 0.7, 0.2, 0.2)]
    pred1 = [Box(0.2, 0.2, 0.02, 0.02)]
    # Image 2: medium GT matched
    gt2 = [Box(0.5, 0.5, 0.05, 0.05)]
    pred2 = [Box(0.5, 0.5, 0.05, 0.05)]

    result = compute_size_bin_recall([(gt1, pred1), (gt2, pred2)], iou_threshold=0.5)
    assert result["small"]["gt"] == 1
    assert result["small"]["tp"] == 1
    assert result["small"]["recall"] == pytest.approx(1.0)
    assert result["medium"]["gt"] == 1
    assert result["medium"]["recall"] == pytest.approx(1.0)
    assert result["large"]["gt"] == 1
    assert result["large"]["tp"] == 0
    assert result["large"]["recall"] == pytest.approx(0.0)
    assert result["meta"]["not_map"] == 1


def test_metrics_json_roundtrip(tmp_path):
    data = build_metrics_template(
        experiment_name="mvp02_test",
        model="yolo26n.pt",
        image_size=640,
        epochs_requested=50,
    )
    assert data["map50"] is None
    path = save_json(data, tmp_path / "metrics.json")
    loaded = load_json(path)
    assert loaded["experiment_name"] == "mvp02_test"
    assert loaded["model"] == "yolo26n.pt"
