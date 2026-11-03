"""Tests for Step 2: Camera calibration."""

import cv2
import numpy as np
import pytest

from playwhy.calibrate import (
    manual_calibration,
    REGION_TEMPLATES,
    _render,
    fit_regions,
)


def test_manual_calibration_exact():
    # Construct exact 4 points
    src_pitch = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]
    dst_pixel = [(100.0, 100.0), (1100.0, 100.0), (1100.0, 600.0), (100.0, 600.0)]
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    H = manual_calibration(dummy_frame, dst_pixel, src_pitch)
    assert H.shape == (3, 3)

    # Test projection of points
    pts_in = np.array(src_pitch, dtype=np.float64).reshape(-1, 1, 2)
    pts_out = cv2.perspectiveTransform(pts_in, H).reshape(-1, 2)
    for p_out, p_exp in zip(pts_out, dst_pixel):
        assert pytest.approx(p_out[0], abs=1e-2) == p_exp[0]
        assert pytest.approx(p_out[1], abs=1e-2) == p_exp[1]


def test_manual_calibration_insufficient_points():
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="≥ 4 point pairs"):
        manual_calibration(dummy_frame, [(0, 0), (1, 1)], [(0, 0), (1, 1)])


def test_fit_regions_recovers_known_homography():
    # Regions rendered from a known camera; the fit must find that camera again.
    # Camera in the stands: whole pitch in front of it, far side narrower.
    true_H = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [105, 0], [105, 68], [0, 68]]),
        np.float32([[100, 420], [760, 420], [600, 150], [250, 150]]),
    )
    shape = (478, 848)
    regions = {name: _render(true_H, polys, shape) for name, polys in REGION_TEMPLATES.items()}
    H, cost = fit_regions(regions)
    assert cost < 0.05
    probe = np.array([[[96.75, 34.0]], [[52.5, 34.0]], [[94.0, 34.0]]], np.float64)
    got = cv2.perspectiveTransform(probe, H)
    want = cv2.perspectiveTransform(probe, true_H)
    # The pitch is symmetric under a half turn, so that fit is equally right.
    half_turn = np.array([[-1, 0, 105], [0, -1, 68], [0, 0, 1]], np.float64)
    want_turned = cv2.perspectiveTransform(probe, true_H @ half_turn)
    assert min(np.abs(got - want).max(), np.abs(got - want_turned).max()) < 6.0  # px


def test_fit_regions_needs_box_or_field():
    assert fit_regions({"center circle": np.ones((10, 10), np.uint8)}) is None


def test_goals_break_quarter_turn_ambiguity():
    # Like a frame where fans hide the penalty box: field + circle alone also fit a
    # 90°-turned pitch; the goal mask must pull the fit back.
    true_H = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [105, 0], [105, 68], [0, 68]]),
        np.float32([[100, 420], [760, 420], [600, 150], [250, 150]]),
    )
    shape = (478, 848)
    regions = {n: _render(true_H, REGION_TEMPLATES[n], shape) for n in ("soccer field", "center circle")}
    goal = np.zeros(shape, np.uint8)
    base = cv2.perspectiveTransform(np.array([[[105.0, 34.0]]]), true_H)[0, 0]
    cv2.rectangle(goal, (int(base[0]) - 15, int(base[1]) - 20), (int(base[0]) + 15, int(base[1])), 1, -1)
    regions["soccer goal"] = goal
    H, _ = fit_regions(regions)
    probe = np.array([[[105.0, 34.0]], [[52.5, 0.0]]], np.float64)
    got = cv2.perspectiveTransform(probe, H)
    half_turn = np.array([[-1, 0, 105], [0, -1, 68], [0, 0, 1]], np.float64)
    errs = [np.abs(got - cv2.perspectiveTransform(probe, G)).max() for G in (true_H, true_H @ half_turn)]
    assert min(errs) < 15.0  # px


def test_fit_cost_is_plain_float_for_json():
    import json
    true_H = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [105, 0], [105, 68], [0, 68]]),
        np.float32([[100, 420], [760, 420], [600, 150], [250, 150]]),
    )
    regions = {n: _render(true_H, p, (478, 848)) for n, p in REGION_TEMPLATES.items()}
    _, cost = fit_regions(regions)
    assert type(cost) is float
    json.dumps({"error": cost, "reliable": bool(cost <= 1.0)})
