"""Tests for Step 2: Camera calibration."""

import cv2
import numpy as np
import pytest

from goal_explainer.calibrate import (
    manual_calibration,
    compute_reprojection_error,
    _white_line_mask,
    detect_lines_hough,
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


def test_white_line_mask():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Green background
    frame[:, :] = (34, 139, 34)
    # White horizontal line
    frame[48:52, :] = (255, 255, 255)

    mask = _white_line_mask(frame)
    assert mask.shape == (100, 100)
    assert mask[50, 50] > 0
    assert mask[10, 10] == 0


def test_reprojection_error_on_black_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Identity homography
    H = np.eye(3)
    err = compute_reprojection_error(frame, H)
    assert isinstance(err, float)
    assert err >= 0.0
