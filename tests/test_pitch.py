"""Tests for pitch template and drawing functions."""

import numpy as np
import pytest

from playwhy.pitch import (
    KEYPOINTS,
    LINES,
    LENGTH,
    WIDTH,
    draw_pitch,
    draw_topdown,
)


def test_pitch_dimensions():
    assert LENGTH == 105.0
    assert WIDTH == 68.0
    assert len(KEYPOINTS) >= 20
    assert len(LINES) >= 6


def test_pitch_keypoints_within_bounds():
    for name, (x, y) in KEYPOINTS.items():
        assert -1.0 <= x <= LENGTH + 1.0, f"Keypoint {name} x={x} out of bounds"
        assert -1.0 <= y <= WIDTH + 1.0, f"Keypoint {name} y={y} out of bounds"


def test_draw_topdown():
    img = draw_topdown(width_px=700, height_px=460)
    assert isinstance(img, np.ndarray)
    assert img.shape == (460, 700, 3)
    # Check that it is not all black
    assert img.mean() > 0


def test_draw_pitch_with_homography():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Simple affine/perspective matrix mapping (0..105, 0..68) to image coordinates
    src = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
    dst = np.float32([[100, 100], [1180, 100], [1180, 620], [100, 620]])
    import cv2
    H = cv2.getPerspectiveTransform(src, dst)

    drawn = draw_pitch(img, H)
    assert isinstance(drawn, np.ndarray)
    assert drawn.shape == (720, 1280, 3)
    assert drawn.max() > 0
