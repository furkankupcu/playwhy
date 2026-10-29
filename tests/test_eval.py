"""Tests for the calibration error metric."""

import cv2
import numpy as np

from eval.calib_eval import HALF_TURN, ground_error_m

H_REF = cv2.getPerspectiveTransform(
    np.float32([[0, 0], [105, 0], [105, 68], [0, 68]]),
    np.float32([[100, 420], [760, 420], [600, 150], [250, 150]]),
)


def test_same_homography_has_zero_error():
    assert ground_error_m(H_REF, H_REF, (848, 478)).max() < 1e-6


def test_half_turn_counts_as_correct():
    assert ground_error_m(H_REF @ HALF_TURN, H_REF, (848, 478)).max() < 1e-6


def test_a_one_metre_shift_reads_as_one_metre():
    shift = np.array([[1, 0, -1.0], [0, 1, 0], [0, 0, 1]])  # pitch point x maps where x-1 used to
    e = ground_error_m(H_REF @ shift, H_REF, (848, 478))
    assert abs(float(np.median(e)) - 1.0) < 1e-6
