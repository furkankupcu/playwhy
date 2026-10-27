"""Tests for GSR evaluation (Hungarian matching and metrics)."""

import pytest

from eval.gsr_eval import match_players, evaluate_frame


def test_match_players_exact():
    preds = [(10.0, 20.0), (30.0, 40.0)]
    gt = [(10.0, 20.0), (30.0, 40.0)]
    matches, recall = match_players(preds, gt, threshold=5.0)
    assert len(matches) == 2
    assert recall == 1.0
    for _, _, dist in matches:
        assert dist == pytest.approx(0.0)


def test_match_players_with_threshold():
    # One within 2m, one 8m away
    preds = [(10.0, 21.0), (50.0, 50.0)]
    gt = [(10.0, 20.0), (30.0, 40.0)]
    matches, recall = match_players(preds, gt, threshold=5.0)
    assert len(matches) == 1
    assert matches[0][0] == 0  # first pred
    assert matches[0][1] == 0  # first gt
    assert matches[0][2] == pytest.approx(1.0)
    assert recall == 0.5


def test_evaluate_frame():
    pred = [
        {"x_m": 12.0, "y_m": 22.0},
        {"x_m": 50.0, "y_m": 50.0},
    ]
    gt = [
        ("p1", 10.0, 20.0),
        ("p2", 80.0, 80.0),
    ]
    result = evaluate_frame(pred, gt, threshold=5.0)
    assert result["total_gt"] == 2
    assert result["total_pred"] == 2
    assert result["matched"] == 1
    assert result["recall"] == 0.5
    assert result["mean_error"] == pytest.approx((2.0**2 + 2.0**2)**0.5)
