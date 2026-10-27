"""Tests for top-down replay rendering."""

import numpy as np
import pytest

from goal_explainer.replay import (
    pitch_to_pixel,
    draw_player,
    draw_ball,
    draw_pass_arrow,
    draw_movement_trail,
    render_frame,
    _build_position_index,
)


def test_pitch_to_pixel():
    # Bottom-left (0, 0)
    px, py = pitch_to_pixel(0.0, 0.0, img_w=700, img_h=460, margin=30)
    assert px == 30
    assert py == 30

    # Top-right (105, 68)
    px, py = pitch_to_pixel(105.0, 68.0, img_w=700, img_h=460, margin=30)
    assert px == 670
    assert py == 430


def test_draw_player():
    img = np.zeros((460, 700, 3), dtype=np.uint8)
    draw_player(img, 100, 100, team="A", player_id=10, has_ball=True)
    # Circle should be drawn around (100, 100)
    assert img[100, 100].max() > 0


def test_draw_ball():
    img = np.zeros((460, 700, 3), dtype=np.uint8)
    draw_ball(img, 200, 200)
    assert (img[200, 200] == (255, 255, 255)).all()


def test_draw_pass_arrow():
    img = np.zeros((460, 700, 3), dtype=np.uint8)
    draw_pass_arrow(img, (50, 50), (150, 50))
    assert img.max() > 0


def test_render_frame():
    pitch_data = [
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "players": [
                {"player_id": 7, "team": "A", "x_m": 50.0, "y_m": 30.0},
                {"player_id": 4, "team": "B", "x_m": 60.0, "y_m": 32.0},
            ],
            "ball": {"x_m": 50.0, "y_m": 30.0},
        }
    ]
    all_positions = _build_position_index(pitch_data)
    rendered = render_frame(pitch_data, frame_idx=0, events={}, all_positions=all_positions)
    assert isinstance(rendered, np.ndarray)
    assert rendered.shape == (460, 700, 3)
