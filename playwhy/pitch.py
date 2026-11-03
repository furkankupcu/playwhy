"""
Standard football pitch template (105 × 68 metres, FIFA Laws of the Game).

All coordinates in metres, origin at the **bottom-left** corner of the pitch
(i.e. the corner where x = 0 and y = 0). The x-axis runs along the length
of the pitch (0 → 105) and the y-axis along the width (0 → 68).
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Pitch dimensions (metres) — FIFA Laws of the Game
# ---------------------------------------------------------------------------
LENGTH: float = 105.0
WIDTH: float = 68.0

PENALTY_AREA_LENGTH: float = 16.5
PENALTY_AREA_WIDTH: float = 40.32

GOAL_AREA_LENGTH: float = 5.5
GOAL_AREA_WIDTH: float = 18.32

PENALTY_SPOT: float = 11.0

CENTER_CIRCLE_RADIUS: float = 9.15

GOAL_WIDTH: float = 7.32

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
_HALF_WIDTH: float = WIDTH / 2.0
_HALF_LENGTH: float = LENGTH / 2.0
_PA_Y_OFFSET: float = (WIDTH - PENALTY_AREA_WIDTH) / 2.0
_GA_Y_OFFSET: float = (WIDTH - GOAL_AREA_WIDTH) / 2.0

# ---------------------------------------------------------------------------
# Named keypoints — dict[str, (x, y)] in metres
# ---------------------------------------------------------------------------
KEYPOINTS: dict[str, tuple[float, float]] = {
    # Corners
    "bottom_left": (0.0, 0.0),
    "bottom_right": (LENGTH, 0.0),
    "top_left": (0.0, WIDTH),
    "top_right": (LENGTH, WIDTH),
    # Halfway line intersections
    "halfway_bottom": (_HALF_LENGTH, 0.0),
    "halfway_top": (_HALF_LENGTH, WIDTH),
    # Center spot
    "center_spot": (_HALF_LENGTH, _HALF_WIDTH),
    # Center circle intersections
    "center_circle_top": (_HALF_LENGTH, _HALF_WIDTH + CENTER_CIRCLE_RADIUS),
    "center_circle_bottom": (_HALF_LENGTH, _HALF_WIDTH - CENTER_CIRCLE_RADIUS),
    "center_circle_left": (_HALF_LENGTH - CENTER_CIRCLE_RADIUS, _HALF_WIDTH),
    "center_circle_right": (_HALF_LENGTH + CENTER_CIRCLE_RADIUS, _HALF_WIDTH),
    # Goal line centres
    "left_goal_center": (0.0, _HALF_WIDTH),
    "right_goal_center": (LENGTH, _HALF_WIDTH),
    # Left penalty area
    "left_pa_bottom_left": (0.0, _PA_Y_OFFSET),
    "left_pa_bottom_right": (PENALTY_AREA_LENGTH, _PA_Y_OFFSET),
    "left_pa_top_right": (PENALTY_AREA_LENGTH, _PA_Y_OFFSET + PENALTY_AREA_WIDTH),
    "left_pa_top_left": (0.0, _PA_Y_OFFSET + PENALTY_AREA_WIDTH),
    # Right penalty area
    "right_pa_bottom_right": (LENGTH, _PA_Y_OFFSET),
    "right_pa_bottom_left": (LENGTH - PENALTY_AREA_LENGTH, _PA_Y_OFFSET),
    "right_pa_top_left": (LENGTH - PENALTY_AREA_LENGTH, _PA_Y_OFFSET + PENALTY_AREA_WIDTH),
    "right_pa_top_right": (LENGTH, _PA_Y_OFFSET + PENALTY_AREA_WIDTH),
    # Left goal area
    "left_ga_bottom_left": (0.0, _GA_Y_OFFSET),
    "left_ga_bottom_right": (GOAL_AREA_LENGTH, _GA_Y_OFFSET),
    "left_ga_top_right": (GOAL_AREA_LENGTH, _GA_Y_OFFSET + GOAL_AREA_WIDTH),
    "left_ga_top_left": (0.0, _GA_Y_OFFSET + GOAL_AREA_WIDTH),
    # Right goal area
    "right_ga_bottom_right": (LENGTH, _GA_Y_OFFSET),
    "right_ga_bottom_left": (LENGTH - GOAL_AREA_LENGTH, _GA_Y_OFFSET),
    "right_ga_top_left": (LENGTH - GOAL_AREA_LENGTH, _GA_Y_OFFSET + GOAL_AREA_WIDTH),
    "right_ga_top_right": (LENGTH, _GA_Y_OFFSET + GOAL_AREA_WIDTH),
    # Penalty spots
    "left_penalty_spot": (PENALTY_SPOT, _HALF_WIDTH),
    "right_penalty_spot": (LENGTH - PENALTY_SPOT, _HALF_WIDTH),
}

# Alias for keypoint dictionary
PITCH_KEYPOINTS: dict[str, tuple[float, float]] = KEYPOINTS

# ---------------------------------------------------------------------------
# Center circle approximation (72 points)
# ---------------------------------------------------------------------------
_CENTER_CIRCLE_PTS: list[tuple[float, float]] = [
    (
        _HALF_LENGTH + CENTER_CIRCLE_RADIUS * math.cos(2.0 * math.pi * i / 72),
        _HALF_WIDTH + CENTER_CIRCLE_RADIUS * math.sin(2.0 * math.pi * i / 72),
    )
    for i in range(73)
]

# ---------------------------------------------------------------------------
# Named polylines — list of (name, [(x, y), ...]) in metres
# ---------------------------------------------------------------------------
LINES: list[tuple[str, list[tuple[float, float]]]] = [
    # Touchlines
    ("bottom_touchline", [(0.0, 0.0), (LENGTH, 0.0)]),
    ("top_touchline", [(0.0, WIDTH), (LENGTH, WIDTH)]),
    # Goal lines
    ("left_goal_line", [(0.0, 0.0), (0.0, WIDTH)]),
    ("right_goal_line", [(LENGTH, 0.0), (LENGTH, WIDTH)]),
    # Halfway line
    ("halfway_line", [(_HALF_LENGTH, 0.0), (_HALF_LENGTH, WIDTH)]),
    # Center circle
    ("center_circle", _CENTER_CIRCLE_PTS),
    # Left penalty area
    (
        "left_penalty_area",
        [
            (0.0, _PA_Y_OFFSET),
            (PENALTY_AREA_LENGTH, _PA_Y_OFFSET),
            (PENALTY_AREA_LENGTH, _PA_Y_OFFSET + PENALTY_AREA_WIDTH),
            (0.0, _PA_Y_OFFSET + PENALTY_AREA_WIDTH),
        ],
    ),
    # Right penalty area
    (
        "right_penalty_area",
        [
            (LENGTH, _PA_Y_OFFSET),
            (LENGTH - PENALTY_AREA_LENGTH, _PA_Y_OFFSET),
            (LENGTH - PENALTY_AREA_LENGTH, _PA_Y_OFFSET + PENALTY_AREA_WIDTH),
            (LENGTH, _PA_Y_OFFSET + PENALTY_AREA_WIDTH),
        ],
    ),
    # Left goal area
    (
        "left_goal_area",
        [
            (0.0, _GA_Y_OFFSET),
            (GOAL_AREA_LENGTH, _GA_Y_OFFSET),
            (GOAL_AREA_LENGTH, _GA_Y_OFFSET + GOAL_AREA_WIDTH),
            (0.0, _GA_Y_OFFSET + GOAL_AREA_WIDTH),
        ],
    ),
    # Right goal area
    (
        "right_goal_area",
        [
            (LENGTH, _GA_Y_OFFSET),
            (LENGTH - GOAL_AREA_LENGTH, _GA_Y_OFFSET),
            (LENGTH - GOAL_AREA_LENGTH, _GA_Y_OFFSET + GOAL_AREA_WIDTH),
            (LENGTH, _GA_Y_OFFSET + GOAL_AREA_WIDTH),
        ],
    ),
]

# Flattened list of (x1, y1, x2, y2) line segments in pitch metres
PITCH_LINES: list[tuple[float, float, float, float]] = []
for _name, _pts in LINES:
    for _i in range(len(_pts) - 1):
        PITCH_LINES.append((_pts[_i][0], _pts[_i][1], _pts[_i + 1][0], _pts[_i + 1][1]))

# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def _project_points(
    points: list[tuple[float, float]],
    H: np.ndarray,
) -> np.ndarray:
    """Project pitch-coordinate points through homography *H*."""
    pts_arr = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts_arr, H)
    return np.round(projected).astype(np.int32)


def draw_pitch(
    img: np.ndarray,
    H: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Project standard pitch markings onto a video frame using homography *H*."""
    out = img.copy()
    for _name, pts in LINES:
        projected = _project_points(pts, H)
        cv2.polylines(out, [projected], isClosed=False, color=color, thickness=thickness)
    return out


def draw_topdown(
    width_px: int = 700,
    height_px: int = 460,
    bg_color: tuple[int, int, int] = (34, 139, 34),
    line_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Render a clean top-down pitch diagram as a BGR image."""
    margin = 10  # px
    img = np.full((height_px, width_px, 3), bg_color, dtype=np.uint8)

    sx = (width_px - 2 * margin) / LENGTH
    sy = (height_px - 2 * margin) / WIDTH

    def _to_px(x: float, y: float) -> tuple[int, int]:
        px = int(round(x * sx + margin))
        py = int(round((WIDTH - y) * sy + margin))
        return (px, py)

    def _pts_to_px(points: list[tuple[float, float]]) -> np.ndarray:
        return np.array([_to_px(x, y) for x, y in points], dtype=np.int32)

    thickness = 2

    # Draw all named lines
    for _name, pts in LINES:
        px_pts = _pts_to_px(pts)
        cv2.polylines(img, [px_pts], isClosed=False, color=line_color, thickness=thickness)

    # Penalty spots (small filled circles)
    for spot_key in ("left_penalty_spot", "right_penalty_spot", "center_spot"):
        sx_coord, sy_coord = KEYPOINTS[spot_key]
        cv2.circle(img, _to_px(sx_coord, sy_coord), 4, line_color, -1)

    return img
