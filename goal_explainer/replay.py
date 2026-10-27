"""
Generate a bird's-eye replay video showing player positions on a pitch diagram.

Reads tracking data (pitch.json) and event data (events.json) from the run
directory, then renders each frame as a top-down pitch view with annotated
player positions, ball location, movement trails, pass arrows, and offside
lines.  The final output is an MP4 video saved to ``runs/<clip_id>/replay.mp4``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import pitch as pitch_module

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PITCH_LENGTH_M: float = 105.0  # FIFA standard pitch length (metres)
PITCH_WIDTH_M: float = 68.0    # FIFA standard pitch width (metres)

TEAM_COLORS: dict[str, tuple[int, int, int]] = {
    "A": (255, 120, 50),   # blue  (BGR)
    "B": (60, 60, 220),    # red   (BGR)
    "goalkeeper": (0, 220, 220),  # yellow (BGR)
    "referee": (160, 160, 160),   # gray   (BGR)
}

TRAIL_LENGTH: int = 15  # number of past frames for movement trails


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def pitch_to_pixel(
    x_m: float,
    y_m: float,
    img_w: int = 700,
    img_h: int = 460,
    margin: int = 30,
) -> tuple[int, int]:
    """Convert pitch coordinates in metres to pixel coordinates on the image.

    Parameters
    ----------
    x_m : float
        Horizontal position on the pitch in metres (0 = left goal-line).
    y_m : float
        Vertical position on the pitch in metres (0 = top touch-line).
    img_w : int
        Width of the output image in pixels.
    img_h : int
        Height of the output image in pixels.
    margin : int
        Pixel margin around the pitch drawing area.

    Returns
    -------
    tuple[int, int]
        ``(px_x, px_y)`` pixel coordinates.
    """
    draw_w = img_w - 2 * margin
    draw_h = img_h - 2 * margin

    px_x = int(margin + (x_m / PITCH_LENGTH_M) * draw_w)
    px_y = int(margin + (y_m / PITCH_WIDTH_M) * draw_h)
    return px_x, px_y


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def draw_player(
    img: np.ndarray,
    x: int,
    y: int,
    team: str,
    player_id: int,
    has_ball: bool = False,
    is_goalkeeper: bool = False,
) -> None:
    """Draw a single player dot on the pitch image.

    Parameters
    ----------
    img : np.ndarray
        BGR image to draw on (modified in-place).
    x, y : int
        Pixel coordinates of the player.
    team : str
        ``"A"``, ``"B"``, or ``"referee"``.
    player_id : int
        Jersey / track ID shown as a label.
    has_ball : bool
        If *True*, draw a larger highlight ring around the player.
    is_goalkeeper : bool
        If *True*, use the goalkeeper colour instead of the team colour.
    """
    if is_goalkeeper:
        color = TEAM_COLORS["goalkeeper"]
    else:
        color = TEAM_COLORS.get(team, TEAM_COLORS["referee"])

    radius = 8
    cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)

    if has_ball:
        cv2.circle(img, (x, y), radius + 4, (0, 255, 255), 2, cv2.LINE_AA)

    # Player ID label above the dot
    label = str(player_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.putText(
        img,
        label,
        (x - tw // 2, y - radius - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_ball(img: np.ndarray, x: int, y: int) -> None:
    """Draw the ball as a small white circle with a dark outline.

    Parameters
    ----------
    img : np.ndarray
        BGR image (modified in-place).
    x, y : int
        Pixel coordinates of the ball.
    """
    cv2.circle(img, (x, y), 5, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_pass_arrow(
    img: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 255),
) -> None:
    """Draw an arrowed line representing a pass.

    Parameters
    ----------
    img : np.ndarray
        BGR image (modified in-place).
    start : tuple[int, int]
        Starting pixel coordinates ``(x, y)``.
    end : tuple[int, int]
        Ending pixel coordinates ``(x, y)``.
    color : tuple[int, int, int]
        BGR colour of the arrow.
    """
    cv2.arrowedLine(img, start, end, color, 2, cv2.LINE_AA, tipLength=0.15)


def draw_movement_trail(
    img: np.ndarray,
    positions: list[tuple[int, int]],
    team: str,
    alpha: float = 0.3,
) -> None:
    """Draw a fading trail of recent positions for a player.

    Uses the last :data:`TRAIL_LENGTH` positions.  Earlier points are drawn
    with lower opacity (simulated via colour blending with the background).

    Parameters
    ----------
    img : np.ndarray
        BGR image (modified in-place).
    positions : list[tuple[int, int]]
        Chronological list of ``(x, y)`` pixel positions.
    team : str
        Team identifier for colouring.
    alpha : float
        Base opacity multiplier for the trail dots.
    """
    trail = positions[-TRAIL_LENGTH:]
    if len(trail) < 2:
        return

    base_color = np.array(TEAM_COLORS.get(team, TEAM_COLORS["referee"]), dtype=np.float64)
    n = len(trail)

    for i in range(n - 1):
        # Fade: earliest point is most transparent
        fade = alpha * ((i + 1) / n)
        color = tuple(int(c * fade) for c in base_color)
        cv2.circle(img, trail[i], 3, color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Frame renderer
# ---------------------------------------------------------------------------

def _draw_offside_line(
    img: np.ndarray,
    offside_x_m: float,
    img_w: int = 700,
    img_h: int = 460,
    margin: int = 30,
) -> None:
    """Draw a vertical offside line across the pitch."""
    px_x_top, px_y_top = pitch_to_pixel(offside_x_m, 0.0, img_w, img_h, margin)
    px_x_bot, px_y_bot = pitch_to_pixel(offside_x_m, PITCH_WIDTH_M, img_w, img_h, margin)
    cv2.line(img, (px_x_top, px_y_top), (px_x_bot, px_y_bot), (0, 0, 255), 1, cv2.LINE_AA)


def render_frame(
    pitch_data: dict[str, Any],
    frame_idx: int,
    events: dict[str, Any],
    all_positions: dict[int, dict[int, dict[str, Any]]],
) -> np.ndarray:
    """Render a single top-down frame of the replay.

    Parameters
    ----------
    pitch_data : dict
        Pitch/tracking payload loaded from ``pitch.json``.  Expected to
        contain ``"frames"`` — a list of per-frame records with ``"players"``
        and ``"ball"`` sub-keys.
    frame_idx : int
        Zero-based index of the current frame.
    events : dict
        Events payload loaded from ``events.json``.  May contain
        ``"passes"``, ``"offsides"``, etc., keyed by frame index (as string).
    all_positions : dict[int, dict[int, dict[str, Any]]]
        Nested mapping ``player_id -> frame_idx -> {x_m, y_m, team, ...}``
        used for drawing movement trails.

    Returns
    -------
    np.ndarray
        BGR image of the rendered frame.
    """
    img_w, img_h = 700, 460

    # Start with a clean pitch image
    img = pitch_module.draw_topdown(img_w, img_h)

    frames = _get_frames(pitch_data)
    if frame_idx >= len(frames):
        return img

    frame = frames[frame_idx]

    # --- Movement trails ------------------------------------------------
    for pid, pos_history in all_positions.items():
        # Collect pixel positions up to this frame
        trail: list[tuple[int, int]] = []
        team_label = "A"
        for fi in range(max(0, frame_idx - TRAIL_LENGTH), frame_idx + 1):
            if fi in pos_history:
                rec = pos_history[fi]
                px = pitch_to_pixel(rec["x_m"], rec["y_m"], img_w, img_h)
                trail.append(px)
                team_label = rec.get("team", "A")
        if trail:
            draw_movement_trail(img, trail, team_label)

    # --- Players --------------------------------------------------------
    players = frame.get("players", frame.get("positions", []))
    ball_holder_id: int | None = frame.get("ball_holder_id")

    for p in players:
        px, py = pitch_to_pixel(p.get("x_m", p.get("x", 0.0)), p.get("y_m", p.get("y", 0.0)), img_w, img_h)
        pid = p.get("player_id", p.get("id", 0))
        draw_player(
            img,
            px,
            py,
            team=p.get("team", "A"),
            player_id=pid,
            has_ball=(pid == ball_holder_id),
            is_goalkeeper=p.get("is_goalkeeper", False),
        )

    # --- Ball -----------------------------------------------------------
    ball = frame.get("ball")
    if ball and "x_m" in ball and "y_m" in ball:
        bx, by = pitch_to_pixel(ball["x_m"], ball["y_m"], img_w, img_h)
        draw_ball(img, bx, by)

    # --- Passes at this frame -------------------------------------------
    frame_key = str(frame_idx)
    passes = events.get("passes", {})
    if frame_key in passes:
        ev = passes[frame_key]
        start_px = pitch_to_pixel(ev["from_x_m"], ev["from_y_m"], img_w, img_h)
        end_px = pitch_to_pixel(ev["to_x_m"], ev["to_y_m"], img_w, img_h)
        draw_pass_arrow(img, start_px, end_px)

    # --- Offside line ---------------------------------------------------
    offsides = events.get("offsides", {})
    if frame_key in offsides:
        _draw_offside_line(img, offsides[frame_key]["x_m"], img_w, img_h)

    # --- Timestamp overlay ----------------------------------------------
    ts = frame.get("timestamp", frame_idx)
    if isinstance(ts, (int, float)):
        ts_text = f"{ts:.2f}s"
    else:
        ts_text = str(ts)
    cv2.putText(
        img,
        ts_text,
        (img_w - 110, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return img


def _get_frames(pitch_data: Any) -> list[dict[str, Any]]:
    if isinstance(pitch_data, list):
        return pitch_data
    if isinstance(pitch_data, dict):
        return pitch_data.get("frames", [])
    return []


# ---------------------------------------------------------------------------
# Build trail index from pitch data
# ---------------------------------------------------------------------------

def _build_position_index(
    pitch_data: Any,
) -> dict[int, dict[int, dict[str, Any]]]:
    """Build a per-player, per-frame position index for trail rendering.

    Returns
    -------
    dict[int, dict[int, dict]]
        ``{player_id: {frame_idx: {x_m, y_m, team, ...}}}``
    """
    index: dict[int, dict[int, dict[str, Any]]] = {}
    for fi, frame in enumerate(_get_frames(pitch_data)):
        for p in frame.get("players", frame.get("positions", [])):
            pid = p.get("player_id", p.get("id", 0))
            if pid not in index:
                index[pid] = {}
            index[pid][fi] = {
                "x_m": p.get("x_m", p.get("x", 0.0)),
                "y_m": p.get("y_m", p.get("y", 0.0)),
                "team": p.get("team", "A"),
            }
    return index


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(clip_id: str, fps: float = 25.0) -> Path:
    """Generate the top-down replay video for a processed clip.

    Parameters
    ----------
    clip_id : str
        Identifier for the clip (maps to ``runs/<clip_id>/``).
    fps : float
        Frame-rate of the output video.

    Returns
    -------
    Path
        Absolute path to the generated ``replay.mp4``.
    """
    run_dir = Path("runs") / clip_id

    # Load data ----------------------------------------------------------
    pitch_path = run_dir / "pitch.json"
    events_path = run_dir / "events.json"

    with open(pitch_path, "r", encoding="utf-8") as fh:
        pitch_data: dict[str, Any] = json.load(fh)

    events: dict[str, Any] = {}
    if events_path.exists():
        with open(events_path, "r", encoding="utf-8") as fh:
            events = json.load(fh)

    all_positions = _build_position_index(pitch_data)

    # Render frames ------------------------------------------------------
    num_frames = len(_get_frames(pitch_data))
    if num_frames == 0:
        raise ValueError(f"No frames found in {pitch_path}")

    # Determine frame size from first render
    sample = render_frame(pitch_data, 0, events, all_positions)
    h, w = sample.shape[:2]

    out_path = run_dir / "replay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    try:
        for fi in range(num_frames):
            frame_img = render_frame(pitch_data, fi, events, all_positions)
            writer.write(frame_img)
    finally:
        writer.release()

    print(f"[replay] Wrote {num_frames} frames → {out_path}")
    return out_path
