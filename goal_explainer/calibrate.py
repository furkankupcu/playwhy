"""
Step 2 — Camera calibration: estimate pitch-to-image homography per frame.

Three paths for the initial homography (tried in order):
  a) Muse Spark keypoint detection on the first frame.
  b) Hough line detection + template matching against the known pitch layout.
  c) Manual 4-point click (fallback — always works via the Gradio UI).

Frame-to-frame propagation:
  ORB feature matching on pitch / grass areas (excluding player masks).

Drift correction:
  Re-project template pitch lines, compare with detected white lines, and
  refine H via ``scipy.optimize.minimize`` on a distance-transform cost.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from scipy import optimize

from .api import get_client  # noqa: F401
from .pitch import LINES, PITCH_KEYPOINTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class HomographyResult(TypedDict):
    """Calibration data for a single frame."""
    frame_idx: int
    H: list[list[float]]       # 3×3 matrix stored row-major
    method: str                # "spark", "hough", "manual", "orb"
    error: float               # reprojection error in pixels
    reliable: bool             # False when error exceeds threshold


class CalibrationError(Exception):
    """Raised when both automatic calibration paths (a and b) fail."""
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REFIT_EVERY = 15               # frames between SAM region re-fits (~0.5 s)
_CALIB_WIDTH = 1280             # calibration runs on frames at most this wide
_HOUGH_MIN_LINE_LEN = 60        # minimum pixel length for Hough segments
_HOUGH_MAX_LINE_GAP = 15        # max gap within a single segment
_ORB_NFEATURES = 2000
_BF_DISTANCE_THRESHOLD = 50     # Hamming distance ceiling for ORB matches

# ---------------------------------------------------------------------------
# Path (a): Muse Spark keypoint detection
# ---------------------------------------------------------------------------


def detect_keypoints_spark(
    client,
    frame: np.ndarray,
) -> dict[str, tuple[float, float]] | None:
    """Ask Muse Spark for visible pitch keypoints in *frame* (pixel coords).

    Returns name → (px_x, px_y), or *None* with fewer than 4 keypoints.
    """
    from .track import jpeg

    names = sorted(PITCH_KEYPOINTS)
    schema = {
        "type": "object",
        "properties": {"keypoints": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": names},
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            "required": ["name", "x", "y"],
            "additionalProperties": False,
        }}},
        "required": ["keypoints"],
        "additionalProperties": False,
    }
    h, w = frame.shape[:2]
    prompt = (
        f"This football broadcast frame is {w}x{h} pixels. List every pitch "
        "keypoint you can clearly see (line intersections and corners only; "
        "skip anything hidden or off-screen), with its pixel coordinates. "
        "'left'/'right' refer to the goal at x=0 / x=105 on a pitch seen from "
        "the main camera side; 'top' is the far touchline."
    )
    try:
        # Full resolution so the returned pixels match the source frame.
        raw = client.spark_json(prompt, schema, images=[jpeg(frame, max_side=max(h, w))])
    except Exception:
        logger.warning("Spark keypoint request failed.", exc_info=True)
        return None

    keypoints = {
        k["name"]: (float(k["x"]), float(k["y"]))
        for k in raw.get("keypoints", [])
        if 0 <= k["x"] < w and 0 <= k["y"] < h
    }
    if len(keypoints) < 4:
        logger.warning("Spark returned only %d keypoints (need >=4).", len(keypoints))
        return None
    logger.info("Spark returned %d pitch keypoints.", len(keypoints))
    return keypoints


# ---------------------------------------------------------------------------
# Path (b): Hough line detection
# ---------------------------------------------------------------------------


def detect_lines_hough(
    frame: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    """Detect white line segments in *frame* using probabilistic Hough.

    Returns
    -------
    list of (x1, y1, x2, y2)
        Detected line segments after length filtering and near-parallel
        merging.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold for white lines on green grass
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # White: low saturation, high value
    mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Edge detection on the white mask
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=_HOUGH_MIN_LINE_LEN,
        maxLineGap=_HOUGH_MAX_LINE_GAP,
    )

    if raw_lines is None:
        return []

    segments: list[tuple[float, float, float, float]] = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length >= _HOUGH_MIN_LINE_LEN:
            segments.append((float(x1), float(y1), float(x2), float(y2)))

    # Merge nearby parallel segments
    segments = _merge_parallel_segments(segments)
    logger.info("Hough detected %d line segments.", len(segments))
    return segments


def _merge_parallel_segments(
    segments: list[tuple[float, float, float, float]],
    angle_tol: float = 5.0,
    dist_tol: float = 20.0,
) -> list[tuple[float, float, float, float]]:
    """Merge nearly-parallel, nearby segments into longer ones."""
    if not segments:
        return segments

    def _angle(seg: tuple[float, float, float, float]) -> float:
        dx = seg[2] - seg[0]
        dy = seg[3] - seg[1]
        return np.degrees(np.arctan2(dy, dx)) % 180.0

    def _midpoint(seg: tuple[float, float, float, float]) -> tuple[float, float]:
        return ((seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2)

    used = [False] * len(segments)
    merged: list[tuple[float, float, float, float]] = []

    for i, seg_i in enumerate(segments):
        if used[i]:
            continue
        group_pts: list[tuple[float, float]] = [
            (seg_i[0], seg_i[1]),
            (seg_i[2], seg_i[3]),
        ]
        used[i] = True

        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            seg_j = segments[j]

            # Check angle similarity
            if abs(_angle(seg_i) - _angle(seg_j)) > angle_tol:
                continue

            # Check distance between midpoints projected onto perpendicular
            mx_i, my_i = _midpoint(seg_i)
            mx_j, my_j = _midpoint(seg_j)
            if np.hypot(mx_i - mx_j, my_i - my_j) > dist_tol * 3:
                continue

            # Perpendicular distance from midpoint of j to line of i
            dx = seg_i[2] - seg_i[0]
            dy = seg_i[3] - seg_i[1]
            length_i = np.hypot(dx, dy)
            if length_i < 1:
                continue
            perp_dist = abs(dy * (mx_j - seg_i[0]) - dx * (my_j - seg_i[1])) / length_i
            if perp_dist > dist_tol:
                continue

            group_pts.extend([(seg_j[0], seg_j[1]), (seg_j[2], seg_j[3])])
            used[j] = True

        # Fit merged segment: take the two most distant points
        if len(group_pts) >= 2:
            pts = np.array(group_pts)
            # PCA-style: project onto the dominant direction
            mean = pts.mean(axis=0)
            centred = pts - mean
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            direction = vt[0]
            proj = centred @ direction
            i_min, i_max = int(proj.argmin()), int(proj.argmax())
            merged.append((
                float(pts[i_min, 0]), float(pts[i_min, 1]),
                float(pts[i_max, 0]), float(pts[i_max, 1]),
            ))

    return merged


# ---------------------------------------------------------------------------
# Template matching for Hough lines
# ---------------------------------------------------------------------------


def match_lines_to_template(
    lines: list[tuple[float, float, float, float]],
    pitch_keypoints: dict[str, tuple[float, float]],
) -> np.ndarray | None:
    """Try to match detected Hough segments to known pitch lines.

    If enough correspondences are found, compute and return the 3×3
    homography *H* (pitch metres → image pixels).  Returns *None* when
    matching fails.
    """
    if len(lines) < 4:
        return None

    # Strategy: use the endpoints of detected lines as candidate pixel
    # correspondences with the pitch template.  We look for line pairs that
    # can plausibly form known pitch structures (e.g., penalty-area rectangle,
    # centre line + touchline intersection).

    # Collect all line endpoints
    px_points: list[tuple[float, float]] = []
    for seg in lines:
        px_points.append((seg[0], seg[1]))
        px_points.append((seg[2], seg[3]))

    # Find intersections of detected lines (more reliable than raw endpoints)
    intersections = _find_line_intersections(lines)
    if len(intersections) < 4:
        return None

    # Attempt to pair intersections with pitch keypoints using geometric
    # consistency.  This is a heuristic: we try all 4-subsets and pick the
    # one with the lowest reprojection residual.

    kp_names = list(pitch_keypoints.keys())
    kp_coords = np.array([pitch_keypoints[n] for n in kp_names], dtype=np.float64)
    int_coords = np.array(intersections, dtype=np.float64)

    best_H: np.ndarray | None = None
    best_inliers = 0

    # Brute force over combinations (limited by intersection count)
    n_ints = min(len(int_coords), 30)  # cap for speed
    n_kps = len(kp_coords)
    rng = np.random.RandomState(0)

    for _ in range(500):
        # Pick 4 random intersections and 4 random keypoints
        if n_ints < 4 or n_kps < 4:
            break
        idx_i = rng.choice(n_ints, size=4, replace=False)
        idx_k = rng.choice(n_kps, size=4, replace=False)

        src = kp_coords[idx_k]  # pitch metres
        dst = int_coords[idx_i]  # image pixels

        H_cand, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H_cand is None:
            continue

        # Count inliers: project all pitch keypoints, find closest intersection
        proj = cv2.perspectiveTransform(
            kp_coords.reshape(-1, 1, 2), H_cand,
        ).reshape(-1, 2)

        dists = np.linalg.norm(
            proj[:, None, :] - int_coords[None, :, :], axis=2,
        )  # (n_kps, n_ints)
        min_dists = dists.min(axis=1)
        inliers = int((min_dists < 15.0).sum())

        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H_cand

    if best_H is not None and best_inliers >= 4:
        logger.info("Hough template matching found %d inliers.", best_inliers)
        return best_H

    return None


def _find_line_intersections(
    lines: list[tuple[float, float, float, float]],
    img_w: int = 1920,
    img_h: int = 1080,
) -> list[tuple[float, float]]:
    """Compute pairwise intersections of line segments within image bounds."""
    intersections: list[tuple[float, float]] = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = _seg_intersection(lines[i], lines[j])
            if pt is not None:
                x, y = pt
                if 0 <= x <= img_w and 0 <= y <= img_h:
                    intersections.append(pt)

    return intersections


def _seg_intersection(
    s1: tuple[float, float, float, float],
    s2: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Compute the intersection of two infinite lines through segments *s1* and *s2*."""
    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


# ---------------------------------------------------------------------------
# Path (c): Manual calibration
# ---------------------------------------------------------------------------


def manual_calibration(
    frame: np.ndarray,
    points_px: list[tuple[float, float]],
    points_pitch: list[tuple[float, float]],
) -> np.ndarray:
    """Compute *H* from user-supplied pixel ↔ pitch point pairs.

    Parameters
    ----------
    frame
        The reference image (unused in the computation but kept in the
        signature for potential visualisation).
    points_px
        At least 4 pixel coordinates clicked by the user.
    points_pitch
        Corresponding pitch coordinates in metres.

    Returns
    -------
    np.ndarray
        3×3 homography mapping pitch metres → image pixels.
    """
    if len(points_px) < 4 or len(points_pitch) < 4:
        raise ValueError("Manual calibration requires ≥ 4 point pairs.")

    src = np.array(points_pitch, dtype=np.float64)
    dst = np.array(points_px, dtype=np.float64)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise CalibrationError("findHomography returned None for manual points.")

    return H


# ---------------------------------------------------------------------------
# Path (a): SAM 3.1 pitch regions + template fitting
# ---------------------------------------------------------------------------

# Text prompt → template polygons (pitch metres) that the SAM mask should cover.
_PA = [(88.5, 13.84), (105.0, 13.84), (105.0, 54.16), (88.5, 54.16)]
_FIELD = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]
_CIRCLE = [(52.5 + 9.15 * np.cos(t), 34 + 9.15 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 48, endpoint=False)]
REGION_TEMPLATES: dict[str, list[list[tuple[float, float]]]] = {
    "soccer field": [_FIELD],
    # both boxes: SAM may return the near one, the far one, or both
    "penalty box": [_PA, [(105.0 - x, y) for x, y in _PA]],
    "center circle": [_CIRCLE],
}
_REGION_WEIGHTS = {"soccer field": 1.0, "penalty box": 2.0, "center circle": 1.0}
_FIT_WIDTH = 320  # masks are compared at this width; plenty for a homography
_REFINE_STARTS = 3  # best-ranked candidates refined; the cheapest one is not always the right basin
_JITTER_STARTS = 12  # extra random restarts on the first frame
_JITTER = 0.08      # restart spread, as a share of the frame width
# Goals break the template's 90° ambiguity: a goal's base must sit on a goal-line centre.
_GOAL_PROMPT = "soccer goal"
_GOAL_CENTRES = np.array([[[0.0, 34.0]], [[105.0, 34.0]]])
_GOAL_WEIGHT = 0.5
# Between refits the camera moves little, so a fit far from the tracked estimate is suspect.
_PRIOR_WEIGHT = 0.3
_PRIOR_GRID = np.array([[[x, y]] for x in (0.0, 52.5, 105.0) for y in (0.0, 34.0, 68.0)])
_ORIENT_AT = (96.75, 34.0)  # centre of the right penalty box


def sam_regions(client, frame: np.ndarray) -> dict[str, np.ndarray]:
    """SAM 3.1 masks (uint8 0/1, frame size) for each prompt in REGION_TEMPLATES."""
    from .api import mask_to_frame, parse_sam_text
    from .track import jpeg

    regions: dict[str, np.ndarray] = {}
    img = jpeg(frame, max_side=max(frame.shape[:2]))
    for prompt in [*REGION_TEMPLATES, _GOAL_PROMPT]:
        dets = [d for d in parse_sam_text(client.sam(prompt, image_jpeg=img)) if d["conf"] >= 0.5]
        if dets:
            regions[prompt] = np.maximum.reduce([mask_to_frame(d, frame.shape) for d in dets])
    return regions


def _render(H: np.ndarray, polys: list[list[tuple[float, float]]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    for poly in polys:
        pts = cv2.perspectiveTransform(np.array(poly, np.float64).reshape(-1, 1, 2), H)
        if not np.isfinite(pts).all() or np.abs(pts).max() > 1e5:
            continue
        cv2.fillPoly(out, [np.round(pts).astype(np.int32)], 1)
    return out


def region_cost(H: np.ndarray, small: dict[str, np.ndarray]) -> float:
    """Weighted mean of (1 − IoU) between projected template regions and SAM masks."""
    total = wsum = 0.0
    for name, mask in small.items():
        if name not in REGION_TEMPLATES:
            continue
        proj = _render(H, REGION_TEMPLATES[name], mask.shape)
        union = np.count_nonzero(proj | mask)
        iou = np.count_nonzero(proj & mask) / union if union else 0.0
        total += _REGION_WEIGHTS[name] * (1.0 - iou)
        wsum += _REGION_WEIGHTS[name]
    return total / wsum if wsum else 1.0


def _keeps_orientation(H: np.ndarray, at: tuple[float, float] = _ORIENT_AT) -> bool:
    """True if H maps pitch → image without mirroring.

    Pitch y grows up, image y grows down, so a camera looking at the pitch from
    above gives a negative Jacobian determinant. Mirrored fits match the masks
    equally well (the template is symmetric), so they are dropped here.
    """
    pts = np.array([[at, (at[0] + 1, at[1]), (at[0], at[1] + 1)]], np.float64)
    p = cv2.perspectiveTransform(pts, H)[0]
    a, b = p[1] - p[0], p[2] - p[0]
    return float(a[0] * b[1] - a[1] * b[0]) < 0


def _quad(mask: np.ndarray) -> np.ndarray | None:
    """Four corners (cyclic order) of the largest blob in *mask*, or None."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _quad_candidates(quad: np.ndarray, template: list[tuple[float, float]]) -> list[np.ndarray]:
    """All 8 corner assignments (4 rotations × 2 directions) of *quad* to *template*."""
    src = np.array(template, np.float32)
    out = []
    for order in (quad, quad[::-1].copy()):
        for k in range(4):
            H = cv2.getPerspectiveTransform(src, np.roll(order, k, axis=0))
            if H is not None and np.isfinite(H).all():
                out.append(H)
    return out


def _goal_bases(mask: np.ndarray) -> np.ndarray:
    """Bottom-centre pixel of each goal blob, shape (n, 1, 2)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    pts = [[[stats[i, 0] + stats[i, 2] / 2, stats[i, 1] + stats[i, 3]]] for i in range(1, n)]
    return np.array(pts, np.float64).reshape(-1, 1, 2)


def _project_ok(pts: np.ndarray, H: np.ndarray) -> np.ndarray | None:
    p = cv2.perspectiveTransform(pts, H)
    return p if np.isfinite(p).all() else None


_LINE_WIDTH = 640      # line cues need more pixels than region cues
_LINE_CAP_PX = 15.0    # distances beyond this count as "no line here"
_PLAYER_WIDTH_M = 0.9  # a player's box is about this wide on the ground (arms, stride)
_LINE_WEIGHT = 1.0
_SCALE_WEIGHT = 1.0


def _template_points() -> np.ndarray:
    """Points every 0.5 m along every pitch marking, shape (n, 1, 2)."""
    pts = []
    for _, poly in LINES:
        for (xa, ya), (xb, yb) in zip(poly, poly[1:]):
            n = max(2, int(np.hypot(xb - xa, yb - ya) * 2))
            pts += [(xa + (xb - xa) * t, ya + (yb - ya) * t) for t in np.linspace(0, 1, n)]
    return np.array(pts, np.float64).reshape(-1, 1, 2)


_TEMPLATE_PTS = _template_points()


class _FitContext:
    """Everything the fit cost needs, precomputed once per frame."""

    def __init__(self, regions, frame, people, prior):
        h, w = next(iter(regions.values())).shape
        s = _FIT_WIDTH / w
        self.S = np.diag([s, s, 1.0])
        size = (_FIT_WIDTH, max(1, round(h * s)))
        self.small = {k: cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) for k, m in regions.items()}
        self.width = _FIT_WIDTH
        field = regions.get("soccer field")
        grown = cv2.dilate(field, np.ones((15, 15), np.uint8)) if field is not None else None

        # Goals count only if their base is on the pitch (training goals stand beside it).
        goals = _goal_bases(regions[_GOAL_PROMPT]) if _GOAL_PROMPT in regions else np.zeros((0, 1, 2))
        if grown is not None and len(goals):
            keep = [g for g in goals if grown[min(h - 1, int(g[0, 1])), min(w - 1, int(g[0, 0]))]]
            goals = np.array(keep, np.float64).reshape(-1, 1, 2)
        self.goals = goals * s

        self.prior = self.S @ prior if prior is not None else None
        self.people = people or []

        # White line pixels, only on the pitch (white stands fooled the old line fit) and not on players.
        self.lines = None
        if frame is not None and grown is not None:
            ls = _LINE_WIDTH / w
            self.L = np.diag([ls, ls, 1.0])
            # Threshold at full size: downscaling first blends thin lines into the grass.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            full_lines = ((hsv[..., 2] > 170) & (hsv[..., 1] < 60)).astype(np.uint8) & grown
            for x1, y1, x2, y2 in self.people:
                full_lines[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = 0
            size_l = (_LINE_WIDTH, max(1, round(h * ls)))
            lines = (cv2.resize(full_lines.astype(np.float32), size_l, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
            on_pitch = cv2.resize(grown, size_l, interpolation=cv2.INTER_NEAREST)
            if lines.sum() > 50:
                self.lines, self.on_pitch = lines, on_pitch
                self.dt = cv2.distanceTransform(1 - lines, cv2.DIST_L2, 3)

    def line_cost(self, H: np.ndarray) -> float:
        """Symmetric chamfer between projected markings and white pixels (0 = on the lines)."""
        p = _project_ok(_TEMPLATE_PTS, self.L @ H)
        if p is None:
            return 1.0
        p = p.reshape(-1, 2)
        h, w = self.lines.shape
        ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        p = p[ok]
        p = p[self.on_pitch[p[:, 1].astype(int), p[:, 0].astype(int)] > 0]
        if len(p) < 20:
            return 1.0
        precision = np.minimum(self.dt[p[:, 1].astype(int), p[:, 0].astype(int)], _LINE_CAP_PX).mean()
        drawn = np.zeros_like(self.lines)
        drawn[p[:, 1].astype(int), p[:, 0].astype(int)] = 1
        drawn = cv2.dilate(drawn, np.ones((3, 3), np.uint8))
        recall = np.minimum(cv2.distanceTransform(1 - drawn, cv2.DIST_L2, 3)[self.lines > 0], _LINE_CAP_PX).mean()
        return float(precision + recall) / (2 * _LINE_CAP_PX)

    def player_width_m(self, H: np.ndarray) -> float | None:
        """Median ground width (m) of the player boxes under *H*; ~0.9 m when H is right."""
        if not self.people:
            return None
        try:
            Hi = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        feet = np.array([[[x1, y2]] for x1, _, _, y2 in self.people] + [[[x2, y2]] for _, _, x2, y2 in self.people],
                        np.float64)
        g = _project_ok(feet, Hi)
        if g is None:
            return None
        n = len(self.people)
        return float(np.median(np.linalg.norm(g[:n, 0] - g[n:, 0], axis=1)))

    def cost(self, H: np.ndarray) -> float:
        """Region IoU + line chamfer + player scale + on-pitch goals + prior (H at calibration scale)."""
        Hs = self.S @ H
        cost = region_cost(Hs, self.small)
        if self.lines is not None:
            cost += _LINE_WEIGHT * self.line_cost(H)
        wm = self.player_width_m(H)
        if wm is not None:
            cost += _SCALE_WEIGHT * (min(1.0, abs(np.log(max(wm, 1e-3) / _PLAYER_WIDTH_M)) / np.log(2.5)))
        if len(self.goals):
            centres = _project_ok(_GOAL_CENTRES, Hs)
            if centres is None:
                return 4.0
            d = np.linalg.norm(self.goals[:, None, 0, :] - centres[None, :, 0, :], axis=2).min(axis=1)
            cost += _GOAL_WEIGHT * float(np.minimum(1.0, d / (0.1 * self.width)).mean())
        if self.prior is not None:
            a, b = _project_ok(_PRIOR_GRID, self.prior), _project_ok(_PRIOR_GRID, Hs)
            if a is None or b is None:
                return 4.0
            d = np.linalg.norm(a - b, axis=2).ravel()
            cost += _PRIOR_WEIGHT * float(np.minimum(1.0, np.median(d) / (0.2 * self.width)))
        return cost


def fit_regions(
    regions: dict[str, np.ndarray],
    extra_candidates: list[np.ndarray] = (),
    prior: np.ndarray | None = None,
    frame: np.ndarray | None = None,
    people: list[list[float]] | None = None,
) -> tuple[np.ndarray, float] | None:
    """Best pitch → image homography for the SAM *regions*, with a reliability cost.

    *frame* adds a white-line fit and *people* (player boxes, frame pixels) a scale
    check: SAM regions alone let a shrunken whole pitch fill the visible grass.
    *prior* (e.g. the tracked homography of this frame) penalises jumps away from it.
    The returned cost is the region cost, or 1.0 when the players' implied size is
    implausible (a wrong scale can still have a near-perfect region overlap).
    """
    if "penalty box" not in regions and "soccer field" not in regions:
        return None
    ctx = _FitContext(regions, frame, people, prior)

    candidates = list(extra_candidates) + ([prior] if prior is not None else [])
    for name, template in (("penalty box", _PA), ("soccer field", _FIELD)):
        q = _quad(regions[name]) if name in regions else None
        if q is not None:
            candidates += _quad_candidates(q, template)
    candidates = [H for H in candidates if _keeps_orientation(H)]
    if not candidates:
        return None
    ranked = sorted(candidates, key=ctx.cost)

    # Refine the best few by moving the image positions of 4 anchor points (well conditioned, in px).
    anchors = np.array(_PA, np.float32)

    def to_h(x: np.ndarray) -> np.ndarray:
        return cv2.getPerspectiveTransform(anchors, x.reshape(4, 2).astype(np.float32))

    def cost(x: np.ndarray) -> float:
        H = to_h(x)
        return ctx.cost(H) if _keeps_orientation(H) else 4.0

    def refine(x0: np.ndarray, maxiter: int) -> tuple[np.ndarray, float]:
        res = optimize.minimize(cost, x0, method="Nelder-Mead",
                                options={"xatol": 0.3, "fatol": 1e-4, "maxiter": maxiter})
        c0 = cost(x0)
        return (res.x, res.fun) if res.fun <= c0 else (x0, c0)

    starts = [cv2.perspectiveTransform(anchors.reshape(-1, 1, 2).astype(np.float64), H0).reshape(-1)
              for H0 in ranked[:_REFINE_STARTS]]
    best_x, best_c = min((refine(x0, 1500) for x0 in starts), key=lambda r: r[1])
    if prior is None:
        # No tracking prior (first frame): the cost has local minima, so also try
        # jittered starts around the best fit and keep the lowest.
        rng = np.random.default_rng(0)
        sigma = _JITTER * regions[next(iter(regions))].shape[1]
        for _ in range(_JITTER_STARTS):
            x, c = refine(best_x + rng.normal(0, sigma, best_x.shape), 600)
            if c < best_c:
                best_x, best_c = x, c
    H = to_h(best_x)
    wm = ctx.player_width_m(H)
    plausible = wm is None or 0.4 <= wm <= 2.0
    return H, region_cost(ctx.S @ H, ctx.small) if plausible else 1.0


# ---------------------------------------------------------------------------
# Initial homography (orchestrates paths a → b → raise)
# ---------------------------------------------------------------------------

_REGION_COST_OK = 0.45  # mean (1 − IoU). Good fits score ~0.1–0.35: SAM's field mask includes the green run-off


def compute_initial_homography(
    client,
    frame: np.ndarray,
    people: list[list[float]] | None = None,
) -> tuple[np.ndarray, str, float | None]:
    """Automatic calibration: SAM regions (seeded by Spark keypoints), then Hough.

    Raises
    ------
    CalibrationError
        If no path works — the UI then offers the manual 4-point path (c).
    """
    seeds: list[np.ndarray] = []
    spark_kps = detect_keypoints_spark(client, frame)
    if spark_kps is not None:
        pairs = [(PITCH_KEYPOINTS[n], px) for n, px in spark_kps.items() if n in PITCH_KEYPOINTS]
        if len(pairs) >= 4:
            H, _ = cv2.findHomography(np.array([p for p, _ in pairs], np.float64),
                                      np.array([q for _, q in pairs], np.float64), cv2.RANSAC, 5.0)
            if H is not None:
                seeds.append(H)

    # --- Path (a): SAM regions; Spark keypoints are only an extra starting point ---
    fit = fit_regions(sam_regions(client, frame), seeds, frame=frame, people=people)
    if fit is not None:
        H, cost = fit
        logger.info("Region fit cost %.3f", cost)
        if cost <= _REGION_COST_OK:
            return H, "sam_regions", cost

    # --- Path (b): Hough lines + template matching ---
    lines = detect_lines_hough(frame)
    if lines:
        H_hough = match_lines_to_template(lines, PITCH_KEYPOINTS)
        if H_hough is not None:
            logger.info("Initial homography via Hough line matching.")
            return H_hough, "hough", None

    raise CalibrationError(
        "Automatic calibration failed. Use manual_calibration() with 4+ clicked points."
    )


# ---------------------------------------------------------------------------
# ORB-based frame-to-frame tracking
# ---------------------------------------------------------------------------


def track_homography_orb(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    prev_H: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Propagate *prev_H* to the current frame via ORB feature matching.

    Parameters
    ----------
    prev_frame, curr_frame
        Consecutive BGR frames.
    prev_H
        3×3 homography for the previous frame (pitch → pixels).
    mask
        Optional binary mask where 255 = area to **use** (typically pitch /
        grass, with player regions masked out).

    Returns
    -------
    np.ndarray
        Updated 3×3 homography for *curr_frame*.
    """
    grey_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    grey_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB.create(nfeatures=_ORB_NFEATURES)

    kp1, des1 = orb.detectAndCompute(grey_prev, mask)
    kp2, des2 = orb.detectAndCompute(grey_curr, mask)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        logger.warning("ORB tracking: insufficient features — carrying forward prev_H.")
        return prev_H.copy()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good: list[cv2.DMatch] = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance and m.distance < _BF_DISTANCE_THRESHOLD:
                good.append(m)

    if len(good) < 4:
        logger.warning("ORB tracking: only %d good matches — carrying forward prev_H.", len(good))
        return prev_H.copy()

    pts_prev = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_curr = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H_delta, inlier_mask = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, 3.0)
    if H_delta is None:
        logger.warning("ORB tracking: RANSAC failed — carrying forward prev_H.")
        return prev_H.copy()

    # The new homography maps pitch → current image:
    # prev_H maps pitch → prev image
    # H_delta maps prev image → curr image
    # So: H_curr = H_delta @ prev_H
    H_curr = H_delta @ prev_H
    return H_curr


# ---------------------------------------------------------------------------
# Drift correction
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Player-region mask helper
# ---------------------------------------------------------------------------


def _player_mask_from_tracks(
    tracks_frame: dict | None,
    img_shape: tuple[int, int],
    margin: int = 15,
) -> np.ndarray:
    """Create a binary mask with 255 on grass (non-player) regions.

    Used by ORB matching to avoid features on moving players.
    """
    mask = np.full(img_shape[:2], 255, dtype=np.uint8)
    if tracks_frame is None:
        return mask

    for obj in tracks_frame.get("objects", []):
        cls = obj.get("cls", "")
        if cls in ("player", "goalkeeper", "referee", "ball"):
            x1, y1, x2, y2 = obj["bbox"]
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img_shape[1], x2 + margin)
            y2 = min(img_shape[0], y2 + margin)
            mask[y1:y2, x1:x2] = 0

    return mask


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run(
    video_path: Path,
    clip_id: str,
    tracks_path: Path | None = None,
    manual_points: list | None = None,
) -> Path:
    """Run calibration for the full clip and save per-frame homographies.

    Produces ``runs/<clip_id>/homography.json``.

    Parameters
    ----------
    video_path
        Source clip.
    clip_id
        Unique run identifier.
    tracks_path
        Optional path to ``tracks.json`` (provides player masks for ORB).
    manual_points
        Optional list of ``{"px": [x, y], "pitch": [x_m, y_m]}`` dicts
        for fallback path (c).  If supplied, path (c) is used directly.

    Returns
    -------
    Path
        Path to the saved ``homography.json``.
    """
    from .track import iter_frames  # local to avoid circular at module level

    out_dir = Path("runs") / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Player boxes (if available) keep ORB features off moving players
    tracks_by_idx: dict[int, dict] = {}
    if tracks_path is not None and tracks_path.exists():
        data = json.loads(tracks_path.read_text(encoding="utf-8"))
        frames_list = data["frames"] if isinstance(data, dict) else data
        tracks_by_idx = {tf["frame_idx"]: tf for tf in frames_list}

    client = get_client(clip_id)
    results: list[HomographyResult] = []
    H_prev: np.ndarray | None = None
    prev_frame: np.ndarray | None = None
    cost = 1.0

    for i, full in enumerate(iter_frames(video_path)):
        # Work at ≤ _CALIB_WIDTH: SAM segments pitch regions badly on 4K input, and
        # ORB on 4K is ~9× slower. H is scaled back to full resolution when saved.
        s = min(1.0, _CALIB_WIDTH / full.shape[1])
        frame = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else full
        to_full = np.diag([1 / s, 1 / s, 1.0])
        people = [[v * s for v in o["bbox"]] for o in (tracks_by_idx.get(i) or {}).get("objects", [])
                  if o.get("cls") in ("player", "goalkeeper")]
        if i == 0:
            # --- Step 1: Initial homography on first frame ---
            if manual_points is not None and len(manual_points) >= 4:
                # Path (c): manual. Points are {"px": [x, y], "pitch": [x_m, y_m]} in full-res px.
                H_curr = manual_calibration(
                    frame,
                    [(p["px"][0] * s, p["px"][1] * s) for p in manual_points],
                    [tuple(p["pitch"]) for p in manual_points],
                )
                m, cost = "manual", None
            else:
                # Paths (a) then (b) — raises CalibrationError so the UI can offer (c)
                H_curr, m, cost = compute_initial_homography(client, frame, people)
        else:
            # --- Step 2: ORB propagation between refits ---
            orb_mask = cv2.resize(_player_mask_from_tracks(tracks_by_idx.get(i - 1), full.shape),
                                  (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            H_curr = track_homography_orb(prev_frame, frame, H_prev, mask=orb_mask)
            m = "orb"

        # --- Step 3: Re-fit to SAM pitch regions (removes ORB drift), seeded with H_curr ---
        # Hand-placed points are trusted: no SAM refit may override them
        if manual_points is None and i % _REFIT_EVERY == 0 and (i > 0 or cost is None):
            fit = fit_regions(sam_regions(client, frame), prior=H_curr if i > 0 else None,
                              extra_candidates=[H_curr], frame=frame, people=people)
            if fit is not None and fit[1] <= _REGION_COST_OK:
                H_curr, cost = fit
                m = m if i == 0 else "sam_regions"
            elif cost is not None:
                cost = fit[1] if fit is not None else 1.0

        # --- Step 4: Reliability = how well the template covers SAM's regions at the last fit ---
        results.append(HomographyResult(
            frame_idx=i,
            H=(to_full @ H_curr).tolist(),
            method=m,
            error=round(cost, 3) if cost is not None else 0.0,
            reliable=cost is None or cost <= _REGION_COST_OK,
        ))
        H_prev, prev_frame = H_curr, frame

    if not results:
        raise CalibrationError("No frames read from video.")

    # --- Step 5: Save ---
    out_path = out_dir / "homography.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    n_reliable = sum(1 for r in results if r["reliable"])
    logger.info("Homography saved to %s — %d/%d frames reliable.", out_path, n_reliable, len(results))
    return out_path
