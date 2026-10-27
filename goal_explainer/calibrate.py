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
from .pitch import PITCH_KEYPOINTS, PITCH_LINES  # noqa: F401

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

_REPROJ_ERROR_THRESHOLD = 15.0   # pixels — above this the frame is unreliable
_DRIFT_CORRECTION_INTERVAL = 30  # correct drift every N frames
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
# Initial homography (orchestrates paths a → b → raise)
# ---------------------------------------------------------------------------


def compute_initial_homography(
    client,
    frame: np.ndarray,
    tracks_frame: dict | None = None,
) -> tuple[np.ndarray, str]:
    """Try automatic calibration paths (a then b) and return *(H, method)*.

    Raises
    ------
    CalibrationError
        If both paths fail — the Gradio UI should then offer path (c).
    """
    # --- Path (a): Muse Spark ---
    spark_kps = detect_keypoints_spark(client, frame)
    if spark_kps is not None:
        # Match Spark keypoints to known pitch coordinates
        matched_px: list[tuple[float, float]] = []
        matched_pitch: list[tuple[float, float]] = []
        for name, px_xy in spark_kps.items():
            if name in PITCH_KEYPOINTS:
                matched_px.append(px_xy)
                matched_pitch.append(PITCH_KEYPOINTS[name])

        if len(matched_px) >= 4:
            src = np.array(matched_pitch, dtype=np.float64)
            dst = np.array(matched_px, dtype=np.float64)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                logger.info("Initial homography via Spark (%d keypoints).", len(matched_px))
                return H, "spark"

    # --- Path (b): Hough lines + template matching ---
    lines = detect_lines_hough(frame)
    if lines:
        H_hough = match_lines_to_template(lines, PITCH_KEYPOINTS)
        if H_hough is not None:
            logger.info("Initial homography via Hough line matching.")
            return H_hough, "hough"

    raise CalibrationError(
        "Automatic calibration failed (paths a and b). "
        "Use manual_calibration() with 4+ clicked points."
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


def _white_line_mask(frame: np.ndarray) -> np.ndarray:
    """Create a binary mask of white pitch-line pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _project_pitch_lines(
    H: np.ndarray,
    img_shape: tuple[int, int],
) -> np.ndarray:
    """Render projected pitch template lines as a binary mask.

    Parameters
    ----------
    H
        Homography mapping pitch metres → image pixels.
    img_shape
        (height, width) of the target image.

    Returns
    -------
    np.ndarray
        uint8 binary mask (255 on projected lines).
    """
    canvas = np.zeros(img_shape[:2], dtype=np.uint8)

    for seg in PITCH_LINES:
        # Each line is ((x1_m, y1_m), (x2_m, y2_m)) in pitch metres
        pts_pitch = np.array([[seg[0], seg[1]]], dtype=np.float64).reshape(-1, 1, 2)
        pts_pitch2 = np.array([[seg[2], seg[3]]], dtype=np.float64).reshape(-1, 1, 2)

        pt1 = cv2.perspectiveTransform(pts_pitch, H).reshape(2).astype(int)
        pt2 = cv2.perspectiveTransform(pts_pitch2, H).reshape(2).astype(int)

        cv2.line(canvas, tuple(pt1), tuple(pt2), 255, 2)

    return canvas


def correct_drift(
    frame: np.ndarray,
    H: np.ndarray,
    pitch_lines: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Refine *H* so that projected pitch lines align with detected white lines.

    Uses ``scipy.optimize.minimize`` (Nelder–Mead) to minimise the sum of
    distance-transform values at the projected line positions.

    Parameters
    ----------
    frame
        Current BGR frame.
    H
        Current homography estimate (pitch → image).
    pitch_lines
        Override for ``PITCH_LINES``; unused when *None*.

    Returns
    -------
    np.ndarray
        Refined 3×3 homography.
    """
    h_img, w_img = frame.shape[:2]

    # Detected white lines → distance transform
    line_mask = _white_line_mask(frame)
    dist_transform = cv2.distanceTransform(255 - line_mask, cv2.DIST_L2, 5)

    # Sample points along template lines in pitch coordinates
    sample_pts_pitch: list[tuple[float, float]] = []
    lines_src = pitch_lines if pitch_lines is not None else PITCH_LINES
    for seg in lines_src:
        x1, y1, x2, y2 = seg
        for alpha in np.linspace(0, 1, 20):
            sample_pts_pitch.append((
                x1 + alpha * (x2 - x1),
                y1 + alpha * (y2 - y1),
            ))

    pts_pitch = np.array(sample_pts_pitch, dtype=np.float64).reshape(-1, 1, 2)

    def _cost(params: np.ndarray) -> float:
        """Cost = mean distance-transform value at projected sample points."""
        H_trial = params.reshape(3, 3)
        # Normalise so H[2,2] == 1
        if abs(H_trial[2, 2]) < 1e-12:
            return 1e6
        H_trial = H_trial / H_trial[2, 2]

        try:
            projected = cv2.perspectiveTransform(pts_pitch, H_trial).reshape(-1, 2)
        except cv2.error:
            return 1e6

        total = 0.0
        count = 0
        for px, py in projected:
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < w_img and 0 <= iy < h_img:
                total += dist_transform[iy, ix]
                count += 1

        return total / max(count, 1)

    x0 = H.flatten()
    result = optimize.minimize(
        _cost,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1e-5, "fatol": 0.1},
    )

    H_refined = result.x.reshape(3, 3)
    H_refined = H_refined / H_refined[2, 2]
    logger.debug("Drift correction: cost %.2f → %.2f", _cost(x0), result.fun)
    return H_refined


# ---------------------------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------------------------


def compute_reprojection_error(
    frame: np.ndarray,
    H: np.ndarray,
) -> float:
    """Average distance (px) from projected pitch lines to nearest detected white line.

    A low value (< ~10 px) indicates good calibration.
    """
    h_img, w_img = frame.shape[:2]

    line_mask = _white_line_mask(frame)
    dist_transform = cv2.distanceTransform(255 - line_mask, cv2.DIST_L2, 5)

    # Sample along pitch lines
    sample_pts: list[tuple[float, float]] = []
    for seg in PITCH_LINES:
        x1, y1, x2, y2 = seg
        for alpha in np.linspace(0, 1, 30):
            sample_pts.append((
                x1 + alpha * (x2 - x1),
                y1 + alpha * (y2 - y1),
            ))

    if not sample_pts:
        return float("inf")

    pts = np.array(sample_pts, dtype=np.float64).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    except cv2.error:
        return float("inf")

    distances: list[float] = []
    for px, py in projected:
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < w_img and 0 <= iy < h_img:
            distances.append(float(dist_transform[iy, ix]))

    if not distances:
        return float("inf")

    return float(np.mean(distances))


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

    results: list[HomographyResult] = []
    H_prev: np.ndarray | None = None
    prev_frame: np.ndarray | None = None

    for i, frame in enumerate(iter_frames(video_path)):
        if i == 0:
            # --- Step 1: Initial homography on first frame ---
            if manual_points is not None and len(manual_points) >= 4:
                # Path (c): manual. Points are {"px": [x, y], "pitch": [x_m, y_m]}.
                H_curr = manual_calibration(
                    frame,
                    [tuple(p["px"]) for p in manual_points],
                    [tuple(p["pitch"]) for p in manual_points],
                )
                m = "manual"
            else:
                # Paths (a) then (b) — raises CalibrationError so the UI can offer (c)
                H_curr, m = compute_initial_homography(
                    get_client(clip_id), frame, tracks_by_idx.get(0),
                )
        else:
            # --- Step 2: ORB propagation ---
            orb_mask = _player_mask_from_tracks(tracks_by_idx.get(i - 1), frame.shape)
            H_curr = track_homography_orb(prev_frame, frame, H_prev, mask=orb_mask)
            m = "orb"
            # --- Step 3: Periodic drift correction ---
            if i % _DRIFT_CORRECTION_INTERVAL == 0:
                H_curr = correct_drift(frame, H_curr)

        # --- Step 4: Error & reliability ---
        err = compute_reprojection_error(frame, H_curr)
        results.append(HomographyResult(
            frame_idx=i,
            H=H_curr.tolist(),
            method=m,
            error=round(err, 3),
            reliable=err < _REPROJ_ERROR_THRESHOLD,
        ))
        H_prev, prev_frame = H_curr, frame

    if not results:
        raise CalibrationError("No frames read from video.")

    # --- Step 5: Save ---
    out_path = out_dir / "homography.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    n_reliable = sum(1 for r in results if r["reliable"])
    logger.info(
        "Homography saved to %s — %d/%d frames reliable (threshold %.1f px).",
        out_path, n_reliable, len(results), _REPROJ_ERROR_THRESHOLD,
    )
    return out_path
