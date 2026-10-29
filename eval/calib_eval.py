"""
Calibration error in metres: automatic calibration vs hand-placed keypoints.

    python -m eval.calib_eval clips/st_goal_crop.mp4 st_goal \
        data/soccertrack/st_goal_manual_points.json --frames 0 150 300

The run id must already have ``tracks.json`` (player boxes feed the fit's scale
check). Keypoints are ``[{"px": [x, y], "pitch": [x_m, y_m]}, ...]`` in video
pixels; for SoccerTrack v2 they come from ``raw/<match>/<match>_keypoints.json``.

Error = distance on the pitch between where the two homographies put the same
pixels, over a grid of pixels that the reference maps inside the pitch. The pitch
is symmetric under a half turn, so that equivalent fit counts as correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HALF_TURN = np.array([[-1, 0, 105], [0, -1, 68], [0, 0, 1]], np.float64)


def ground_error_m(H: np.ndarray, H_ref: np.ndarray, size: tuple[int, int], top: float = 0.15) -> np.ndarray:
    """Per-pixel pitch distance (m) between *H* and *H_ref* (both pitch → image)."""
    w, h = size
    us, vs = np.meshgrid(np.linspace(0, w - 1, 40), np.linspace(h * top, h - 1, 12))
    px = np.stack([us.ravel(), vs.ravel()], 1).reshape(-1, 1, 2).astype(np.float64)
    ref = cv2.perspectiveTransform(px, np.linalg.inv(H_ref)).reshape(-1, 2)
    on = (ref[:, 0] >= 0) & (ref[:, 0] <= 105) & (ref[:, 1] >= 0) & (ref[:, 1] <= 68)
    errs = [np.linalg.norm(cv2.perspectiveTransform(px, np.linalg.inv(G)).reshape(-1, 2)[on] - ref[on], axis=1)
            for G in (H, H @ HALF_TURN)]
    return min(errs, key=lambda e: float(np.median(e)) if len(e) else np.inf)


def main() -> None:
    from goal_explainer.api import get_client
    from goal_explainer.calibrate import _CALIB_WIDTH, compute_initial_homography
    from goal_explainer.track import read_frame

    ap = argparse.ArgumentParser(prog="eval.calib_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("run_id")
    ap.add_argument("keypoints", help="JSON list of {px, pitch} hand-placed points")
    ap.add_argument("--frames", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    pts = json.loads(Path(args.keypoints).read_text(encoding="utf-8"))
    H_ref, _ = cv2.findHomography(np.array([p["pitch"] for p in pts], np.float64),
                                  np.array([p["px"] for p in pts], np.float64), cv2.RANSAC, 5.0)
    tracks = json.loads((Path("runs") / args.run_id / "tracks.json").read_text(encoding="utf-8"))["frames"]
    client = get_client(args.run_id)
    rows = []
    for fi in args.frames:
        full = read_frame(Path(args.video), fi)
        s = min(1.0, _CALIB_WIDTH / full.shape[1])
        frame = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        people = [[v * s for v in o["bbox"]] for o in tracks[fi]["objects"] if o["cls"] in ("player", "goalkeeper")]
        try:
            H, method, cost = compute_initial_homography(client, frame, people)
        except Exception as exc:  # noqa: BLE001 — a failed frame is a result too
            rows.append({"frame": fi, "error": str(exc)})
            print(f"frame {fi}: calibration failed ({exc})")
            continue
        e = ground_error_m(np.diag([1 / s, 1 / s, 1.0]) @ H, H_ref, (full.shape[1], full.shape[0]))
        rows.append({"frame": fi, "method": method, "median_m": round(float(np.median(e)), 2),
                     "p90_m": round(float(np.percentile(e, 90)), 2)})
        print(f"frame {fi}: {method}, median {np.median(e):.1f} m, p90 {np.percentile(e, 90):.1f} m")
    out = Path("runs") / args.run_id / "calib_eval.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
