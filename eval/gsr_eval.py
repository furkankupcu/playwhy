"""
Evaluate player position estimation against SoccerNet Game State Reconstruction.
Dataset: ~200 clips × 30s, with hand-annotated player pitch positions.

Metrics:
- Mean/median position error (metres)
- Recall (fraction of players matched within 5m threshold)
- Breakdown by calibration method (a/b/c)

Uses Hungarian algorithm (scipy.optimize.linear_sum_assignment) for matching.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_gsr_annotations(gsr_dir: Path) -> dict[str, list[list[tuple[str, float, float]]]]:
    """Load SoccerNet-GSR annotations from per-clip JSON files.

    Expected format: each JSON file contains per-frame player positions
    in metres on the pitch coordinate system.  Structure assumed::

        {
            "clip_id": "...",
            "frames": [
                {
                    "frame_idx": 0,
                    "players": [
                        {"player_id": "p1", "x": 52.3, "y": 34.1},
                        ...
                    ]
                },
                ...
            ]
        }

    .. note::
       Check the actual SoccerNet-GSR release format and adjust the
       parser below accordingly — field names or nesting may differ.

    Args:
        gsr_dir: Directory containing one JSON file per clip.

    Returns:
        Mapping of *clip_id* → list of frames, where each frame is a
        list of ``(player_id, x, y)`` tuples.
    """
    annotations: dict[str, list[list[tuple[str, float, float]]]] = {}

    for json_path in sorted(gsr_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)

        clip_id: str = data.get("clip_id", json_path.stem)
        frames: list[list[tuple[str, float, float]]] = []

        for frame_data in data.get("frames", []):
            players: list[tuple[str, float, float]] = []
            for p in frame_data.get("players", []):
                players.append((str(p["player_id"]), float(p["x"]), float(p["y"])))
            frames.append(players)

        annotations[clip_id] = frames

    return annotations


def load_predictions(runs_dir: Path, clip_id: str) -> list[dict[str, Any]]:
    """Load pitch.json from ``runs/{clip_id}/``.

    The file is expected to contain a list of per-frame dicts, each with
    a ``players`` key holding a list of ``{"x": …, "y": …}`` entries
    (positions in metres on the pitch).

    Args:
        runs_dir: Root runs directory.
        clip_id: Identifier for the clip.

    Returns:
        Per-frame player position dicts as stored in *pitch.json*.
    """
    pitch_path = runs_dir / clip_id / "pitch.json"
    with open(pitch_path, "r", encoding="utf-8") as fh:
        data: Any = json.load(fh)

    # Accept either a bare list or a dict with a "frames" key.
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "frames" in data:
        return data["frames"]

    raise ValueError(f"Unexpected pitch.json structure in {pitch_path}")


def match_players(
    pred_positions: list[tuple[float, float]],
    gt_positions: list[tuple[float, float]],
    threshold: float = 5.0,
) -> tuple[list[tuple[int, int, float]], float]:
    """Match predicted player positions to ground-truth using the Hungarian algorithm.

    A cost matrix of pairwise Euclidean distances is built and solved
    with :func:`scipy.optimize.linear_sum_assignment`.  Matches whose
    distance exceeds *threshold* are discarded.

    Args:
        pred_positions: List of ``(x, y)`` predicted positions.
        gt_positions: List of ``(x, y)`` ground-truth positions.
        threshold: Maximum allowed distance (metres) for a valid match.

    Returns:
        A tuple of:
        - List of ``(pred_idx, gt_idx, distance)`` for accepted matches.
        - Recall (fraction of GT players matched within threshold).
    """
    if not pred_positions or not gt_positions:
        return [], 0.0

    pred_arr = np.array(pred_positions, dtype=np.float64)  # (N, 2)
    gt_arr = np.array(gt_positions, dtype=np.float64)      # (M, 2)

    # Cost matrix: (N_pred × M_gt) Euclidean distances.
    diff = pred_arr[:, np.newaxis, :] - gt_arr[np.newaxis, :, :]  # (N, M, 2)
    cost_matrix = np.linalg.norm(diff, axis=2)                    # (N, M)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int, float]] = []
    for r, c in zip(row_indices, col_indices):
        dist = float(cost_matrix[r, c])
        if dist <= threshold:
            matches.append((int(r), int(c), dist))

    recall = len(matches) / len(gt_positions) if gt_positions else 0.0
    return matches, recall


def evaluate_frame(
    pred: list[dict[str, Any]],
    gt: list[tuple[str, float, float]],
    threshold: float = 5.0,
) -> dict[str, Any]:
    """Evaluate a single frame of predictions against ground-truth.

    Args:
        pred: List of prediction dicts, each with ``"x"`` and ``"y"`` keys.
        gt: List of ``(player_id, x, y)`` ground-truth entries.
        threshold: Distance threshold in metres for a valid match.

    Returns:
        Dictionary with keys:
        ``matched``, ``total_gt``, ``total_pred``,
        ``mean_error``, ``median_error``, ``recall``, ``errors``.
    """
    pred_positions: list[tuple[float, float]] = [
        (float(p.get("x", p.get("x_m", 0.0))), float(p.get("y", p.get("y_m", 0.0))))
        for p in pred
    ]
    gt_positions: list[tuple[float, float]] = [
        (x, y) for _, x, y in gt
    ]

    matches, recall = match_players(pred_positions, gt_positions, threshold)
    errors: list[float] = [dist for _, _, dist in matches]

    return {
        "matched": len(matches),
        "total_gt": len(gt_positions),
        "total_pred": len(pred_positions),
        "mean_error": float(statistics.mean(errors)) if errors else 0.0,
        "median_error": float(statistics.median(errors)) if errors else 0.0,
        "recall": recall,
        "errors": errors,
    }


def evaluate_clip(
    clip_id: str,
    runs_dir: Path,
    gsr_dir: Path,
    gt_annotations: dict[str, list[list[tuple[str, float, float]]]] | None = None,
    threshold: float = 5.0,
) -> dict[str, Any]:
    """Evaluate all frames of a single clip.

    Args:
        clip_id: Clip identifier.
        runs_dir: Root runs directory.
        gsr_dir: Directory with GSR annotation JSONs.
        gt_annotations: Pre-loaded annotations (avoids re-reading from disk).
        threshold: Distance threshold in metres.

    Returns:
        Summary dict with per-frame results aggregated into overall
        ``mean_error``, ``median_error``, ``recall``, and the
        ``calibration_method`` recorded in *homography.json*.
    """
    # Ground truth -----------------------------------------------------------
    if gt_annotations is None:
        gt_annotations = load_gsr_annotations(gsr_dir)

    if clip_id not in gt_annotations:
        return {
            "clip_id": clip_id,
            "status": "missing_gt",
            "mean_error": float("inf"),
            "median_error": float("inf"),
            "recall": 0.0,
            "calibration_method": "unknown",
            "num_frames": 0,
        }

    gt_frames = gt_annotations[clip_id]

    # Predictions ------------------------------------------------------------
    try:
        pred_frames = load_predictions(runs_dir, clip_id)
    except (FileNotFoundError, ValueError):
        return {
            "clip_id": clip_id,
            "status": "missing_pred",
            "mean_error": float("inf"),
            "median_error": float("inf"),
            "recall": 0.0,
            "calibration_method": "unknown",
            "num_frames": 0,
        }

    # Calibration method -----------------------------------------------------
    homography_path = runs_dir / clip_id / "homography.json"
    calibration_method = "unknown"
    if homography_path.exists():
        with open(homography_path, "r", encoding="utf-8") as fh:
            h_data: dict[str, Any] = json.load(fh)
        calibration_method = h_data.get("method", "unknown")

    # Per-frame evaluation ---------------------------------------------------
    num_frames = min(len(pred_frames), len(gt_frames))
    frame_results: list[dict[str, Any]] = []
    all_errors: list[float] = []
    total_matched = 0
    total_gt = 0

    for i in range(num_frames):
        pred_players = pred_frames[i].get("players", pred_frames[i].get("positions", []))
        gt_players = gt_frames[i]

        result = evaluate_frame(pred_players, gt_players, threshold)
        frame_results.append(result)
        all_errors.extend(result["errors"])
        total_matched += result["matched"]
        total_gt += result["total_gt"]

    overall_recall = total_matched / total_gt if total_gt > 0 else 0.0

    return {
        "clip_id": clip_id,
        "status": "ok",
        "mean_error": float(statistics.mean(all_errors)) if all_errors else 0.0,
        "median_error": float(statistics.median(all_errors)) if all_errors else 0.0,
        "recall": overall_recall,
        "calibration_method": calibration_method,
        "num_frames": num_frames,
        "total_matched": total_matched,
        "total_gt": total_gt,
    }


def evaluate_all(
    runs_dir: Path,
    gsr_dir: Path,
    clip_ids: list[str] | None = None,
    threshold: float = 5.0,
) -> dict[str, Any]:
    """Evaluate every clip and aggregate results.

    Args:
        runs_dir: Root runs directory.
        gsr_dir: GSR annotation directory.
        clip_ids: Explicit list of clip IDs.  If *None*, all clips found
            in the GSR directory are evaluated.
        threshold: Match distance threshold (metres).

    Returns:
        Full results dict with overall stats, per-method breakdown,
        per-clip summaries, and the top-3 worst clips.
    """
    gt_annotations = load_gsr_annotations(gsr_dir)

    if clip_ids is None:
        clip_ids = sorted(gt_annotations.keys())

    per_clip: list[dict[str, Any]] = []
    for cid in clip_ids:
        summary = evaluate_clip(cid, runs_dir, gsr_dir, gt_annotations, threshold)
        per_clip.append(summary)

    # Overall aggregation ----------------------------------------------------
    ok_clips = [c for c in per_clip if c["status"] == "ok" and c["num_frames"] > 0]
    all_mean_errors = [c["mean_error"] for c in ok_clips if c["mean_error"] < float("inf")]
    all_recalls = [c["recall"] for c in ok_clips]

    overall_mean = float(statistics.mean(all_mean_errors)) if all_mean_errors else 0.0
    overall_median = float(statistics.median(all_mean_errors)) if all_mean_errors else 0.0
    overall_recall = float(statistics.mean(all_recalls)) if all_recalls else 0.0

    # Per-method breakdown ---------------------------------------------------
    method_groups: dict[str, list[dict[str, Any]]] = {}
    for c in ok_clips:
        method_groups.setdefault(c["calibration_method"], []).append(c)

    per_method: dict[str, dict[str, Any]] = {}
    for method, clips in sorted(method_groups.items()):
        m_errors = [c["mean_error"] for c in clips]
        m_recalls = [c["recall"] for c in clips]
        per_method[method] = {
            "mean_error": float(statistics.mean(m_errors)),
            "median_error": float(statistics.median(m_errors)),
            "recall": float(statistics.mean(m_recalls)),
            "count": len(clips),
        }

    # Top 3 worst clips (highest mean error) ---------------------------------
    sorted_clips = sorted(ok_clips, key=lambda c: c["mean_error"], reverse=True)
    worst_3 = [
        {
            "clip_id": c["clip_id"],
            "mean_error": c["mean_error"],
            "calibration_method": c["calibration_method"],
        }
        for c in sorted_clips[:3]
    ]

    return {
        "threshold_m": threshold,
        "num_clips_evaluated": len(ok_clips),
        "num_clips_skipped": len(per_clip) - len(ok_clips),
        "overall": {
            "mean_error": overall_mean,
            "median_error": overall_median,
            "recall": overall_recall,
        },
        "per_method": per_method,
        "worst_clips": worst_3,
        "per_clip": per_clip,
    }


def print_report(results: dict[str, Any]) -> None:
    """Print a formatted evaluation report to stdout.

    Includes overall stats, per-method breakdown, and the three worst
    clips by mean error.

    Args:
        results: Dict returned by :func:`evaluate_all`.
    """
    overall = results["overall"]
    print("=" * 64)
    print("  SoccerNet-GSR Evaluation Report")
    print("=" * 64)
    print(f"  Clips evaluated : {results['num_clips_evaluated']}")
    print(f"  Clips skipped   : {results['num_clips_skipped']}")
    print(f"  Threshold       : {results['threshold_m']:.1f} m")
    print("-" * 64)
    print(f"  {'Metric':<20s} {'Value':>10s}")
    print(f"  {'Mean error (m)':<20s} {overall['mean_error']:>10.2f}")
    print(f"  {'Median error (m)':<20s} {overall['median_error']:>10.2f}")
    print(f"  {'Recall (<threshold)':<20s} {overall['recall']:>10.2%}")
    print()

    # Per-method table -------------------------------------------------------
    per_method = results.get("per_method", {})
    if per_method:
        print("-" * 64)
        print(f"  {'Method':<14s} {'Mean(m)':>9s} {'Med(m)':>9s} {'Recall':>9s} {'Clips':>7s}")
        print("-" * 64)
        for method, stats in sorted(per_method.items()):
            print(
                f"  {method:<14s} "
                f"{stats['mean_error']:>9.2f} "
                f"{stats['median_error']:>9.2f} "
                f"{stats['recall']:>8.2%} "
                f"{stats['count']:>7d}"
            )
        print()

    # Worst clips ------------------------------------------------------------
    worst = results.get("worst_clips", [])
    if worst:
        print("-" * 64)
        print("  Worst 3 clips:")
        print(f"  {'Clip':<30s} {'Mean err (m)':>13s} {'Method':>12s}")
        print("-" * 64)
        for w in worst:
            print(
                f"  {w['clip_id']:<30s} "
                f"{w['mean_error']:>13.2f} "
                f"{w['calibration_method']:>12s}"
            )
    print("=" * 64)


def main() -> None:
    """CLI entry-point for SoccerNet-GSR evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate player-position predictions against SoccerNet-GSR ground truth.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory containing per-clip prediction outputs (default: runs).",
    )
    parser.add_argument(
        "--gsr-dir",
        type=Path,
        required=True,
        help="Directory with SoccerNet-GSR annotation JSON files.",
    )
    parser.add_argument(
        "--clips",
        nargs="*",
        default=None,
        help="Optional list of clip IDs to evaluate (default: all in gsr-dir).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Match distance threshold in metres (default: 5.0).",
    )

    args = parser.parse_args()

    results = evaluate_all(
        runs_dir=args.runs_dir,
        gsr_dir=args.gsr_dir,
        clip_ids=args.clips,
        threshold=args.threshold,
    )

    print_report(results)

    # Persist results --------------------------------------------------------
    output_path = Path("eval") / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
