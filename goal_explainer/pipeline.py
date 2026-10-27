"""
CLI entry point: ``python -m goal_explainer.pipeline clip.mp4``

Runs every analysis step in order. If a step fails the remaining steps are
skipped (they would read stale or missing files). Progress is written to
``runs/<clip_id>/status.json`` after every step, so a UI can poll it.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from . import calibrate, explain, geometry, graphs, read, track

STEPS: list[str] = ["track", "calibrate", "geometry", "read", "explain", "graphs"]


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def run_pipeline(
    video_path: Path,
    clip_id: str,
    manual_points: list[dict[str, Any]] | None = None,
    skip_to: int = 1,
) -> dict[str, Any]:
    """Run steps ``skip_to``…6 and return the status log.

    ``manual_points`` are ``{"px": [x, y], "pitch": [x_m, y_m]}`` dicts for
    calibration path (c). Steps before ``skip_to`` reuse their files on disk.
    """
    run_dir = Path("runs") / clip_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Work on a copy inside the run dir; refresh it so a new clip never meets stale outputs
    input_video = run_dir / "input.mp4"
    if Path(video_path).resolve() != input_video.resolve():
        shutil.copy2(str(video_path), str(input_video))
    video_path = input_video

    log: dict[str, Any] = {"clip_id": clip_id, "ok": True, "failed_step": None, "steps": []}
    status_path = run_dir / "status.json"

    def save() -> None:
        status_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    for num, name in enumerate(STEPS, start=1):
        entry: dict[str, Any] = {"step": num, "name": name, "status": "pending", "elapsed_s": 0.0, "error": None}
        log["steps"].append(entry)

        if num < skip_to or not log["ok"]:
            entry["status"] = "skipped"
            save()
            continue

        print(f"[{_ts()}] Step {num}/{len(STEPS)} ({name}) — starting")
        entry["status"] = "running"
        save()
        t0 = time.perf_counter()
        try:
            if name == "track":
                track.run(video_path, clip_id)
            elif name == "calibrate":
                calibrate.run(video_path, clip_id, run_dir / "tracks.json", manual_points)
            elif name == "geometry":
                geometry.run(clip_id)
            elif name == "read":
                read.run(video_path, clip_id)
            elif name == "explain":
                explain.run(video_path, clip_id)
            elif name == "graphs":
                graphs.run(video_path, clip_id)
            entry["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            log["ok"] = False
            log["failed_step"] = name
            traceback.print_exc()
        entry["elapsed_s"] = round(time.perf_counter() - t0, 3)
        save()
        print(f"[{_ts()}]   {name} → {entry['status']} ({entry['elapsed_s']:.1f}s)")

    print(f"[{_ts()}] Pipeline finished (ok={log['ok']}) — {status_path}")
    return log


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="goal_explainer.pipeline",
        description="Run the Goal Explainer analysis pipeline on a football clip.",
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--clip-id", default=None, help="Clip identifier (default: file name stem).")
    parser.add_argument(
        "--manual-points", default=None,
        help='JSON list, e.g. \'[{"px": [100, 200], "pitch": [88.5, 13.84]}, ...]\' (4+ points).',
    )
    parser.add_argument("--skip-to", type=int, default=1, help="Step number to resume from (1–6).")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        sys.exit(f"Error: video file not found — {video_path}")

    manual_points = None
    if args.manual_points:
        try:
            manual_points = json.loads(args.manual_points)
        except json.JSONDecodeError as exc:
            sys.exit(f"Error: invalid --manual-points JSON — {exc}")

    log = run_pipeline(video_path, args.clip_id or video_path.stem, manual_points, args.skip_to)
    sys.exit(0 if log["ok"] else 1)


if __name__ == "__main__":
    main()
