"""
FastAPI web app for Goal Explainer.

Run from the repo root (the pipeline writes to ``./runs``):

    uvicorn server:app --port 8000

Upload a clip → the pipeline runs in a background thread → the page polls
``/api/clips/<id>/status`` and then renders ``/api/clips/<id>/result``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from eval.calib_eval import ground_error_m
from goal_explainer.explain import answer_question
from goal_explainer.pipeline import STEPS, run_pipeline
from goal_explainer.pitch import PITCH_KEYPOINTS
from goal_explainer.track import video_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUNS = Path("runs")
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_SECONDS = 25.0  # scope is ≤20 s clips; a little slack for trimming

_CLIP_ID = re.compile(r"^[0-9a-f]{16}$")
_CUE_ID = re.compile(r"^cue_[0-9-]+_moment_[0-9]+$")

app = FastAPI(title="Goal Explainer")
_running: set[str] = set()
_lock = threading.Lock()


def _run_dir(clip_id: str) -> Path:
    if not _CLIP_ID.match(clip_id):  # ids become paths: never trust them
        raise HTTPException(404, "Unknown clip")
    d = RUNS / clip_id
    if not (d / "input.mp4").is_file():
        raise HTTPException(404, "Unknown clip")
    return d


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _start(clip_id: str, manual_points: list[dict[str, Any]] | None = None, skip_to: int = 1) -> bool:
    """Start the pipeline in a thread unless one is already running for this clip."""
    with _lock:
        if clip_id in _running:
            return False
        _running.add(clip_id)

    def work() -> None:
        try:
            run_pipeline(RUNS / clip_id / "input.mp4", clip_id, manual_points, skip_to)
        except Exception:  # noqa: BLE001 — run_pipeline already logs step errors
            logger.exception("Pipeline crashed for %s", clip_id)
        finally:
            with _lock:
                _running.discard(clip_id)

    threading.Thread(target=work, daemon=True).start()
    return True


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/api/clips")
async def upload_clip(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 300 MB.")
    if not data:
        raise HTTPException(400, "Empty file.")

    # Content hash as id: same clip → same run (and cached API calls); no name clashes.
    clip_id = hashlib.sha256(data).hexdigest()[:16]
    run_dir = RUNS / clip_id
    run_dir.mkdir(parents=True, exist_ok=True)
    video = run_dir / "input.mp4"
    if not video.is_file():
        video.write_bytes(data)

    try:
        fps, n = video_info(video)
    except FileNotFoundError:
        video.unlink(missing_ok=True)
        raise HTTPException(400, "Could not read this video. Upload an MP4 or MOV (H.264/H.265).")
    if n <= 0 or n / fps > MAX_SECONDS:
        video.unlink(missing_ok=True)
        raise HTTPException(400, f"Clip must be at most {MAX_SECONDS:.0f} seconds long.")

    status = _read_json(run_dir / "status.json")
    if not (status and status.get("ok") and status["steps"][-1]["status"] == "done"):
        _start(clip_id)
    return {"clip_id": clip_id}


@app.get("/api/clips/{clip_id}/status")
def clip_status(clip_id: str) -> dict[str, Any]:
    run_dir = _run_dir(clip_id)
    status = _read_json(run_dir / "status.json") or {
        "clip_id": clip_id, "ok": True, "failed_step": None,
        "steps": [{"step": i + 1, "name": s, "status": "pending"} for i, s in enumerate(STEPS)],
    }
    with _lock:
        status["running"] = clip_id in _running
    return status


class Point(BaseModel):
    px: list[float] = Field(min_length=2, max_length=2)
    pitch: list[float] = Field(min_length=2, max_length=2)


class CalibrationRequest(BaseModel):
    points: list[Point] = Field(min_length=4, max_length=12)


@app.post("/api/clips/{clip_id}/calibrate")
def calibrate_clip(clip_id: str, body: CalibrationRequest) -> dict[str, Any]:
    run_dir = _run_dir(clip_id)
    if not (run_dir / "tracks.json").is_file():
        raise HTTPException(409, "Tracking has not finished yet.")
    if not _start(clip_id, [p.model_dump() for p in body.points], skip_to=2):
        raise HTTPException(409, "Analysis is already running for this clip.")
    return {"clip_id": clip_id}


@app.post("/api/clips/{clip_id}/reference")
def calibration_reference(clip_id: str, body: CalibrationRequest) -> dict[str, Any]:
    """Compare the automatic fit of frame 0 with hand-marked points; changes no analysis result."""
    run_dir = _run_dir(clip_id)
    homographies = _read_json(run_dir / "homography.json")
    if not homographies:
        raise HTTPException(409, "Calibration has not finished yet.")
    points = [p.model_dump() for p in body.points]
    H_ref, mask = cv2.findHomography(np.array([p["pitch"] for p in points], np.float64),
                                     np.array([p["px"] for p in points], np.float64), cv2.RANSAC, 10.0)
    if H_ref is None:
        raise HTTPException(422, "These points do not define a camera. Spread them out and avoid straight lines.")
    cap = cv2.VideoCapture(str(run_dir / "input.mp4"))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    errors = ground_error_m(np.array(homographies[0]["H"], np.float64), H_ref, size)
    if not len(errors):
        raise HTTPException(422, "These points map no visible pixel onto the pitch. Check their names.")
    result = {"median_m": round(float(np.median(errors)), 2), "p90_m": round(float(np.percentile(errors, 90)), 2),
              "points_used": int(mask.sum()), "points_given": len(points)}
    (run_dir / "calib_reference.json").write_text(json.dumps({**result, "points": points}, indent=2), encoding="utf-8")
    return result


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=300)


@app.post("/api/clips/{clip_id}/ask")
def ask_clip(clip_id: str, body: Question) -> dict[str, Any]:
    run_dir = _run_dir(clip_id)
    if not (run_dir / "explanation.json").is_file():
        raise HTTPException(409, "Analysis has not finished.")
    try:
        return answer_question(clip_id, body.question)
    except Exception as exc:  # noqa: BLE001 - the viewer sees a message, the log keeps the cause
        logger.exception("Question failed for %s", clip_id)
        raise HTTPException(502, "Muse Spark could not answer just now. Try again.") from exc


@app.get("/api/clips/{clip_id}/result")
def clip_result(clip_id: str) -> dict[str, Any]:
    run_dir = _run_dir(clip_id)
    explanation = _read_json(run_dir / "explanation.json")
    if explanation is None:
        raise HTTPException(409, "Analysis has not finished.")
    fps, n = video_info(run_dir / "input.mp4")
    homographies = _read_json(run_dir / "homography.json") or []
    tracks = _read_json(run_dir / "tracks.json")
    shirts = tracks.get("shirts") if isinstance(tracks, dict) else None
    reliable = sum(1 for h in homographies if h.get("reliable"))
    return {
        "clip_id": clip_id,
        "fps": fps,
        "duration": n / fps,
        "reliable_share": reliable / len(homographies) if homographies else None,
        "shirts": shirts,
        "explanation": explanation,
        "events": _read_json(run_dir / "events.json") or {},
        "cues": _read_json(run_dir / "cues.json") or [],
        "pitch": _read_json(run_dir / "pitch.json") or [],
        # per-frame pitch → image homography, so the page can draw graphs on the video
        "homographies": [h["H"] for h in homographies],
        "graphs": _read_json(run_dir / "graphs.json"),
        "graph_video": (run_dir / "graphs.mp4").is_file(),
    }


@app.get("/api/clips/{clip_id}/video")
def clip_video(clip_id: str) -> FileResponse:
    return FileResponse(_run_dir(clip_id) / "input.mp4", media_type="video/mp4")


@app.get("/api/clips/{clip_id}/graphs.mp4")
def graph_video(clip_id: str) -> FileResponse:
    path = _run_dir(clip_id) / "graphs.mp4"
    if not path.is_file():
        raise HTTPException(404, "No graph video for this clip")
    return FileResponse(path, media_type="video/mp4", filename=f"goal-explainer-{clip_id}.mp4")


@app.get("/api/clips/{clip_id}/frame0.jpg")
def first_frame(clip_id: str) -> Response:
    cap = cv2.VideoCapture(str(_run_dir(clip_id) / "input.mp4"))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, "Could not read the first frame.")
    return Response(cv2.imencode(".jpg", frame)[1].tobytes(), media_type="image/jpeg")


@app.get("/api/clips/{clip_id}/crops/{cue_id}.jpg")
def cue_crop(clip_id: str, cue_id: str) -> FileResponse:
    path = _run_dir(clip_id) / "crops" / f"{cue_id}.jpg"
    if not _CUE_ID.match(cue_id) or not path.is_file():
        raise HTTPException(404, "No crop")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/pitch/keypoints")
def pitch_keypoints() -> dict[str, list[float]]:
    return {k: list(v) for k, v in PITCH_KEYPOINTS.items()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", media_type="text/html")
