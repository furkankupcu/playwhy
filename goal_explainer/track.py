"""
Step 1 — Object tracking with SAM 3.1.

SAM 3.1 takes one short noun phrase per request and tracks ≤16 objects, so we
send one request per class. Team comes from the prompt itself ("player in red
shirt"); the two shirt colours are read from the first frame by Muse Spark.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, TypedDict

import cv2
import numpy as np

from .api import get_client, parse_sam_text

logger = logging.getLogger(__name__)


class TrackEntry(TypedDict, total=False):
    """Single object detection in one frame."""
    object_id: int
    cls: str              # "ball", "player", "goalkeeper", "referee"
    team: str | None      # "A", "B", or None
    bbox: list[int]       # [x1, y1, x2, y2] source pixels
    conf: float
    foot_xy: list[int]    # [x, y] — bottom-centre of bbox
    interpolated: bool    # True when added by ball interpolation


class TracksFrame(TypedDict):
    """All detections for one video frame."""
    frame_idx: int
    timestamp: float
    objects: list[TrackEntry]


_COLOURS: dict[str, tuple[int, int, int]] = {
    "ball": (0, 255, 255), "player_A": (255, 0, 0), "player_B": (0, 0, 255),
    "goalkeeper": (0, 255, 0), "referee": (0, 165, 255),
}


# ---------------------------------------------------------------------------
# Video helpers (streamed: a 20 s 1080p clip does not fit in RAM as arrays)
# ---------------------------------------------------------------------------

def video_info(video_path: Path) -> tuple[float, int]:
    """Return ``(fps, frame_count)``."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def iter_frames(video_path: Path) -> Iterator[np.ndarray]:
    """Yield BGR frames one by one."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def read_frame(video_path: Path, idx: int) -> np.ndarray:
    """Read a single frame by index."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IndexError(f"Frame {idx} not readable in {video_path}")
    return frame


def jpeg(frame: np.ndarray, max_side: int = 1280) -> bytes:
    """Encode a frame as JPEG, down-scaled so the long side is ≤ *max_side*."""
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SHIRT_SCHEMA = {
    "type": "object",
    "properties": {
        "team_a_shirt": {"type": "string"},
        "team_b_shirt": {"type": "string"},
    },
    "required": ["team_a_shirt", "team_b_shirt"],
    "additionalProperties": False,
}


def shirt_colours(client, first_frame: np.ndarray) -> tuple[str, str]:
    """Ask Muse Spark for the two outfield shirt colours as 1–2 word phrases."""
    r = client.spark_json(
        "This is a football broadcast frame. Name the main shirt colour of each "
        "team's outfield players (not goalkeepers or referees) in one or two "
        "plain words each, e.g. 'red', 'light blue', 'white'.",
        _SHIRT_SCHEMA,
        images=[jpeg(first_frame)],
    )
    return r["team_a_shirt"].strip().lower(), r["team_b_shirt"].strip().lower()


def build_prompts(colour_a: str, colour_b: str) -> list[tuple[str, str, str | None]]:
    """``(prompt, cls, team)`` triples, one SAM request each."""
    # Tested on real footage: "football"/"football player" match nothing, and
    # "soccer player in black shirt" misses a team that "player in black" finds.
    return [
        ("soccer ball", "ball", None),
        ("goalkeeper", "goalkeeper", None),
        (f"player in {colour_a}", "player", "A"),
        (f"player in {colour_b}", "player", "B"),
        ("referee", "referee", None),
    ]


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def foot_position(bbox: list[int]) -> tuple[int, int]:
    """Bottom-centre pixel of ``[x1, y1, x2, y2]`` (ground contact point)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)


def keep_best_ball(tracks: list[TracksFrame]) -> None:
    """Keep only the most confident ball per frame (SAM may return several)."""
    for tf in tracks:
        balls = [o for o in tf["objects"] if o["cls"] == "ball"]
        if len(balls) > 1:
            best = max(balls, key=lambda o: o.get("conf", 0.0))
            tf["objects"] = [o for o in tf["objects"] if o["cls"] != "ball" or o is best]


def assign_goalkeepers(tracks: list[TracksFrame], frame_w: int) -> None:
    """Give each goalkeeper the team whose players stand on the same side of frame.

    shortcut: uses screen side, wrong if the camera pans across halfway; upgrade
    to pitch-x side once homographies exist.
    """
    xs: dict[str, list[float]] = {"A": [], "B": []}
    for tf in tracks:
        for o in tf["objects"]:
            if o["cls"] == "player" and o.get("team") in xs:
                xs[o["team"]].append(o["foot_xy"][0])
    if not xs["A"] or not xs["B"]:
        return
    left, right = sorted(("A", "B"), key=lambda t: float(np.mean(xs[t])))
    for tf in tracks:
        for o in tf["objects"]:
            if o["cls"] == "goalkeeper":
                o["team"] = left if o["foot_xy"][0] < frame_w / 2 else right


def interpolate_ball(tracks: list[TracksFrame]) -> list[TracksFrame]:
    """Fill ball gaps by linear interpolation; added entries get ``interpolated``."""
    known: dict[int, list[int]] = {}
    for tf in tracks:
        for obj in tf["objects"]:
            if obj["cls"] == "ball":
                known[tf["frame_idx"]] = obj["bbox"]
                break
    if len(known) < 2:
        return tracks

    by_idx = {tf["frame_idx"]: tf for tf in tracks}
    keys = sorted(known)
    for f0, f1 in zip(keys, keys[1:]):
        if f1 - f0 <= 1:
            continue
        b0, b1 = np.array(known[f0], float), np.array(known[f1], float)
        for f in range(f0 + 1, f1):
            if f not in by_idx:
                continue
            bbox = [int(v) for v in b0 + (f - f0) / (f1 - f0) * (b1 - b0)]
            by_idx[f]["objects"].append(TrackEntry(
                object_id=-1, cls="ball", team=None, bbox=bbox, conf=0.0,
                foot_xy=list(foot_position(bbox)), interpolated=True,
            ))
    return tracks


# ---------------------------------------------------------------------------
# Overlay video
# ---------------------------------------------------------------------------

def _colour_for(entry: TrackEntry) -> tuple[int, int, int]:
    if entry["cls"] == "player":
        return _COLOURS.get(f"player_{entry.get('team')}", (255, 255, 255))
    return _COLOURS.get(entry["cls"], (255, 255, 255))


def _write_overlay_video(video_path: Path, tracks: list[TracksFrame], out_path: Path, fps: float) -> None:
    by_idx = {tf["frame_idx"]: tf["objects"] for tf in tracks}
    writer = None
    for i, frame in enumerate(iter_frames(video_path)):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
        for obj in by_idx.get(i, []):
            x1, y1, x2, y2 = obj["bbox"]
            colour = _colour_for(obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label = f"#{obj['object_id']}" + (" (i)" if obj.get("interpolated") else "")
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        writer.write(frame)
    if writer is not None:
        writer.release()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(video_path: Path, clip_id: str) -> Path:
    """Track all classes and write ``tracks.json`` + ``tracks_overlay.mp4``."""
    out_dir = Path("runs") / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fps, n_frames = video_info(video_path)
    first = read_frame(video_path, 0)
    client = get_client(clip_id)

    colour_a, colour_b = shirt_colours(client, first)
    prompts = build_prompts(colour_a, colour_b)
    file_id = client.upload_file(video_path)

    tracks: list[TracksFrame] = [
        TracksFrame(frame_idx=i, timestamp=i / fps, objects=[]) for i in range(n_frames)
    ]
    for pi, (prompt, cls, team) in enumerate(prompts):
        dets = parse_sam_text(client.sam(prompt, video_file_id=file_id))
        logger.info("SAM %r: %d detections", prompt, len(dets))
        for d in dets:
            if not 0 <= d["frame"] < n_frames:
                continue
            tracks[d["frame"]]["objects"].append(TrackEntry(
                # ids are per request, so offset them to stay unique across prompts
                object_id=pi * 1000 + d["id"], cls=cls, team=team, bbox=d["bbox"],
                conf=d["conf"], foot_xy=list(foot_position(d["bbox"])),
            ))

    keep_best_ball(tracks)
    assign_goalkeepers(tracks, first.shape[1])
    interpolate_ball(tracks)

    tracks_path = out_dir / "tracks.json"
    tracks_path.write_text(json.dumps({
        "fps": fps,
        "num_frames": n_frames,
        "shirts": {"A": colour_a, "B": colour_b},
        "frames": tracks,
    }, ensure_ascii=False), encoding="utf-8")
    logger.info("Tracks written to %s (%d frames)", tracks_path, n_frames)

    _write_overlay_video(video_path, tracks, out_dir / "tracks_overlay.mp4", fps)
    return tracks_path
