"""
Step 1 — Object tracking with SAM 3.1.

SAM 3.1 takes one short noun phrase per request and tracks ≤16 objects, so we
send one request per role. Team and role come from the prompt itself ("player in
yellow"); the shirt colours of both teams, both keepers and the referee are read
from the first frame by Muse Spark.
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

_KIT_ROLES = ("team_a", "team_b", "goalkeeper_a", "goalkeeper_b", "referee")
_KIT_SCHEMA = {
    "type": "object",
    "properties": {f"{r}_shirt": {"type": "string"} for r in _KIT_ROLES},
    "required": [f"{r}_shirt" for r in _KIT_ROLES],
    "additionalProperties": False,
}


def kit_colours(client, frame: np.ndarray) -> dict[str, str]:
    """Muse Spark's 1–2 word shirt colour per role, e.g. {"team_a": "white", ...}.

    Roles it cannot see come back as "unknown". Needs a sharp frame: at ~15 px
    per player (a messenger-compressed clip) it cannot see the keepers.
    """
    r = client.spark_json(
        "This is a football match frame. Give the main shirt colour, in one or two "
        "plain words (e.g. 'white', 'light blue'), of: team A's outfield players, "
        "team B's outfield players, team A's goalkeeper, team B's goalkeeper "
        "(goalkeepers wear a different colour from their team), and the referee. "
        "Use 'unknown' if you cannot see one.",
        _KIT_SCHEMA,
        images=[jpeg(frame, max_side=max(frame.shape[:2]))],
    )
    return {role: r[f"{role}_shirt"].strip().lower() for role in _KIT_ROLES}


def build_prompts(kits: dict[str, str]) -> list[tuple[str, str, str | None]]:
    """``(prompt, cls, team)`` triples, one SAM request each.

    Tested on real footage: "football", "football player" and "goalkeeper" match
    nothing, and "soccer player in black shirt" misses a team that "player in
    black" finds, so every person is found by shirt colour.
    """
    prompts = [("soccer ball", "ball", None)]
    roles = [("team_a", "player", "A"), ("team_b", "player", "B"),
             ("goalkeeper_a", "goalkeeper", "A"), ("goalkeeper_b", "goalkeeper", "B"),
             ("referee", "referee", None)]
    used: set[str] = set()
    for role, cls, team in roles:
        colour = kits.get(role, "unknown")
        if colour == "unknown" or colour in used:
            continue  # a shared colour would track the same people twice
        used.add(colour)
        prompts.append((f"player in {colour}", cls, team))
    return prompts


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


def _iou(a: list[int], b: list[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


_DUP_IOU = 0.5        # boxes from two prompts this close are the same person
_DUP_MIN_FRAMES = 5   # shorter overlaps are two players briefly crossing


def find_duplicates(tracks: list[TracksFrame]) -> dict[tuple[int, int], list[int]]:
    """Id pairs from *different* prompts that box the same person → frames where they do.

    E.g. a light-blue keeper that "player in white" also picks up from far away.
    """
    pairs: dict[tuple[int, int], list[int]] = {}
    for tf in tracks:
        people = [o for o in tf["objects"] if o["cls"] != "ball"]
        for i, a in enumerate(people):
            for b in people[i + 1:]:
                if a["object_id"] // 1000 != b["object_id"] // 1000 and _iou(a["bbox"], b["bbox"]) > _DUP_IOU:
                    key = tuple(sorted((a["object_id"], b["object_id"])))
                    pairs.setdefault(key, []).append(tf["frame_idx"])
    return {k: v for k, v in pairs.items() if len(v) >= _DUP_MIN_FRAMES}


def drop_boxes(tracks: list[TracksFrame], object_id: int, frames: list[int]) -> None:
    """Remove *object_id* from the given frames only (it may be right elsewhere)."""
    for f in frames:
        tracks[f]["objects"] = [o for o in tracks[f]["objects"] if o["object_id"] != object_id]


def pick_shirt(client, frame: np.ndarray, bbox: list[int], colours: list[str]) -> str:
    """Ask Muse Spark which of *colours* the person in *bbox* wears."""
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1), (y2 - y1) // 2
    crop = frame[max(0, y1 - pad_y):y2 + pad_y, max(0, x1 - pad_x):x2 + pad_x]
    r = client.spark_json(
        "Which shirt colour does the football player in the centre of this crop wear?",
        {"type": "object", "properties": {"shirt": {"type": "string", "enum": colours}},
         "required": ["shirt"], "additionalProperties": False},
        images=[jpeg(crop, max_side=512)],
    )
    return r["shirt"]


def resolve_duplicates(client, video_path: Path, tracks: list[TracksFrame],
                       prompts: list[tuple[str, str, str | None]]) -> None:
    """Keep one label per doubly-detected person, chosen by Muse Spark from a crop."""
    colour_of = {pi: p.removeprefix("player in ") for pi, (p, _, _) in enumerate(prompts)}
    for (a, b), frames in find_duplicates(tracks).items():
        mid = frames[len(frames) // 2]
        box = next(o["bbox"] for o in tracks[mid]["objects"] if o["object_id"] == a)
        ca, cb = colour_of[a // 1000], colour_of[b // 1000]
        winner = pick_shirt(client, read_frame(video_path, mid), box, [ca, cb])
        loser = b if winner == ca else a
        logger.info("Ids %d/%d box the same person in %d frames; Spark says %s, dropping %d",
                    a, b, len(frames), winner, loser)
        drop_boxes(tracks, loser, frames)


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

    kits = kit_colours(client, first)
    prompts = build_prompts(kits)
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
    resolve_duplicates(client, video_path, tracks, prompts)
    interpolate_ball(tracks)

    tracks_path = out_dir / "tracks.json"
    tracks_path.write_text(json.dumps({
        "fps": fps,
        "num_frames": n_frames,
        "shirts": {"A": kits["team_a"], "B": kits["team_b"]},
        "kits": kits,
        "frames": tracks,
    }, ensure_ascii=False), encoding="utf-8")
    logger.info("Tracks written to %s (%d frames)", tracks_path, n_frames)

    _write_overlay_video(video_path, tracks, out_dir / "tracks_overlay.mp4", fps)
    return tracks_path
