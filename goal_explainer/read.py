"""
Read visual cues (body orientation, gaze, arm position, slips, reactions)
from key player crops at key moments using Muse Spark.

Step 4 of the Goal Explainer pipeline. Crops are saved to
``runs/<clip_id>/crops/cue_<player>_<moment>.jpg`` so the UI can show them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np

from .api import get_client
from .track import jpeg, read_frame

logger = logging.getLogger(__name__)


class PlayerCue(TypedDict):
    """Visual cue extracted from a player crop at a specific moment."""

    cue_id: str
    player_id: int
    moment_id: str
    frame_idx: int
    orientation: str
    gaze: str
    arm_raised: bool
    slip: bool
    reaction_delay: str
    confidence: float
    visible: bool


_CUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "orientation": {"type": "string"},
        "gaze": {"type": "string"},
        "arm_raised": {"type": "boolean"},
        "slip": {"type": "boolean"},
        "reaction_delay": {"type": "string", "enum": ["quick", "normal", "slow", "none", "unknown"]},
        "confidence": {"type": "number"},
    },
    "required": ["visible", "orientation", "gaze", "arm_raised", "slip", "reaction_delay", "confidence"],
    "additionalProperties": False,
}


def build_cue_prompt(player_id: int, moment_description: str) -> str:
    """Prompt for one player crop (image 1) plus the full frame (image 2)."""
    return (
        f"Image 1 is a close crop of football player #{player_id}; image 2 is the "
        f"full broadcast frame for context. Moment: {moment_description or 'n/a'}.\n"
        "Describe only what is visible:\n"
        "- orientation: body orientation relative to ball and own goal (short phrase)\n"
        "- gaze: where the player is looking (short phrase)\n"
        "- arm_raised: true if an arm is raised (e.g. appealing for offside)\n"
        "- slip: true if the player slips or loses balance\n"
        "- reaction_delay: how quickly the player reacts to the play\n"
        "- confidence: 0-1\n"
        "If the player is not clearly visible, set visible=false, confidence=0 "
        "and use 'unknown' for the text fields. Do not guess."
    )


def _boxes_near(tracks: list[dict[str, Any]], player_id: int, frame_idx: int, max_df: int) -> dict[str, Any] | None:
    """Closest-in-time detection of *player_id* within ±*max_df* frames."""
    best, best_df = None, max_df + 1
    for tf in tracks[max(0, frame_idx - max_df): frame_idx + max_df + 1]:
        for obj in tf["objects"]:
            if obj.get("object_id") == player_id and abs(tf["frame_idx"] - frame_idx) < best_df:
                best, best_df = (tf["frame_idx"], obj), abs(tf["frame_idx"] - frame_idx)
    return None if best is None else {"frame_idx": best[0], "bbox": best[1]["bbox"]}


def crop(frame: np.ndarray, bbox: list[int], padding: float = 0.3) -> np.ndarray:
    """Crop *bbox* with *padding* on every side, clipped to the frame."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = int((x2 - x1) * padding), int((y2 - y1) * padding)
    return frame[max(0, y1 - py):min(fh, y2 + py), max(0, x1 - px):min(fw, x2 + px)]


def run(video_path: Path, clip_id: str) -> Path:
    """Read cues for every key-moment player; write ``cues.json`` and crops."""
    run_dir = Path("runs") / clip_id
    data = json.loads((run_dir / "tracks.json").read_text(encoding="utf-8"))
    tracks: list[dict[str, Any]] = data["frames"] if isinstance(data, dict) else data
    fps = float(data.get("fps", 25.0)) if isinstance(data, dict) else 25.0
    events = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    client = get_client(clip_id)
    half_s = int(round(0.5 * fps))
    cues: list[PlayerCue] = []

    for moment in events.get("key_moments", []):
        mid, fidx = moment["moment_id"], int(moment["frame_idx"])
        for pid in moment.get("player_ids", []):
            hit = _boxes_near(tracks, pid, fidx, half_s)
            if hit is None:
                logger.warning("No box for player %s within ±0.5 s of frame %d", pid, fidx)
                continue
            frame = read_frame(video_path, hit["frame_idx"])
            img = crop(frame, hit["bbox"])
            if img.size == 0:
                continue
            cue_id = f"cue_{pid}_{mid}"
            cv2.imwrite(str(crops_dir / f"{cue_id}.jpg"), img)
            r = client.spark_json(
                build_cue_prompt(pid, moment.get("description", "")),
                _CUE_SCHEMA,
                images=[jpeg(img), jpeg(frame)],
            )
            cues.append(PlayerCue(
                cue_id=cue_id, player_id=pid, moment_id=mid, frame_idx=hit["frame_idx"],
                orientation=r["orientation"], gaze=r["gaze"], arm_raised=r["arm_raised"],
                slip=r["slip"], reaction_delay=r["reaction_delay"],
                confidence=float(r["confidence"]), visible=r["visible"],
            ))

    out_path = run_dir / "cues.json"
    out_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d cues to %s", len(cues), out_path)
    return out_path
