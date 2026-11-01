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
    jersey_number: str  # "" when not readable


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
        "jersey_number": {"type": "string"},
    },
    "required": ["visible", "orientation", "gaze", "arm_raised", "slip", "reaction_delay", "confidence",
                 "jersey_number"],
    "additionalProperties": False,
}


def build_cue_prompt(player_id: int, moment_description: str) -> str:
    """Prompt for one context crop (image 1, player boxed in yellow) plus the full frame (image 2)."""
    return (
        "Image 1 is an enlarged crop from a football match video; the player to describe "
        "is inside the yellow box, with nearby players and the ball for context. Image 2 "
        f"is the full frame. Moment: {moment_description or 'n/a'}.\n"
        "The footage is low resolution, so give your best reading and lower the confidence "
        "rather than refusing:\n"
        "- orientation: body orientation relative to the ball and the player's own goal (short phrase)\n"
        "- gaze: where the player is looking (short phrase)\n"
        "- arm_raised: true if an arm is raised (e.g. appealing for offside)\n"
        "- slip: true if the player slips or loses balance\n"
        "- reaction_delay: how quickly the player reacts to the play\n"
        "- jersey_number: the shirt number if you can read it, else empty\n"
        "- confidence: 0-1 for the whole reading\n"
        "Set visible=false only if the boxed player is hidden behind others or cut off."
    )


def _boxes_near(tracks: list[dict[str, Any]], player_id: int, frame_idx: int, max_df: int) -> dict[str, Any] | None:
    """Closest-in-time detection of *player_id* within ±*max_df* frames."""
    best, best_df = None, max_df + 1
    for tf in tracks[max(0, frame_idx - max_df): frame_idx + max_df + 1]:
        for obj in tf["objects"]:
            if obj.get("object_id") == player_id and abs(tf["frame_idx"] - frame_idx) < best_df:
                best, best_df = (tf["frame_idx"], obj), abs(tf["frame_idx"] - frame_idx)
    return None if best is None else {"frame_idx": best[0], "bbox": best[1]["bbox"]}


_CONTEXT = 1.5    # crop this many box sizes around the player: ball and neighbours in view
_MIN_SIDE = 384   # small players are upscaled so the model has pixels to read


def crop(frame: np.ndarray, bbox: list[int], padding: float = _CONTEXT) -> np.ndarray:
    """Context crop around *bbox*, player boxed in yellow, upscaled to ≥ _MIN_SIDE px."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = int((x2 - x1) * padding), int((y2 - y1) * padding)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    out = frame[cy1:min(fh, y2 + py), cx1:min(fw, x2 + px)].copy()
    if out.size == 0:
        return out
    cv2.rectangle(out, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 255), 1)
    s = _MIN_SIDE / max(out.shape[:2])
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else out


_GOAL_PROMPT = (
    "This is a short football clip. Did a goal get scored in it? Answer with one JSON object "
    'and nothing else: {"goal": true|false, "t_shot": <seconds from the start when the '
    'scoring shot or header is struck>, "t_goal": <seconds when the ball crosses the line>, '
    '"description": "<one sentence: how the goal was scored, e.g. low shot from the edge of '
    'the box into the near corner>"}. Use null times if there is no goal.'
)


# A later shot this far past Muse Spark's goal time means it named the wrong moment.
_GOAL_SLACK_S = 2.0


def find_goal(client, video_path: Path, duration: float) -> dict[str, Any] | None:
    """Ask Muse Spark (watching the clip) whether and when a goal is scored.

    The ball is often lost in the crowded box in the last seconds, so the goal
    cannot be read off the tracked positions alone.
    """
    try:
        r = client.spark_video_json(_GOAL_PROMPT, client.upload_file(video_path))
    except Exception:  # noqa: BLE001 — a missing goal must not stop the cue reading
        logger.warning("Goal question to Muse Spark failed", exc_info=True)
        return None
    if not r.get("goal") or r.get("t_shot") is None:
        return None
    t_shot = float(r["t_shot"])
    t_goal = float(r["t_goal"]) if r.get("t_goal") is not None else t_shot
    if not 0 <= t_shot <= t_goal <= duration + 0.5:
        logger.warning("Goal times out of range: %s", r)
        return None
    return {"event_id": "goal_0", "timestamp": t_goal, "t_shot": t_shot,
            "description": str(r.get("description", ""))}


def reconcile_goal(goal: dict[str, Any], events: dict[str, Any], pitch: list[dict[str, Any]]) -> dict[str, Any]:
    """Move the goal if Muse Spark's time comes well before a later shot of the attacking team.

    Muse Spark can name an earlier moment of the same attack (17 s for a goal at 23 s when the clip
    ends right after it). An earlier saved shot does not count against its time: only a later one does.
    """
    from collections import Counter
    teams = Counter(p["team"] for p in events.get("passes", [])).most_common(1)
    team_at = {(f["frame_idx"], p["player_id"]): p["team"] for f in pitch for p in f["players"]}
    shots = [s for s in events.get("shots", [])
             if not teams or team_at.get((s["frame_idx"], s["shooter_id"])) == teams[0][0]]
    if not shots or max(s["timestamp"] for s in shots) <= goal["timestamp"] + _GOAL_SLACK_S:
        return goal
    shot = max(shots, key=lambda s: s["timestamp"])
    gx, gy = shot["goal_xy"]
    after = [(p["timestamp"], (p["x_m"] - gx) ** 2 + (p["y_m"] - gy) ** 2) for f in pitch for p in f["players"]
             if p["team"] == "ball" and shot["timestamp"] < p["timestamp"] <= shot["timestamp"] + _GOAL_SLACK_S]
    t_goal = min(after, key=lambda a: a[1])[0] if after else shot["timestamp"] + 1.0
    logger.warning("Goal moved from %.1fs to %.1fs: tracking shows a later shot", goal["timestamp"], t_goal)
    return {**goal, "timestamp": t_goal, "t_shot": shot["timestamp"]}


def drop_after_goal(events: dict[str, Any], t_goal: float) -> None:
    """Play after the goal cannot explain it; kick-off walks also read as passes."""
    for key in ("passes", "shots", "key_moments", "pass_chain", "offside"):
        if key in events:
            events[key] = [e for e in events[key] if not isinstance(e, dict) or e.get("timestamp", 0) <= t_goal + 0.5]


def players_near_goal(pitch: list[dict[str, Any]], frame_idx: int, goal_x: float, k: int = 3) -> list[int]:
    """The *k* people (any team, keepers included) nearest the attacked goal at *frame_idx*."""
    rows = next((f["players"] for f in pitch if f["frame_idx"] == frame_idx), [])
    people = [p for p in rows if p["team"] in ("A", "B")]
    people.sort(key=lambda p: (p["x_m"] - goal_x) ** 2 + (p["y_m"] - 34) ** 2)
    return [p["player_id"] for p in people[:k]]


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

    goal = find_goal(client, video_path, len(tracks) / fps)
    if goal is not None:
        pitch = json.loads((run_dir / "pitch.json").read_text(encoding="utf-8"))
        goal = reconcile_goal(goal, events, pitch)
        drop_after_goal(events, goal["timestamp"])
        events["goals"] = [goal]
        shot_frame = min(int(round(goal["t_shot"] * fps)), len(tracks) - 1)
        goal_x = 105.0 if events.get("attacking_direction") != "left" else 0.0
        # re-runs must not stack goal moments
        events["key_moments"] = [m for m in events.get("key_moments", []) if m.get("event_type") != "goal"]
        events["key_moments"].append({
            "moment_id": f"moment_{1 + max((int(m['moment_id'].split('_')[1]) for m in events.get('key_moments', [])), default=-1)}",
            "frame_idx": shot_frame, "timestamp": goal["t_shot"], "event_type": "goal",
            "player_ids": players_near_goal(pitch, shot_frame, goal_x),
            "description": f"The scoring shot. {goal['description']}",
        })
        (run_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
        logger.info("Muse Spark: goal at %.1fs (shot %.1fs): %s", goal["timestamp"], goal["t_shot"], goal["description"])
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
                jersey_number=r["jersey_number"].strip(),
            ))

    out_path = run_dir / "cues.json"
    out_path.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d cues to %s", len(cues), out_path)
    return out_path
