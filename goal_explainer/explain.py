"""
Generate evidence-based tactical explanations from events and cues.
Every claim must reference specific evidence from the data.

Step 5 of the Goal Explainer pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import cv2

from .api import get_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class Evidence(TypedDict):
    """A single piece of evidence backing a claim."""

    type: str  # "event" | "cue"
    id: str  # event_id or "cue_{player_id}_{moment_id}"
    description: str


class Claim(TypedDict):
    """A single tactical claim about the goal."""

    claim_id: int
    text: str
    t_start: float
    t_end: float
    player_ids: list[int]
    evidence: list[Evidence]


class Explanation(TypedDict):
    """Full explanation for a clip."""

    clip_id: str
    claims: list[Claim]
    summary: str


_EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "integer"},
                "text": {"type": "string"},
                "t_start": {"type": "number"},
                "t_end": {"type": "number"},
                "player_ids": {"type": "array", "items": {"type": "integer"}},
                "evidence": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["event", "cue"]},
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["type", "id", "description"],
                    "additionalProperties": False,
                }},
            },
            "required": ["claim_id", "text", "t_start", "t_end", "player_ids", "evidence"],
            "additionalProperties": False,
        }},
    },
    "required": ["summary", "claims"],
    "additionalProperties": False,
}

# A claim's time window must lie within this many seconds of one of its evidence items.
_EVIDENCE_SLACK_S = 1.5


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def player_names(tracks: dict[str, Any], cues: list[dict[str, Any]]) -> dict[int, str]:
    """Human names for tracked ids: "blue #6", "blue keeper", or "blue player 3".

    Shirt colours come from tracks.json (read by Muse Spark), numbers from the cues
    (Muse Spark read them off the crops). Ids are internal and mean nothing to a reader.
    """
    from collections import Counter

    kits = tracks.get("kits") or {}
    colour = {"A": kits.get("team_a") or tracks.get("shirts", {}).get("A", "team A"),
              "B": kits.get("team_b") or tracks.get("shirts", {}).get("B", "team B")}
    numbers: dict[int, Counter] = {}
    for c in cues:
        n = str(c.get("jersey_number", "")).strip()
        if n.isdigit() and c.get("visible"):
            numbers.setdefault(c["player_id"], Counter())[n] += 1
    first_seen: dict[int, tuple[str, str]] = {}
    for f in tracks.get("frames", []):
        for o in f["objects"]:
            if o["cls"] in ("player", "goalkeeper") and o.get("team") in colour:
                first_seen.setdefault(o["object_id"], (o["team"], o["cls"]))
    names: dict[int, str] = {}
    count: Counter = Counter()
    for pid, (team, cls) in first_seen.items():
        if pid in numbers:
            names[pid] = f"{colour[team]} #{numbers[pid].most_common(1)[0][0]}"
        elif cls == "goalkeeper":
            names[pid] = f"{colour[team]} keeper"
        else:
            count[team] += 1
            names[pid] = f"{colour[team]} player {count[team]}"
    return names


def build_explanation_prompt(
    events: dict[str, Any],
    cues: list[dict[str, Any]],
    names: dict[int, str] | None = None,
) -> str:
    """Build a prompt with all events and cues data for explanation generation.

    Asks Muse Spark to produce 3–6 tactical claims about why the goal happened,
    each grounded in specific evidence from the provided data.
    """
    events_json = json.dumps(events, indent=2, ensure_ascii=False)
    cues_json = json.dumps(cues, indent=2, ensure_ascii=False)

    # Build a lookup of valid evidence IDs so the model knows what to cite
    valid_event_ids: list[str] = []
    for key in ("events", "passes", "shots", "goals", "key_moments", "pass_chain"):
        for item in events.get(key, []):
            eid = item.get("event_id") if isinstance(item, dict) else None
            mid = item.get("moment_id") if isinstance(item, dict) else None
            if eid is not None:
                valid_event_ids.append(str(eid))
            if mid is not None:
                valid_event_ids.append(str(mid))
    valid_event_ids = list(dict.fromkeys(valid_event_ids))

    valid_cue_ids = [_cue_id(c) for c in cues]

    return (
        "You are an expert football analyst. Below are structured data about a "
        "goal: detected events and visual player cues.\n\n"
        "=== EVENTS ===\n"
        f"{events_json}\n\n"
        "=== PLAYER CUES ===\n"
        f"{cues_json}\n\n"
        "=== VALID EVIDENCE IDs ===\n"
        f"Event IDs: {valid_event_ids}\n"
        f"Cue IDs: {valid_cue_ids}\n\n"
        + (f"=== PLAYER NAMES (id: name) ===\n{json.dumps({str(k): v for k, v in names.items()})}\n"
           "In the claim text and summary refer to players ONLY by these names, never by id; "
           "keep the numeric ids in player_ids.\n\n" if names else "")
        + "Write 3-6 tactical claims explaining WHY this goal happened. "
        "Each claim must be a concise, insightful observation about tactics, "
        "positioning, decision-making, or execution.\n\n"
        "For each claim give t_start/t_end (seconds in the clip), the player_ids "
        "involved and the evidence items it rests on, plus a one-paragraph summary.\n\n"
        "RULES:\n"
        "- Do NOT invent events. Every claim must cite at least one evidence "
        "item from the data provided.\n"
        "- Evidence IDs MUST come from the VALID EVIDENCE IDs listed above.\n"
        "- t_start and t_end must be in seconds within the clip duration.\n"
        "- t_start must be strictly less than t_end.\n"
        "- Each claim should reference specific players by their IDs.\n"
        "- A cue with visible=false or confidence < 0.5 is not evidence."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _collect_valid_ids(
    events: dict[str, Any], cues: list[dict[str, Any]]
) -> set[str]:
    """Collect all valid event and cue IDs from the source data."""
    ids: set[str] = set()
    for key in ("events", "passes", "shots", "goals", "key_moments", "pass_chain"):
        for item in events.get(key, []):
            if isinstance(item, dict):
                eid = item.get("event_id")
                mid = item.get("moment_id")
                if eid is not None:
                    ids.add(str(eid))
                if mid is not None:
                    ids.add(str(mid))
    for cue in cues:
        ids.add(_cue_id(cue))
    return ids


def _cue_id(cue: dict[str, Any]) -> str:
    return cue.get("cue_id") or f"cue_{cue.get('player_id')}_{cue.get('moment_id')}"


def _evidence_times(events: dict[str, Any], cues: list[dict[str, Any]]) -> dict[str, float]:
    """Timestamp (s) of every evidence id that has one; cues take their moment's time."""
    times: dict[str, float] = {}
    for key in ("passes", "shots", "goals", "key_moments", "pass_chain"):
        for item in events.get(key, []):
            if isinstance(item, dict) and "timestamp" in item:
                for k in ("event_id", "moment_id"):
                    if item.get(k) is not None:
                        times[str(item[k])] = float(item["timestamp"])
    for cue in cues:
        t = times.get(str(cue.get("moment_id")))
        if t is not None:
            times[_cue_id(cue)] = t
    return times


def validate_claim(
    claim: dict[str, Any],
    events: dict[str, Any],
    cues: list[dict[str, Any]],
    clip_duration: float,
) -> tuple[bool, str]:
    """Validate a single claim against the source data and clip duration.

    Checks:
    - ``t_start`` and ``t_end`` are within ``[0, clip_duration]``.
    - ``t_start < t_end``.
    - Every evidence item references a real event or cue ID.
    - At least one evidence item exists.

    Returns:
        ``(is_valid, reason_if_invalid)``
    """
    t_start = claim.get("t_start", -1)
    t_end = claim.get("t_end", -1)

    if not (0 <= t_start <= clip_duration):
        return False, f"t_start={t_start} out of range [0, {clip_duration}]"
    if not (0 <= t_end <= clip_duration):
        return False, f"t_end={t_end} out of range [0, {clip_duration}]"
    if t_start >= t_end:
        return False, f"t_start ({t_start}) must be < t_end ({t_end})"

    evidence_list: list[dict[str, Any]] = claim.get("evidence", [])
    if not evidence_list:
        return False, "Claim has no evidence items"

    valid_ids = _collect_valid_ids(events, cues)
    for ev in evidence_list:
        ev_id = str(ev.get("id", ""))
        if ev_id not in valid_ids:
            return False, f"Evidence ID '{ev_id}' not found in source data"

    # The claim must point at the moment its evidence happened
    times = _evidence_times(events, cues)
    ev_times = [times[str(ev["id"])] for ev in evidence_list if str(ev["id"]) in times]
    if ev_times and not any(
        t_start - _EVIDENCE_SLACK_S <= t <= t_end + _EVIDENCE_SLACK_S for t in ev_times
    ):
        return False, f"claim window {t_start}-{t_end}s is far from its evidence ({ev_times})"

    return True, ""


def validate_explanation(
    explanation: dict[str, Any],
    events: dict[str, Any],
    cues: list[dict[str, Any]],
    clip_duration: float,
) -> dict[str, Any]:
    """Validate all claims in an explanation, removing invalid ones.

    Invalid claims are logged as warnings and dropped from the output.

    Returns:
        A cleaned copy of the explanation dict with only valid claims.
    """
    raw_claims: list[dict[str, Any]] = explanation.get("claims", [])
    valid_claims: list[dict[str, Any]] = []

    for claim in raw_claims:
        ok, reason = validate_claim(claim, events, cues, clip_duration)
        if ok:
            valid_claims.append(claim)
        else:
            logger.warning(
                "Removed invalid claim %s: %s",
                claim.get("claim_id", "?"),
                reason,
            )

    cleaned = dict(explanation)
    cleaned["claims"] = valid_claims
    return cleaned


# ---------------------------------------------------------------------------
# Video duration helper
# ---------------------------------------------------------------------------


def _get_clip_duration(video_path: Path) -> float:
    """Return the duration of the video clip in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return float(frame_count) / fps


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run(video_path: Path, clip_id: str) -> Path:
    """Execute Step 5: generate evidence-based tactical explanations.

    Reads:
        ``runs/<clip_id>/events.json`` — detected events.
        ``runs/<clip_id>/cues.json``   — player visual cues.

    Writes:
        ``runs/<clip_id>/explanation.json`` — validated :class:`Explanation`.

    Returns:
        The path to the written ``explanation.json`` file.
    """
    run_dir = Path("runs") / clip_id
    events_path = run_dir / "events.json"
    cues_path = run_dir / "cues.json"

    events: dict[str, Any] = json.loads(events_path.read_text(encoding="utf-8"))
    cues: list[dict[str, Any]] = json.loads(cues_path.read_text(encoding="utf-8"))

    clip_duration = _get_clip_duration(video_path)
    logger.info("Clip duration: %.2f s", clip_duration)

    tracks = json.loads((run_dir / "tracks.json").read_text(encoding="utf-8"))
    names = player_names(tracks, cues)
    prompt = build_explanation_prompt(events, cues, names)

    explanation = get_client(clip_id).spark_json(prompt, _EXPLANATION_SCHEMA)
    explanation["clip_id"] = clip_id
    explanation["player_names"] = {str(k): v for k, v in names.items()}  # for the UI labels
    explanation = validate_explanation(explanation, events, cues, clip_duration)
    times = _evidence_times(events, cues)
    for claim in explanation["claims"]:
        for ev in claim["evidence"]:
            ev["t"] = times.get(str(ev["id"]))  # lets the UI jump to the evidence

    logger.info(
        "Explanation has %d valid claims after validation",
        len(explanation.get("claims", [])),
    )

    out_path = run_dir / "explanation.json"
    out_path.write_text(
        json.dumps(explanation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote explanation to %s", out_path)
    return out_path
