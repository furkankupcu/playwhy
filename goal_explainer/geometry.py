"""
Geometry analysis: convert pixel tracks + homography into pitch-space positions,
detect passes, offside, shots, open lanes, and key moments.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlayerPosition:
    """A single player's position on the pitch for one frame."""
    frame_idx: int
    timestamp: float
    player_id: int
    team: str
    x_m: float
    y_m: float
    estimated: bool = False  # ball only: ground track interpolated between touches

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PossessionEvent:
    """Records which player holds the ball at a given frame."""
    frame_idx: int
    timestamp: float
    player_id: int
    team: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PassEvent:
    """A detected pass between two players on the same team."""
    event_id: int
    frame_idx: int
    timestamp: float
    passer_id: int
    receiver_id: int
    team: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    success: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_id"] = f"pass_{self.event_id}"  # namespaced: ids must not collide across kinds
        d["start_xy"] = list(self.start_xy)
        d["end_xy"] = list(self.end_xy)
        return d


@dataclass
class OffsideSnapshot:
    """Offside analysis at the moment of a pass."""
    frame_idx: int
    timestamp: float
    offside_line_x: float
    attacking_players_x: list[float]
    is_offside: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShotEvent:
    """A detected shot on goal."""
    event_id: int
    frame_idx: int
    timestamp: float
    shooter_id: int
    distance_m: float
    angle_deg: float
    goal_xy: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_id"] = f"shot_{self.event_id}"
        d["goal_xy"] = list(self.goal_xy)
        return d


@dataclass
class OpenLane:
    """Whether a passing lane between two teammates is clear of opponents."""
    frame_idx: int
    from_id: int
    to_id: int
    team: str
    lane_clear: bool
    nearest_opponent_dist: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KeyMoment:
    """A significant tactical event in the build-up to the goal."""
    moment_id: int
    frame_idx: int
    timestamp: float
    event_type: str          # "pass", "shot", "lost_man", "run"
    player_ids: list[int] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["moment_id"] = f"moment_{self.moment_id}"
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def point_to_segment_distance(
    p: np.ndarray | tuple[float, float] | list[float],
    a: np.ndarray | tuple[float, float] | list[float],
    b: np.ndarray | tuple[float, float] | list[float],
) -> float:
    """Return the shortest distance from point *p* to line segment *a*–*b*.

    Accepts 2-element numpy arrays, tuples, or lists.
    """
    p_arr = np.asarray(p, dtype=float)
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    ab = b_arr - a_arr
    ap = p_arr - a_arr
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return float(np.linalg.norm(ap))
    t = float(np.dot(ap, ab)) / ab_len_sq
    t = max(0.0, min(1.0, t))
    proj = a_arr + t * ab
    return float(np.linalg.norm(p_arr - proj))


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def pixel_to_pitch(px: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """Convert pixel coordinates to pitch metres using a homography.

    *H* maps pitch → image, so we invert it first.

    Args:
        px: ``(x_pixel, y_pixel)`` in image space.
        H:  3×3 homography matrix (pitch → image).

    Returns:
        ``(x_m, y_m)`` in pitch-space metres.
    """
    H_inv = np.linalg.inv(H)
    pt = np.array([px[0], px[1], 1.0], dtype=np.float64)
    dst = H_inv @ pt
    if abs(dst[2]) < 1e-12:
        return (0.0, 0.0)
    return (float(dst[0] / dst[2]), float(dst[1] / dst[2]))


# ---------------------------------------------------------------------------
# Position building & smoothing
# ---------------------------------------------------------------------------

_BALL_OUT_M = 1.0   # a ball further off the pitch than this is not in play
_STILL_M = 0.5      # moved less than this in a second: still
_ALONE_M = 4.0      # ... and no player this close: a lost track, not a dead ball
_KEEPER_ZONE_M = 25.0  # a keeper's median spot is this close to a goal line


def build_positions(
    tracks: list[dict], homographies: list[dict]
) -> list[PlayerPosition]:
    """Build pitch-space positions from per-frame tracks and homography data.

    Args:
        tracks: List of per-frame dicts, each with ``frame_idx``,
            ``timestamp``, and ``objects`` (list of tracked entities).
        homographies: List of dicts with ``frame_idx`` and ``H`` (3×3 list).

    Returns:
        List of :class:`PlayerPosition` for every player in every frame.
    """
    h_by_frame: dict[int, np.ndarray] = {}
    h_entries = homographies if isinstance(homographies, list) else homographies.get("frames", [])
    for entry in h_entries:
        fidx = entry["frame_idx"]
        H = np.array(entry["H"], dtype=np.float64).reshape(3, 3)
        # Skip degenerate homographies
        if abs(np.linalg.det(H)) < 1e-6:
            continue
        h_by_frame[fidx] = H

    EXCLUDE_LABELS = {"referee", "pitch_line", "pitch_lines", "line", "ref"}

    positions: list[PlayerPosition] = []
    keeper_frames_inside: dict[tuple[str, int], int] = {}  # (team, id) → frames inside the lines
    track_frames = tracks if isinstance(tracks, list) else tracks.get("frames", [])
    for frame in track_frames:
        fidx = frame["frame_idx"]
        ts = frame.get("timestamp", 0.0)
        if fidx not in h_by_frame:
            continue
        H = h_by_frame[fidx]
        for obj in frame.get("objects", []):
            label = obj.get("cls", obj.get("label", "")).lower()
            if label in EXCLUDE_LABELS:
                continue
            foot_xy = obj.get("foot_xy")
            if foot_xy is None:
                # Fallback: derive foot from bbox bottom-center
                bbox = obj.get("bbox")
                if bbox and len(bbox) == 4:
                    foot_xy = [(bbox[0] + bbox[2]) / 2.0, bbox[3]]
                else:
                    continue
            x_m, y_m = pixel_to_pitch(tuple(foot_xy), H)
            # Substitutes warming up, ball boys and stewards wear kit colours too.
            if not (-_OFF_PITCH_M <= x_m <= _PITCH_LENGTH + _OFF_PITCH_M
                    and -_OFF_PITCH_M <= y_m <= _PITCH_WIDTH + _OFF_PITCH_M):
                continue
            pid = obj.get("object_id", obj.get("track_id", -1))
            team = "ball" if label == "ball" else (obj.get("team") or "unknown")
            if label == "goalkeeper":
                # a referee or coach in the keeper's colour stands away from both goals
                inside = 0 <= x_m <= _PITCH_LENGTH and 0 <= y_m <= _PITCH_WIDTH and min(x_m, _PITCH_LENGTH - x_m) <= _KEEPER_ZONE_M
                keeper_frames_inside[(team, pid)] = keeper_frames_inside.get((team, pid), 0) + inside
            positions.append(
                PlayerPosition(
                    frame_idx=fidx,
                    timestamp=ts,
                    player_id=pid,
                    # ball is told apart by team == "ball" downstream
                    team=team,
                    x_m=x_m,
                    y_m=y_m,
                )
            )

    # One keeper per team: the keeper colour also matches substitutes warming up by
    # the touchline, so keep the candidate seen inside the lines the longest.
    best: dict[str, tuple[int, int]] = {}
    for (team, pid), n in keeper_frames_inside.items():
        if team not in best or n > best[team][1]:
            best[team] = (pid, n)
    not_keepers = {pid for (team, pid) in keeper_frames_inside if best[team][0] != pid}
    return [p for p in positions if p.player_id not in not_keepers]


def assign_keepers(positions: list[PlayerPosition], keepers: set[int], attacking_team: str | None,
                   attacking_direction: str) -> None:
    """A keeper plays for the team defending the goal he stands in front of.

    Muse Spark cannot tell which team a keeper's colour belongs to, so the kit label is often wrong.
    """
    if attacking_team not in ("A", "B"):
        return
    defending = "B" if attacking_team == "A" else "A"
    attacked_x = _PITCH_LENGTH if attacking_direction == "right" else 0.0
    for pid in keepers:
        xs = [p.x_m for p in positions if p.player_id == pid]
        if not xs:
            continue
        team = defending if abs(float(np.median(xs)) - attacked_x) < _PITCH_LENGTH / 2 else attacking_team
        for p in positions:
            if p.player_id == pid:
                p.team = team


def smooth_positions(
    positions: list[PlayerPosition],
    window: int = 11,
    polyorder: int = 3,
) -> list[PlayerPosition]:
    """Apply Savitzky-Golay smoothing to player trajectories.

    Positions are grouped by ``player_id``; x and y coordinates are smoothed
    independently. Groups shorter than *window* are returned unchanged.
    """
    from collections import defaultdict

    by_player: dict[int, list[PlayerPosition]] = defaultdict(list)
    for pos in positions:
        by_player[pos.player_id].append(pos)

    smoothed: list[PlayerPosition] = []
    for pid, plist in by_player.items():
        plist.sort(key=lambda p: p.frame_idx)
        if len(plist) < window:
            smoothed.extend(plist)
            continue
        xs = np.array([p.x_m for p in plist], dtype=np.float64)
        ys = np.array([p.y_m for p in plist], dtype=np.float64)
        xs_smooth = savgol_filter(xs, window, polyorder)
        ys_smooth = savgol_filter(ys, window, polyorder)
        for i, p in enumerate(plist):
            smoothed.append(
                PlayerPosition(
                    frame_idx=p.frame_idx,
                    timestamp=p.timestamp,
                    player_id=p.player_id,
                    team=p.team,
                    x_m=float(xs_smooth[i]),
                    y_m=float(ys_smooth[i]),
                )
            )
    return smoothed


_TOUCH_M = 1.5        # ball this close to a player's feet is on the ground
_MAX_FLIGHT_S = 4.0   # longer gaps between touches are left as measured


def drop_stale_ball(ball: list[PlayerPosition], players: list[PlayerPosition], fps: float = 25.0) -> list[PlayerPosition]:
    """Drop "ball" positions that cannot be the ball in play.

    When SAM 3.1 loses the ball (a cross into a crowded box) it can lock onto a still white
    object off the pitch and stay there to the end of the clip. A ball off the pitch, or one that
    sits still for a second with no player near it, is not tracked any more: showing no ball is
    better than showing a wrong one.
    """
    from collections import defaultdict

    feet: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in players:
        feet[p.frame_idx].append(p)
    ball = sorted(ball, key=lambda b: b.frame_idx)
    on_pitch = [b for b in ball if -_BALL_OUT_M <= b.x_m <= _PITCH_LENGTH + _BALL_OUT_M
                and -_BALL_OUT_M <= b.y_m <= _PITCH_WIDTH + _BALL_OUT_M]
    window = max(2, int(fps))
    stale: set[int] = set()
    for i in range(len(on_pitch) - window + 1):
        run = on_pitch[i:i + window]
        if run[-1].frame_idx - run[0].frame_idx > 2 * window:
            continue  # a gap, not a still ball
        still = max(math.hypot(b.x_m - run[0].x_m, b.y_m - run[0].y_m) for b in run) < _STILL_M
        alone = all(min((math.hypot(p.x_m - b.x_m, p.y_m - b.y_m) for p in feet[b.frame_idx]), default=99) > _ALONE_M
                    for b in run)
        if still and alone:
            stale.update(b.frame_idx for b in run)
    return [b for b in on_pitch if b.frame_idx not in stale]


def ground_ball_track(ball: list[PlayerPosition], players: list[PlayerPosition]) -> None:
    """Replace the ball's ground position between two touches by a straight line.

    The ball box's bottom is only on the grass when the ball is; in the air its
    projection lands far behind the real spot. Between two touches the ground
    track of a rolling *or* flying ball is (close to) a straight line at constant
    speed, so we interpolate between touch points and flag those frames
    ``estimated``. Mutates *ball* in place.
    """
    from collections import defaultdict

    feet: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in players:
        feet[p.frame_idx].append(p)
    ball = sorted(ball, key=lambda b: b.frame_idx)
    touches = [
        i for i, b in enumerate(ball)
        if any(math.hypot(p.x_m - b.x_m, p.y_m - b.y_m) < _TOUCH_M for p in feet[b.frame_idx])
    ]
    for i0, i1 in zip(touches, touches[1:]):
        a, b = ball[i0], ball[i1]
        if i1 - i0 < 2 or b.timestamp - a.timestamp > _MAX_FLIGHT_S:
            continue
        for k in range(i0 + 1, i1):
            f = (ball[k].timestamp - a.timestamp) / (b.timestamp - a.timestamp)
            ball[k].x_m = a.x_m + f * (b.x_m - a.x_m)
            ball[k].y_m = a.y_m + f * (b.y_m - a.y_m)
            ball[k].estimated = True


# ---------------------------------------------------------------------------
# Possession detection
# ---------------------------------------------------------------------------

def detect_possession(
    positions: list[PlayerPosition],
    ball_positions: list[PlayerPosition],
    threshold_m: float = 1.5,
    min_frames: int = 3,
) -> list[PossessionEvent]:
    """Determine ball possession by nearest player sustained over *min_frames*.

    Args:
        positions: All player positions (smoothed).
        ball_positions: Ball positions (``player_id == -1`` or ``team == "ball"``).
        threshold_m: Max distance in metres for a player to be considered "on the ball".
        min_frames: Consecutive frames a player must be closest to lock possession.

    Returns:
        Chronological list of :class:`PossessionEvent` records.
    """
    from collections import defaultdict

    pos_by_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        pos_by_frame[p.frame_idx].append(p)

    ball_by_frame: dict[int, PlayerPosition] = {}
    for bp in ball_positions:
        ball_by_frame[bp.frame_idx] = bp

    # First pass — find closest player per frame
    frame_indices = sorted(ball_by_frame.keys())
    closest_per_frame: list[tuple[int, float, PlayerPosition | None, float]] = []
    for fidx in frame_indices:
        ball = ball_by_frame[fidx]
        players = pos_by_frame.get(fidx, [])
        best_dist = float("inf")
        best_player: PlayerPosition | None = None
        for pl in players:
            d = math.hypot(pl.x_m - ball.x_m, pl.y_m - ball.y_m)
            if d < best_dist:
                best_dist = d
                best_player = pl
        closest_per_frame.append((fidx, ball.timestamp, best_player, best_dist))

    # Second pass — require sustained proximity
    events: list[PossessionEvent] = []
    streak_pid: int | None = None
    streak_count: int = 0
    streak_start_idx: int = 0

    for i, (fidx, ts, player, dist) in enumerate(closest_per_frame):
        if player is None or dist > threshold_m:
            streak_pid = None
            streak_count = 0
            continue
        if player.player_id == streak_pid:
            streak_count += 1
        else:
            streak_pid = player.player_id
            streak_count = 1
            streak_start_idx = i

        if streak_count >= min_frames:
            # Emit for the first qualifying frame and continue
            if streak_count == min_frames:
                # Back-fill from streak start
                for j in range(streak_start_idx, i + 1):
                    sf, sts, sp, _ = closest_per_frame[j]
                    if sp is not None:
                        events.append(
                            PossessionEvent(
                                frame_idx=sf,
                                timestamp=sts,
                                player_id=sp.player_id,
                                team=sp.team,
                            )
                        )
            else:
                events.append(
                    PossessionEvent(
                        frame_idx=fidx,
                        timestamp=ts,
                        player_id=player.player_id,
                        team=player.team,
                    )
                )

    return events


# ---------------------------------------------------------------------------
# Pass detection
# ---------------------------------------------------------------------------

_MIN_PASS_M = 3.0  # the ball must travel this far between the two players


def detect_passes(
    possession: list[PossessionEvent],
    positions: list[PlayerPosition],
    ball_positions: list[PlayerPosition],
) -> list[PassEvent]:
    """Detect passes: possession moves to a teammate *and* the ball travels ≥ 3 m.

    Without the travel check, two teammates standing by a still ball (a corner
    being set up) "pass" every time the nearer one changes.

    Args:
        possession: Chronological possession events.
        positions: All player positions.

    Returns:
        List of :class:`PassEvent` records.
    """
    from collections import defaultdict

    pos_lookup: dict[tuple[int, int], PlayerPosition] = {}
    for p in positions:
        pos_lookup[(p.frame_idx, p.player_id)] = p
    ball_at = {b.frame_idx: b for b in ball_positions}

    passes: list[PassEvent] = []
    eid = 0
    for i in range(1, len(possession)):
        prev = possession[i - 1]
        curr = possession[i]
        if prev.player_id == curr.player_id:
            continue
        if prev.team != curr.team:
            continue  # interception, not a pass
        b0, b1 = ball_at.get(prev.frame_idx), ball_at.get(curr.frame_idx)
        if b0 is None or b1 is None or math.hypot(b1.x_m - b0.x_m, b1.y_m - b0.y_m) < _MIN_PASS_M:
            continue  # the ball did not go anywhere

        start_pos = pos_lookup.get((prev.frame_idx, prev.player_id))
        end_pos = pos_lookup.get((curr.frame_idx, curr.player_id))

        start_xy = (start_pos.x_m, start_pos.y_m) if start_pos else (0.0, 0.0)
        end_xy = (end_pos.x_m, end_pos.y_m) if end_pos else (0.0, 0.0)

        passes.append(
            PassEvent(
                event_id=eid,
                frame_idx=curr.frame_idx,
                timestamp=curr.timestamp,
                passer_id=prev.player_id,
                receiver_id=curr.player_id,
                team=curr.team,
                start_xy=start_xy,
                end_xy=end_xy,
                success=True,  # same-team transfer is deemed successful
            )
        )
        eid += 1

    return passes


# ---------------------------------------------------------------------------
# Offside detection
# ---------------------------------------------------------------------------

def compute_offside_line(
    positions: list[PlayerPosition],
    pass_events: list[PassEvent],
    attacking_direction: str = "right",
) -> list[OffsideSnapshot]:
    """Compute offside status at the moment of each pass.

    Uses the second-to-last defender's x coordinate as the offside line.

    Args:
        positions: All player positions.
        pass_events: Detected passes.
        attacking_direction: ``"right"`` means the attacking team plays
            towards increasing x; ``"left"`` towards decreasing x.

    Returns:
        One :class:`OffsideSnapshot` per pass event.
    """
    from collections import defaultdict

    pos_by_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        pos_by_frame[p.frame_idx].append(p)

    snapshots: list[OffsideSnapshot] = []
    for pe in pass_events:
        frame_players = pos_by_frame.get(pe.frame_idx, [])
        defending_team = [
            p for p in frame_players if p.team != pe.team
        ]
        attacking_in_frame = [
            p for p in frame_players if p.team == pe.team
        ]

        # Determine offside line from defending team
        if len(defending_team) < 2:
            continue

        if attacking_direction == "right":
            # Defenders sorted by descending x (towards own goal)
            defending_xs = sorted([d.x_m for d in defending_team], reverse=True)
            offside_x = defending_xs[1]  # second-to-last
            attacking_xs = [a.x_m for a in attacking_in_frame]
            is_offside = any(ax > offside_x for ax in attacking_xs if ax > pe.start_xy[0])
        else:
            defending_xs = sorted([d.x_m for d in defending_team])
            offside_x = defending_xs[1]
            attacking_xs = [a.x_m for a in attacking_in_frame]
            is_offside = any(ax < offside_x for ax in attacking_xs if ax < pe.start_xy[0])

        snapshots.append(
            OffsideSnapshot(
                frame_idx=pe.frame_idx,
                timestamp=pe.timestamp,
                offside_line_x=offside_x,
                attacking_players_x=attacking_xs,
                is_offside=is_offside,
            )
        )

    return snapshots


# ---------------------------------------------------------------------------
# Shot detection
# ---------------------------------------------------------------------------

_GOAL_WIDTH = 7.32          # metres (FIFA standard)
_PITCH_LENGTH = 105.0       # metres (FIFA standard)
_PITCH_WIDTH = 68.0         # metres (FIFA standard)
_OFF_PITCH_M = 3.0          # people further outside the lines are not in play


_SHOT_WINDOW_S = 2.0        # the ball must reach the goal mouth this soon after the touch
_GOAL_LINE_M = 2.0          # "reached": within this of the goal line, or beyond it…
_GOAL_MOUTH_HALF_M = 8.0    # …and within this of the goal centre across the pitch


def _releases(possession: list[PossessionEvent]) -> list[PossessionEvent]:
    """Last frame of each run of possession by one player (the moment the ball leaves)."""
    out = []
    for i, ev in enumerate(possession):
        nxt = possession[i + 1] if i + 1 < len(possession) else None
        if nxt is None or nxt.player_id != ev.player_id or nxt.frame_idx > ev.frame_idx + 1:
            out.append(ev)
    return out


def detect_shots(
    possession: list[PossessionEvent],
    positions: list[PlayerPosition],
    ball_positions: list[PlayerPosition],
) -> list[ShotEvent]:
    """A shot is a touch after which the ball reaches a goal mouth within 2 s.

    Not "the last touch of the clip": a clip that ends on a pass or a corner
    being set up has no shot.
    """
    pos_lookup = {(p.frame_idx, p.player_id): p for p in positions}
    ball = sorted(ball_positions, key=lambda b: b.frame_idx)
    shots: list[ShotEvent] = []
    for rel in _releases(possession):
        shooter = pos_lookup.get((rel.frame_idx, rel.player_id))
        if shooter is None:
            continue
        after = [b for b in ball if rel.timestamp < b.timestamp <= rel.timestamp + _SHOT_WINDOW_S]
        goal_x = next((
            gx for gx in (0.0, _PITCH_LENGTH)
            for b in after
            if (b.x_m <= gx + _GOAL_LINE_M if gx == 0.0 else b.x_m >= gx - _GOAL_LINE_M)
            and abs(b.y_m - _PITCH_WIDTH / 2) <= _GOAL_MOUTH_HALF_M
        ), None)
        if goal_x is None:
            continue
        goal_y = _PITCH_WIDTH / 2.0
        distance = math.hypot(shooter.x_m - goal_x, shooter.y_m - goal_y)
        # Angle subtended by the posts from the shooter's position
        angle_left = math.atan2(goal_y - _GOAL_WIDTH / 2 - shooter.y_m, goal_x - shooter.x_m)
        angle_right = math.atan2(goal_y + _GOAL_WIDTH / 2 - shooter.y_m, goal_x - shooter.x_m)
        shots.append(ShotEvent(
            event_id=len(shots),
            frame_idx=rel.frame_idx,
            timestamp=rel.timestamp,
            shooter_id=rel.player_id,
            distance_m=round(distance, 2),
            angle_deg=round(abs(math.degrees(angle_right - angle_left)), 2),
            goal_xy=(goal_x, goal_y),
        ))
    return shots


# ---------------------------------------------------------------------------
# Open lanes
# ---------------------------------------------------------------------------

def detect_open_lanes(
    positions: list[PlayerPosition],
    possession: list[PossessionEvent],
    lane_width_m: float = 2.0,
) -> list[OpenLane]:
    """Check passing-lane clarity between the ball holder and each teammate.

    For every frame where a player has possession, draw a line to each teammate
    and test whether any opponent falls within *lane_width_m* of that segment.

    Args:
        positions: All player positions.
        possession: Chronological possession events.
        lane_width_m: Width of the corridor to test.

    Returns:
        List of :class:`OpenLane` records.
    """
    from collections import defaultdict

    pos_by_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        pos_by_frame[p.frame_idx].append(p)

    possession_frames: dict[int, PossessionEvent] = {}
    for pe in possession:
        possession_frames[pe.frame_idx] = pe

    lanes: list[OpenLane] = []
    for fidx, pe in possession_frames.items():
        frame_players = pos_by_frame.get(fidx, [])
        holder = None
        teammates: list[PlayerPosition] = []
        opponents: list[PlayerPosition] = []
        for pl in frame_players:
            if pl.player_id == pe.player_id:
                holder = pl
            elif pl.team == pe.team:
                teammates.append(pl)
            else:
                opponents.append(pl)

        if holder is None:
            continue

        a = np.array([holder.x_m, holder.y_m])
        for tm in teammates:
            b = np.array([tm.x_m, tm.y_m])
            nearest_opp = float("inf")
            for opp in opponents:
                op = np.array([opp.x_m, opp.y_m])
                d = point_to_segment_distance(op, a, b)
                if d < nearest_opp:
                    nearest_opp = d

            lanes.append(
                OpenLane(
                    frame_idx=fidx,
                    from_id=pe.player_id,
                    to_id=tm.player_id,
                    team=pe.team,
                    lane_clear=nearest_opp > lane_width_m,
                    nearest_opponent_dist=round(nearest_opp, 2),
                )
            )

    return lanes


# ---------------------------------------------------------------------------
# Lost-man detection
# ---------------------------------------------------------------------------

def detect_lost_man(
    positions: list[PlayerPosition],
    window: int = 15,
) -> list[KeyMoment]:
    """Detect moments where an attacker loses their defensive marker.

    For each attacker, track the distance to the closest defender over time.
    If this distance increases rapidly (gradient above threshold over *window*
    frames), emit a ``"lost_man"`` key moment.

    Args:
        positions: All player positions (smoothed).
        window: Number of frames over which to compute the gradient.

    Returns:
        List of :class:`KeyMoment` records of type ``"lost_man"``.
    """
    from collections import defaultdict

    pos_by_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        pos_by_frame[p.frame_idx].append(p)

    # Identify teams
    teams: set[str] = {p.team for p in positions if p.team != "ball"}
    if len(teams) < 2:
        return []

    team_list = sorted(teams)
    # Treat both teams symmetrically (either could be attacking)
    moments: list[KeyMoment] = []
    mid = 0

    for atk_team in team_list:
        def_team = [t for t in team_list if t != atk_team][0]

        # Build per-attacker closest-defender distance series
        attacker_ids: set[int] = {
            p.player_id for p in positions if p.team == atk_team
        }
        all_frames = sorted(pos_by_frame.keys())

        for aid in attacker_ids:
            dists: list[tuple[int, float, float]] = []  # (frame, dist, timestamp)
            for fidx in all_frames:
                atk_pos = None
                defenders: list[PlayerPosition] = []
                for pl in pos_by_frame[fidx]:
                    if pl.player_id == aid and pl.team == atk_team:
                        atk_pos = pl
                    elif pl.team == def_team:
                        defenders.append(pl)
                if atk_pos is None or not defenders:
                    continue
                min_d = min(
                    math.hypot(atk_pos.x_m - d.x_m, atk_pos.y_m - d.y_m)
                    for d in defenders
                )
                dists.append((fidx, min_d, atk_pos.timestamp))

            if len(dists) < window:
                continue

            dist_arr = np.array([d[1] for d in dists])
            # Gradient: change in closest-defender distance over window
            grad = np.gradient(dist_arr)
            smoothed_grad = (
                savgol_filter(grad, min(window, len(grad) if len(grad) % 2 == 1 else len(grad) - 1), min(3, min(window, len(grad) if len(grad) % 2 == 1 else len(grad) - 1) - 1))
                if len(grad) >= window
                else grad
            )

            threshold = 0.3  # metres per frame
            for i, g in enumerate(smoothed_grad):
                if g > threshold:
                    fidx_moment = dists[i][0]
                    ts_moment = dists[i][2]
                    # Find closest defender id at that frame
                    closest_def_id = -1
                    min_dd = float("inf")
                    for pl in pos_by_frame.get(fidx_moment, []):
                        if pl.team == def_team:
                            dd = math.hypot(
                                pl.x_m - next(
                                    (pp.x_m for pp in pos_by_frame[fidx_moment] if pp.player_id == aid),
                                    0,
                                ),
                                pl.y_m - next(
                                    (pp.y_m for pp in pos_by_frame[fidx_moment] if pp.player_id == aid),
                                    0,
                                ),
                            )
                            if dd < min_dd:
                                min_dd = dd
                                closest_def_id = pl.player_id

                    moments.append(
                        KeyMoment(
                            moment_id=mid,
                            frame_idx=fidx_moment,
                            timestamp=ts_moment,
                            event_type="lost_man",
                            player_ids=[aid, closest_def_id],
                            description=(
                                f"Player {aid} ({atk_team}) lost their marker "
                                f"(defender {closest_def_id}) — separation increasing rapidly."
                            ),
                        )
                    )
                    mid += 1
                    break  # one moment per attacker

    return moments


# ---------------------------------------------------------------------------
# Pass chain reconstruction
# ---------------------------------------------------------------------------

def build_pass_chain(
    pass_events: list[PassEvent],
    shot: ShotEvent | None,
) -> list[PassEvent]:
    """Trace backwards from the shot to reconstruct the passing chain.

    Starting from the shooter, find the most recent pass whose receiver is the
    shooter, then repeat for the passer, and so on.

    Args:
        pass_events: All detected passes.
        shot: The shot event (may be ``None``).

    Returns:
        The pass chain in chronological order (earliest pass first).
    """
    if shot is None or not pass_events:
        return []

    chain: list[PassEvent] = []
    target_player = shot.shooter_id
    target_frame = shot.frame_idx

    # Work backwards through passes
    remaining = sorted(pass_events, key=lambda p: p.frame_idx, reverse=True)
    visited: set[int] = set()

    while True:
        found = False
        for pe in remaining:
            if pe.event_id in visited:
                continue
            if pe.receiver_id == target_player and pe.frame_idx <= target_frame:
                chain.append(pe)
                visited.add(pe.event_id)
                target_player = pe.passer_id
                target_frame = pe.frame_idx
                found = True
                break
        if not found:
            break

    chain.reverse()
    return chain


# ---------------------------------------------------------------------------
# Key moments aggregation
# ---------------------------------------------------------------------------

_MAX_LOST_MAN = 4  # each key moment costs Muse Spark calls; a goal needs a handful


def collect_key_moments(
    passes: list[PassEvent],
    shots: list[ShotEvent],
    lost_man: list[KeyMoment],
    attackers: set[int] | None = None,
) -> list[KeyMoment]:
    """Key moments for Muse Spark to read: every pass and shot, plus the last few
    lost-man moments of *attackers* (player ids) before the final event.

    Open lanes stay in events.json as evidence but are not key moments: one per
    frame flooded the list (99 moments, 32 min of reading on a 24 s clip).
    """
    moments: list[KeyMoment] = []
    for pe in passes:
        moments.append(KeyMoment(
            moment_id=0, frame_idx=pe.frame_idx, timestamp=pe.timestamp, event_type="pass",
            player_ids=[pe.passer_id, pe.receiver_id],
            description=f"Pass from player {pe.passer_id} to {pe.receiver_id}.",
        ))
    for se in shots:
        moments.append(KeyMoment(
            moment_id=0, frame_idx=se.frame_idx, timestamp=se.timestamp, event_type="shot",
            player_ids=[se.shooter_id],
            description=f"Shot by player {se.shooter_id} from {se.distance_m:.1f}m, angle {se.angle_deg:.1f}°.",
        ))
    end_t = max((m.timestamp for m in moments), default=float("inf"))
    candidates = [m for m in lost_man
                  if (attackers is None or m.player_ids[0] in attackers) and m.timestamp <= end_t]
    moments += sorted(candidates, key=lambda m: m.timestamp)[-_MAX_LOST_MAN:]

    moments.sort(key=lambda m: m.frame_idx)
    for i, m in enumerate(moments):
        m.moment_id = i
    return moments


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(clip_id: str) -> Path:
    """Execute the full geometry pipeline for a clip.

    1. Load ``tracks.json`` and ``homography.json`` from ``runs/{clip_id}/``.
    2. Build and smooth positions.
    3. Detect ball positions.
    4. Detect possession, passes, offside, shots, open lanes, lost man.
    5. Build pass chain.
    6. Collect key moments.
    7. Save ``pitch.json`` and ``events.json``.

    Args:
        clip_id: Unique identifier for the clip being analysed.

    Returns:
        Path to the written ``events.json``.
    """
    run_dir = Path("runs") / clip_id
    tracks = _load_json(run_dir / "tracks.json")
    homographies = _load_json(run_dir / "homography.json")

    # Build & smooth player positions
    raw_positions = build_positions(tracks, homographies)
    positions = smooth_positions(raw_positions)

    # Separate ball positions
    player_positions = [p for p in positions if p.team != "ball"]
    ball_positions = drop_stale_ball([p for p in positions if p.team == "ball"], player_positions)
    positions = player_positions + ball_positions
    ground_ball_track(ball_positions, player_positions)

    # Tactical analysis
    possession = detect_possession(player_positions, ball_positions)
    passes = detect_passes(possession, player_positions, ball_positions)
    shots = detect_shots(possession, player_positions, ball_positions)
    # Attack direction = side of the goal that was shot at (else where the ball ended)
    if shots:
        attacking_direction = "right" if shots[0].goal_xy[0] > 0 else "left"
    else:
        if ball_positions:
            end_x = max(ball_positions, key=lambda b: b.frame_idx).x_m
        else:
            last = max((p.frame_idx for p in player_positions), default=0)
            xs = [p.x_m for p in player_positions if p.frame_idx == last]
            end_x = sum(xs) / len(xs) if xs else _PITCH_LENGTH
        attacking_direction = "right" if end_x > _PITCH_LENGTH / 2 else "left"
    from collections import Counter
    teams = Counter(ev.team for ev in possession).most_common(1)
    keepers = {o["object_id"] for f in tracks["frames"] for o in f["objects"] if o.get("cls") == "goalkeeper"}
    assign_keepers(positions, keepers, teams[0][0] if teams else None, attacking_direction)
    offside = compute_offside_line(player_positions, passes, attacking_direction)
    open_lanes = detect_open_lanes(player_positions, possession)
    lost_man = detect_lost_man(player_positions)

    # Chain & key moments
    shot = shots[0] if shots else None
    pass_chain = build_pass_chain(passes, shot)
    # Attacking team = the team that had the ball most
    attackers = {p.player_id for p in player_positions if teams and p.team == teams[0][0]} - keepers
    key_moments = collect_key_moments(passes, shots, lost_man, attackers or None)

    # ---- Serialize --------------------------------------------------------

    # pitch.json — positions grouped by frame
    from collections import defaultdict
    frames_dict: dict[int, list[dict]] = defaultdict(list)
    for p in positions:
        frames_dict[p.frame_idx].append(p.to_dict())

    pitch_data = [
        {"frame_idx": fidx, "players": players}
        for fidx, players in sorted(frames_dict.items())
    ]

    pitch_path = run_dir / "pitch.json"
    with open(pitch_path, "w", encoding="utf-8") as fh:
        json.dump(pitch_data, fh, indent=2)

    # events.json
    events_data = {
        "attacking_direction": attacking_direction,
        "passes": [p.to_dict() for p in passes],
        "shots": [s.to_dict() for s in shots],
        "offside": [o.to_dict() for o in offside],
        "open_lanes": [l.to_dict() for l in open_lanes],
        "key_moments": [k.to_dict() for k in key_moments],
        "pass_chain": [p.to_dict() for p in pass_chain],
    }

    events_path = run_dir / "events.json"
    with open(events_path, "w", encoding="utf-8") as fh:
        json.dump(events_data, fh, indent=2)

    return events_path
