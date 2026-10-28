"""Tests for geometry analysis and event detection."""

import numpy as np
import pytest

from goal_explainer.geometry import (
    PlayerPosition,
    PossessionEvent,
    PassEvent,
    ShotEvent,
    OpenLane,
    KeyMoment,
    pixel_to_pitch,
    point_to_segment_distance,
    smooth_positions,
    detect_possession,
    detect_passes,
    compute_offside_line,
    detect_shots,
    detect_open_lanes,
    detect_lost_man,
    build_pass_chain,
    collect_key_moments,
)


def test_point_to_segment_distance():
    a = (0.0, 0.0)
    b = (10.0, 0.0)
    # Point directly above midpoint
    assert pytest.approx(point_to_segment_distance((5.0, 3.0), a, b)) == 3.0
    # Point before segment start
    assert pytest.approx(point_to_segment_distance((-4.0, 0.0), a, b)) == 4.0
    # Point after segment end
    assert pytest.approx(point_to_segment_distance((13.0, 4.0), a, b)) == 5.0


def test_pixel_to_pitch():
    # Construct an identity homography with offset
    # H maps pitch to pixel: px = x + 100, py = y + 50
    H = np.array([
        [1.0, 0.0, 100.0],
        [0.0, 1.0, 50.0],
        [0.0, 0.0, 1.0],
    ])
    px = (150.0, 80.0)
    x, y = pixel_to_pitch(px, H)
    assert pytest.approx(x) == 50.0
    assert pytest.approx(y) == 30.0


def test_smooth_positions():
    # Create noisy positions for a player moving linearly
    positions = []
    for i in range(25):
        noise = 0.2 if i % 2 == 0 else -0.2
        positions.append(
            PlayerPosition(
                frame_idx=i,
                timestamp=i * 0.04,
                player_id=10,
                team="A",
                x_m=float(i + noise),
                y_m=20.0,
            )
        )
    smoothed = smooth_positions(positions, window=7, polyorder=2)
    assert len(smoothed) == len(positions)
    assert smoothed[0].player_id == 10
    assert abs(smoothed[-1].x_m - 24.0) < 1.0


def _pass_scene(ball_moves: bool):
    """Player 1 at x=10, player 2 at x=20; the ball either travels 1 → 2 or sits still."""
    players, ball = [], []
    for f in range(30):
        players.append(PlayerPosition(frame_idx=f, timestamp=f * 0.04, player_id=1, team="A", x_m=10.0, y_m=20.0))
        players.append(PlayerPosition(frame_idx=f, timestamp=f * 0.04, player_id=2, team="A", x_m=20.0, y_m=20.0))
        x = (10.0 if f < 10 else 10.0 + (f - 9) * 1.0 if f < 19 else 20.0) if ball_moves else 15.0
        ball.append(PlayerPosition(frame_idx=f, timestamp=f * 0.04, player_id=-1, team="ball", x_m=x, y_m=20.0))
    return players, ball


def test_possession_and_pass_detection():
    players, ball = _pass_scene(ball_moves=True)
    possession = detect_possession(players, ball, threshold_m=1.5, min_frames=3)
    assert {p.player_id for p in possession} == {1, 2}
    passes = detect_passes(possession, players, ball)
    assert [(p.passer_id, p.receiver_id, p.team) for p in passes] == [(1, 2, "A")]


def test_no_pass_when_the_ball_stays_put():
    # Two teammates by a still ball (corner set-up): nearest player flips, no pass.
    players, _ = _pass_scene(ball_moves=False)
    ball = [PlayerPosition(frame_idx=f, timestamp=f * 0.04, player_id=-1, team="ball", x_m=10.5, y_m=20.0) for f in range(30)]
    for f in range(15, 30):  # player 2 walks up next to the ball
        players[2 * f + 1] = PlayerPosition(frame_idx=f, timestamp=f * 0.04, player_id=2, team="A", x_m=10.2, y_m=20.0)
    possession = detect_possession(players, ball, threshold_m=1.5, min_frames=3)
    assert detect_passes(possession, players, ball) == []


def test_detect_shots():
    possession = [PossessionEvent(frame_idx=f, timestamp=f * 0.04, player_id=9, team="A") for f in range(8, 11)]
    positions = [PlayerPosition(frame_idx=10, timestamp=0.4, player_id=9, team="A", x_m=95.0, y_m=34.0)]
    # after the touch the ball flies into the right-hand goal
    ball = [PlayerPosition(frame_idx=f, timestamp=f * 0.04, player_id=-1, team="ball", x_m=95.0 + (f - 10) * 1.0, y_m=34.0)
            for f in range(10, 22)]
    shots = detect_shots(possession, positions, ball)
    assert len(shots) == 1
    s = shots[0]
    assert s.shooter_id == 9 and s.goal_xy == (105.0, 34.0)
    assert pytest.approx(s.distance_m) == 10.0
    assert s.angle_deg > 0.0


def test_no_shot_when_the_ball_does_not_reach_goal():
    possession = [PossessionEvent(frame_idx=f, timestamp=f * 0.04, player_id=9, team="A") for f in range(8, 11)]
    positions = [PlayerPosition(frame_idx=10, timestamp=0.4, player_id=9, team="A", x_m=95.0, y_m=34.0)]
    ball = [PlayerPosition(frame_idx=f, timestamp=f * 0.04, player_id=-1, team="ball", x_m=95.0, y_m=34.0 - (f - 10))
            for f in range(10, 22)]  # rolled sideways, away from goal
    assert detect_shots(possession, positions, ball) == []


def test_offside_detection():
    # Pass at frame 5: receiver is at x=80. Defenders at x=75 and x=70.
    pass_event = PassEvent(
        event_id=1,
        frame_idx=5,
        timestamp=0.2,
        passer_id=8,
        receiver_id=11,
        team="A",
        start_xy=(50.0, 30.0),
        end_xy=(80.0, 30.0),
        success=True,
    )
    positions = [
        # Passer
        PlayerPosition(frame_idx=5, timestamp=0.2, player_id=8, team="A", x_m=50.0, y_m=30.0),
        # Receiver
        PlayerPosition(frame_idx=5, timestamp=0.2, player_id=11, team="A", x_m=80.0, y_m=30.0),
        # Defenders: Goalkeeper at 100, CB at 75, LB at 70
        PlayerPosition(frame_idx=5, timestamp=0.2, player_id=101, team="B", x_m=100.0, y_m=34.0),
        PlayerPosition(frame_idx=5, timestamp=0.2, player_id=102, team="B", x_m=75.0, y_m=25.0),
        PlayerPosition(frame_idx=5, timestamp=0.2, player_id=103, team="B", x_m=70.0, y_m=40.0),
    ]
    snapshots = compute_offside_line(positions, [pass_event], attacking_direction="right")
    assert len(snapshots) == 1
    # 2nd to last defender sorted descending: [100.0, 75.0, 70.0] -> index 1 is 75.0
    assert snapshots[0].offside_line_x == 75.0
    # Receiver is at 80.0 > 75.0 -> Offside!
    assert snapshots[0].is_offside is True


def test_open_lanes():
    possession = [PossessionEvent(frame_idx=0, timestamp=0.0, player_id=1, team="A")]
    positions = [
        PlayerPosition(frame_idx=0, timestamp=0.0, player_id=1, team="A", x_m=20.0, y_m=20.0),
        PlayerPosition(frame_idx=0, timestamp=0.0, player_id=2, team="A", x_m=40.0, y_m=20.0),
        PlayerPosition(frame_idx=0, timestamp=0.0, player_id=3, team="B", x_m=30.0, y_m=40.0),
    ]
    lanes = detect_open_lanes(positions, possession, lane_width_m=2.0)
    assert len(lanes) == 1
    assert lanes[0].lane_clear is True
    assert lanes[0].nearest_opponent_dist == 20.0


def test_build_positions_drops_people_off_the_pitch():
    from goal_explainer.geometry import build_positions
    H = np.eye(3)  # 1 px = 1 m
    tracks = [{"frame_idx": 0, "timestamp": 0.0, "objects": [
        {"object_id": 1, "cls": "player", "team": "A", "foot_xy": [50, 30]},
        {"object_id": 2, "cls": "player", "team": "A", "foot_xy": [120, 30]},  # warming up behind the goal
        {"object_id": 3, "cls": "goalkeeper", "team": "B", "foot_xy": [104, 34]},
    ]}]
    got = build_positions(tracks, [{"frame_idx": 0, "H": H.tolist()}])
    assert sorted(p.player_id for p in got) == [1, 3]


def test_build_positions_keeps_one_keeper_per_team():
    from goal_explainer.geometry import build_positions
    H = np.eye(3)  # 1 px = 1 m
    keeper = {"object_id": 4000, "cls": "goalkeeper", "team": "B", "foot_xy": [70, 33]}
    sub = {"object_id": 4001, "cls": "goalkeeper", "team": "B", "foot_xy": [8, -1]}  # warming up, 1 m outside
    tracks = [{"frame_idx": i, "timestamp": i / 25, "objects": [dict(keeper), dict(sub)]} for i in range(5)]
    got = build_positions(tracks, [{"frame_idx": i, "H": H.tolist()} for i in range(5)])
    assert {p.player_id for p in got} == {4000}


def test_ground_ball_track_straightens_flight_between_touches():
    from goal_explainer.geometry import PlayerPosition, ground_ball_track
    fps = 10
    players, ball = [], []
    for i in range(11):
        players.append(PlayerPosition(i, i / fps, 1, "A", 10.0, 10.0))
        players.append(PlayerPosition(i, i / fps, 2, "A", 20.0, 10.0))
        # touch at frame 0 (player 1) and frame 10 (player 2); in between the
        # projected ball "flies" far off, as an airborne ball does
        x, y = (10.0, 10.0) if i == 0 else (20.0, 10.0) if i == 10 else (40.0, 30.0)
        ball.append(PlayerPosition(i, i / fps, -1, "ball", x, y))
    ground_ball_track(ball, players)
    assert ball[5].estimated and abs(ball[5].x_m - 15.0) < 1e-9 and abs(ball[5].y_m - 10.0) < 1e-9
    assert not ball[0].estimated and not ball[10].estimated


def test_key_moments_are_capped_to_the_attack():
    from goal_explainer.geometry import KeyMoment, PassEvent
    passes = [PassEvent(0, 100, 4.0, 1, 2, "A", (0, 0), (5, 0), True)]
    lost = [KeyMoment(0, f, f / 25, "lost_man", [pid, 9], "") for f, pid in
            [(10, 1), (20, 2), (30, 3), (40, 4), (50, 5), (60, 77), (200, 1)]]  # 77: defender; 200: after the pass
    km = collect_key_moments(passes, [], lost, attackers={1, 2, 3, 4, 5})
    assert [m.event_type for m in km].count("lost_man") == 4
    assert {m.player_ids[0] for m in km if m.event_type == "lost_man"} == {2, 3, 4, 5}  # the last four before the pass
    assert [m.moment_id for m in km] == list(range(len(km)))
