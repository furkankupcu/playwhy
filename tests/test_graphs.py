"""Tests for step 6: marking and team-shape graphs."""

from goal_explainer.graphs import attacking_team, frame_graph, marking_state


def P(pid, team, x, y):
    return {"player_id": pid, "team": team, "x_m": x, "y_m": y}


def test_marking_state_bands():
    assert [marking_state(d) for d in (0.5, 2.0, 4.9, 5.0, 12)] == ["tight", "loose", "loose", "free", "free"]


def test_marking_keeps_its_defender_unless_another_is_clearly_closer():
    prev: dict[int, int] = {}
    people = [P(1, "B", 10, 10), P(20, "A", 11, 10), P(21, "A", 13, 10)]
    g = frame_graph(people, set(), "B", prev)
    assert [(m["att"], m["def"], m["state"]) for m in g["marks"]] == [(1, 20, "tight")]
    # 21 is now 0.5 m closer than 20: not enough to switch (no flicker)
    people = [P(1, "B", 10, 10), P(20, "A", 11.5, 10), P(21, "A", 11.0, 10)]
    assert frame_graph(people, set(), "B", prev)["marks"][0]["def"] == 20
    # 21 is now clearly closer: switch
    people = [P(1, "B", 10, 10), P(20, "A", 14, 10), P(21, "A", 10.5, 10)]
    assert frame_graph(people, set(), "B", prev)["marks"][0]["def"] == 21


def test_keepers_are_not_attackers_and_lanes_can_be_blocked():
    people = [P(1, "B", 0, 0), P(2, "B", 10, 0), P(3, "B", 5, 8), P(9, "B", 50, 30),  # 9: keeper
              P(20, "A", 5, 0.5)]  # sits in the 1→2 lane
    g = frame_graph(people, {9}, "B", {})
    assert {m["att"] for m in g["marks"]} == {1, 2, 3}
    lanes = {(min(l["a"], l["b"]), max(l["a"], l["b"])): l["blocked"] for l in g["links"] if l["team"] == "B"}
    assert lanes[(1, 2)] is True and lanes[(1, 3)] is False


def test_attacking_team_is_nearest_the_ball_most_often():
    frames = [{"players": [P(1, "A", 0, 0), P(2, "B", 5, 0), P(-1, "ball", 4.5, 0)]} for _ in range(3)]
    frames.append({"players": [P(1, "A", 0, 0), P(2, "B", 5, 0), P(-1, "ball", 0.2, 0)]})
    assert attacking_team(frames) == "B"
    assert attacking_team([{"players": []}]) is None
