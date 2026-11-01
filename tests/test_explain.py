"""Tests for Step 5: Explain (validation, prompt building, schema enforcement)."""

import pytest

from goal_explainer.explain import (
    build_explanation_prompt,
    validate_claim,
    validate_explanation,
    _collect_valid_ids,
)


def test_collect_valid_ids():
    events = {
        "passes": [{"event_id": "pass_01"}],
        "shots": [{"event_id": "shot_01"}],
        "key_moments": [{"moment_id": "moment_01"}],
    }
    cues = [
        {"player_id": 7, "moment_id": "moment_01"},
        {"player_id": 4, "moment_id": "moment_02"},
    ]
    ids = _collect_valid_ids(events, cues)
    assert "pass_01" in ids
    assert "shot_01" in ids
    assert "moment_01" in ids
    assert "cue_7_moment_01" in ids
    assert "cue_4_moment_02" in ids
    assert "non_existent" not in ids


def test_validate_claim_valid():
    events = {"passes": [{"event_id": 10}]}
    cues = [{"player_id": 9, "moment_id": 1}]
    claim = {
        "claim_id": 1,
        "text": "Player 9 made a quick turn while marker hesitated.",
        "t_start": 2.0,
        "t_end": 4.5,
        "player_ids": [9],
        "evidence": [
            {"type": "event", "id": "10", "description": "Pass received"},
            {"type": "cue", "id": "cue_9_1", "description": "Quick reaction"},
        ],
    }
    is_valid, reason = validate_claim(claim, events, cues, clip_duration=10.0)
    assert is_valid is True
    assert reason == ""


def test_validate_claim_out_of_bounds():
    events = {"passes": [{"event_id": 10}]}
    cues = []
    # t_end exceeds clip_duration
    claim = {
        "claim_id": 1,
        "text": "Test claim",
        "t_start": 2.0,
        "t_end": 12.0,
        "player_ids": [9],
        "evidence": [{"type": "event", "id": "10", "description": "Pass received"}],
    }
    is_valid, reason = validate_claim(claim, events, cues, clip_duration=10.0)
    assert is_valid is False
    assert "out of range" in reason


def test_validate_claim_invalid_time_order():
    events = {"passes": [{"event_id": 10}]}
    cues = []
    claim = {
        "claim_id": 1,
        "text": "Test claim",
        "t_start": 5.0,
        "t_end": 3.0,
        "player_ids": [9],
        "evidence": [{"type": "event", "id": "10", "description": "Pass received"}],
    }
    is_valid, reason = validate_claim(claim, events, cues, clip_duration=10.0)
    assert is_valid is False
    assert "t_start" in reason and "t_end" in reason


def test_validate_claim_invalid_evidence():
    events = {"passes": [{"event_id": 10}]}
    cues = []
    # Citing an evidence ID that doesn't exist
    claim = {
        "claim_id": 1,
        "text": "Hallucinated claim",
        "t_start": 1.0,
        "t_end": 3.0,
        "player_ids": [9],
        "evidence": [{"type": "event", "id": "fake_event_999", "description": "Fake"}],
    }
    is_valid, reason = validate_claim(claim, events, cues, clip_duration=10.0)
    assert is_valid is False
    assert "fake_event_999" in reason


def test_validate_explanation():
    events = {"passes": [{"event_id": 1}]}
    cues = []
    raw_explanation = {
        "clip_id": "test_goal",
        "claims": [
            {
                "claim_id": 1,
                "text": "Valid claim",
                "t_start": 0.5,
                "t_end": 2.0,
                "player_ids": [10],
                "evidence": [{"type": "event", "id": "1", "description": "Pass"}],
            },
            {
                "claim_id": 2,
                "text": "Invalid evidence claim",
                "t_start": 1.0,
                "t_end": 3.0,
                "player_ids": [10],
                "evidence": [{"type": "event", "id": "unknown_id", "description": "Fake"}],
            },
        ],
        "summary": "Goal test summary.",
    }
    cleaned = validate_explanation(raw_explanation, events, cues, clip_duration=5.0)
    assert len(cleaned["claims"]) == 1
    assert cleaned["claims"][0]["claim_id"] == 1
    assert cleaned["summary"] == "Goal test summary."


def test_player_names_use_colour_number_and_role():
    from goal_explainer.explain import build_explanation_prompt, player_names
    tracks = {"kits": {"team_a": "white", "team_b": "blue"}, "frames": [{"objects": [
        {"object_id": 1001, "cls": "player", "team": "A"},
        {"object_id": 2003, "cls": "player", "team": "B"},
        {"object_id": 2004, "cls": "player", "team": "B"},
        {"object_id": 4000, "cls": "goalkeeper", "team": "B"},
        {"object_id": 5000, "cls": "referee", "team": None},
    ]}]}
    cues = [{"player_id": 2004, "jersey_number": "6", "visible": True},
            {"player_id": 2004, "jersey_number": "8", "visible": True},
            {"player_id": 2004, "jersey_number": "6", "visible": True},
            {"player_id": 1001, "jersey_number": "", "visible": True}]
    names = player_names(tracks, cues)
    assert names == {1001: "white player 1", 2003: "blue player 1", 2004: "blue #6", 4000: "blue keeper"}
    prompt = build_explanation_prompt({"passes": []}, [], names)
    assert '"2004": "blue #6"' in prompt and "never by id" in prompt


def test_pass_facts_and_strong_cues():
    from goal_explainer.explain import add_pass_facts, strong_cues
    ev = {"attacking_direction": "right",
          "passes": [{"frame_idx": 5, "start_xy": [80.6, 49.9], "end_xy": [67.8, 46.0]}],
          "key_moments": [{"event_type": "pass", "frame_idx": 5, "description": "Pass from player 1 to 2."}]}
    add_pass_facts(ev)
    assert ev["passes"][0]["direction"] == "backward" and ev["passes"][0]["length_m"] == 13.4
    assert ev["key_moments"][0]["description"].endswith("13.4 m, backward.")
    cues = [{"visible": True, "confidence": 0.6}, {"visible": True, "confidence": 0.4}, {"visible": False, "confidence": 0.9}]
    assert strong_cues(cues) == cues[:1]


def test_pass_facts_measure_receiver_space():
    from goal_explainer.explain import add_pass_facts
    ev = {"attacking_direction": "right",
          "passes": [{"frame_idx": 5, "receiver_id": 2, "start_xy": [60, 30], "end_xy": [70, 30]}]}
    players = [{"player_id": 2, "team": "B", "x_m": 70, "y_m": 30}, {"player_id": 9, "team": "A", "x_m": 73, "y_m": 34},
               {"player_id": 8, "team": "A", "x_m": 60, "y_m": 30}, {"player_id": 0, "team": "ball", "x_m": 70, "y_m": 30}]
    add_pass_facts(ev, [{"frame_idx": 5, "players": players}])
    assert ev["passes"][0]["receiver_nearest_opponent_m"] == 5.0
    assert ev["passes"][0]["opponents_goal_side_of_receiver"] == 1


def test_audit_edits_words_only():
    from goal_explainer.explain import audit_explanation

    class Client:
        def spark_json(self, prompt, schema):
            assert "far post" in prompt
            return {"summary": "clean", "claims": [
                {"claim_id": 1, "text": "tapped in", "commentary": "tapped in!", "removed": ["at the far post"]},
                {"claim_id": 99, "text": "made up", "commentary": "made up", "removed": []}]}

    claims = [{"claim_id": 1, "text": "tapped in at the far post", "commentary": "far post!", "t_start": 1, "t_end": 2, "evidence": [{"id": "g"}]},
              {"claim_id": 2, "text": "kept", "commentary": "kept", "t_start": 3, "t_end": 4, "evidence": [{"id": "h"}]}]
    out = audit_explanation(Client(), {"summary": "x", "claims": claims}, {}, [])
    assert out["summary"] == "clean" and out["audit_removed"] == ["at the far post"]
    assert [c["text"] for c in out["claims"]] == ["tapped in", "kept"]  # unknown ids cannot add claims
    assert out["claims"][0]["evidence"] == [{"id": "g"}] and out["claims"][0]["t_start"] == 1


def test_name_ids_hides_internal_ids():
    from goal_explainer.explain import name_ids
    names = {2016: "blue player 9", 1004: "white player 4", 2001: "blue player 2"}
    assert name_ids("player 2016 got clear of defender 1004, pass to 2001", names) == \
        "blue player 9 got clear of white player 4, pass to blue player 2"
    assert name_ids("Player 7777 in 2026", names) == "Player 7777 in 2026"


def test_one_shirt_number_names_one_player():
    from goal_explainer.explain import player_names
    tracks = {"kits": {"team_a": "blue"}, "frames": [{"objects": [
        {"object_id": 1, "cls": "player", "team": "A"}, {"object_id": 2, "cls": "player", "team": "A"}]}]}
    cue = lambda pid: {"player_id": pid, "jersey_number": "32", "visible": True}  # noqa: E731
    names = player_names(tracks, [cue(1), cue(2), cue(2)])
    assert names[2] == "blue #32" and names[1] == "blue player 1"


def test_why_facts_measure_numbers_and_gaps():
    from goal_explainer.explain import _collect_valid_ids, add_why_facts
    def player(pid, team, x, y):
        return {"player_id": pid, "team": team, "x_m": x, "y_m": y, "timestamp": 1.0}
    players = [player(0, "ball", 90, 34), player(1, "A", 92, 34), player(2, "A", 95, 40), player(3, "A", 80, 10),
               player(11, "B", 97, 20), player(12, "B", 96, 26), player(13, "B", 97, 50), player(14, "B", 70, 34),
               player(99, "B", 104, 34)]  # the keeper must not count
    events = {"attacking_direction": "right", "goals": [{"t_shot": 1.0}],
              "passes": [{"frame_idx": 25, "timestamp": 1.0}]}
    add_why_facts(events, [{"frame_idx": 25, "players": players}], {99})
    text = events["why"][0]["description"]
    assert "2 attackers v 1 defenders within 15 m" in text
    assert "3 defenders goal side" in text
    assert "2 attackers v 3 defenders in the penalty area" in text
    assert "3 attackers with no defender within 5 m" in text
    assert "16 m between 14 and 13" in text
    assert "why_0" in _collect_valid_ids(events, [])

