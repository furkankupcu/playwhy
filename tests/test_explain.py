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
