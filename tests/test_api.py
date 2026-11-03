"""Tests for Meta Model API client and caching."""

import json
from pathlib import Path
import pytest

from playwhy.api import MetaAPIClient, get_client, SAM_MODEL, SPARK_MODEL


def test_client_init(tmp_path: Path):
    client = MetaAPIClient(
        base_url="https://api.example.com",
        api_key="secret",
        cache_dir=tmp_path / "cache",
    )
    assert client.base_url == "https://api.example.com"
    assert client.api_key == "secret"
    assert (tmp_path / "cache").is_dir()


def test_cache_key_deterministic(tmp_path: Path):
    client = MetaAPIClient(cache_dir=tmp_path)
    p1 = {"model": "sam-3.1", "prompts": ["ball", "player"]}
    p2 = {"prompts": ["ball", "player"], "model": "sam-3.1"}
    # Sort keys should ensure identical hash
    assert client._cache_key(p1) == client._cache_key(p2)


def test_cache_read_write(tmp_path: Path):
    client = MetaAPIClient(cache_dir=tmp_path)
    payload = {"query": "test"}
    key = client._cache_key(payload)

    # Cache miss
    assert client._read_cache(key) is None

    # Write cache
    data = {"result": "success", "score": 0.99}
    client._write_cache(key, data)

    # Cache hit
    cached = client._read_cache(key)
    assert cached == data


def test_get_client_factory():
    client = get_client("test_clip_123")
    assert client.cache_dir is not None
    assert "test_clip_123" in str(client.cache_dir)


# Real SAM 3.1 output ("soccer goal" on an 848x478 frame): two objects with masks.
_SAM_GOALS = (
    "<0f>0<|box;x1=153;y1=147;x2=198;y2=160;w=848;h=478;c=0.7344398498535156|><|mask;x=0;y=0;c=0.7344398498535156;data=14,46,!!!!!=I5eKRjowgB.ozFUQ'M$jh0cHe=lS-V|>,"
    "1<|box;x1=483;y1=292;x2=563;y2=341;w=848;h=478;c=0.9298245906829834|><|mask;x=0;y=0;c=0.9298245906829834;data=50,81,!!!!!YCcewx.s{`_gc_9fMt)N%3K!JXWKXqaLl4daU6y8[$q-HLsh7Kz8I&[%6)a9Iv52W|>\n"
)


def test_parse_sam_text_and_masks():
    from playwhy.api import mask_to_frame, parse_sam_text
    dets = sorted(parse_sam_text(_SAM_GOALS), key=lambda d: d["id"])
    # boxes are half-open: SAM's inclusive x2/y2 + 1
    assert [(d["frame"], d["id"], d["bbox"]) for d in dets] == [
        (0, 0, [153, 147, 199, 161]), (0, 1, [483, 292, 564, 342]),
    ]
    assert abs(dets[1]["conf"] - 0.9298) < 1e-3
    m = mask_to_frame(dets[1], (478, 848))
    ys, xs = m.nonzero()
    assert m.shape == (478, 848) and len(xs) > 500
    assert xs.min() >= 483 and xs.max() < 564 and ys.min() >= 292 and ys.max() < 342
    assert parse_sam_text("") == []


def test_load_dotenv(tmp_path, monkeypatch):
    from playwhy.api import load_dotenv
    env = tmp_path / ".env"
    env.write_text('# comment\nGE_TEST_KEY="abc"\nGE_TEST_OTHER=x=y\n\nGE_TEST_SET=fromfile\n', encoding="utf-8")
    monkeypatch.delenv("GE_TEST_KEY", raising=False)
    monkeypatch.delenv("GE_TEST_OTHER", raising=False)
    monkeypatch.setenv("GE_TEST_SET", "fromenv")
    load_dotenv(env)
    import os
    assert os.environ["GE_TEST_KEY"] == "abc"
    assert os.environ["GE_TEST_OTHER"] == "x=y"
    assert os.environ["GE_TEST_SET"] == "fromenv"


def test_build_prompts_skips_unknown_and_shared_colours():
    from playwhy.track import build_prompts
    kits = {"team_a": "white", "team_b": "black", "goalkeeper_a": "yellow",
            "goalkeeper_b": "white", "referee": "unknown"}
    assert build_prompts(kits) == [
        ("soccer ball", "ball", None),
        ("player in white", "player", "A"),
        ("player in black", "player", "B"),
        ("player in yellow", "goalkeeper", "A"),
    ]


def test_find_duplicates_and_drop_boxes():
    from playwhy.track import drop_boxes, find_duplicates
    def frame(i, objs):
        return {"frame_idx": i, "timestamp": i / 25, "objects": objs}
    keeper_as_white = {"object_id": 1009, "cls": "player", "bbox": [100, 100, 120, 150]}
    keeper = {"object_id": 4000, "cls": "goalkeeper", "bbox": [101, 101, 121, 151]}
    other = {"object_id": 1001, "cls": "player", "bbox": [300, 100, 320, 150]}
    same_prompt = {"object_id": 1002, "cls": "player", "bbox": [300, 100, 320, 150]}  # same prompt: never a pair
    tracks = [frame(i, [dict(keeper_as_white), dict(keeper), dict(other), dict(same_prompt)]) for i in range(6)]
    tracks.append(frame(6, [dict(keeper_as_white)]))  # outside the overlap: must survive
    assert find_duplicates(tracks[:4]) == {}  # 4 frames: two players briefly crossing
    pairs = find_duplicates(tracks)
    assert pairs == {(1009, 4000): [0, 1, 2, 3, 4, 5]}
    drop_boxes(tracks, 1009, pairs[(1009, 4000)])
    assert all(o["object_id"] != 1009 for tf in tracks[:6] for o in tf["objects"])
    assert [o["object_id"] for o in tracks[6]["objects"]] == [1009]


def test_resolve_duplicates_when_one_id_is_in_two_pairs(tmp_path, monkeypatch):
    # 1009 doubles both a keeper (4000) and a black player (2001); resolving the
    # first pair drops 1009 from frames the second pair was found in.
    from playwhy import track
    def frame(i):
        return {"frame_idx": i, "timestamp": i / 25, "objects": [
            {"object_id": 1009, "cls": "player", "bbox": [100, 100, 120, 150]},
            {"object_id": 4000, "cls": "goalkeeper", "bbox": [101, 101, 121, 151]},
            {"object_id": 2001, "cls": "player", "bbox": [100, 100, 121, 150]},
        ]}
    tracks = [frame(i) for i in range(6)]
    prompts = [("soccer ball", "ball", None), ("player in white", "player", "A"), ("player in black", "player", "B"),
               ("player in yellow", "goalkeeper", "A"), ("player in light blue", "goalkeeper", "B")]
    monkeypatch.setattr(track, "read_frame", lambda *a: None)
    picks = iter(["light blue", "black", "black"])
    monkeypatch.setattr(track, "pick_shirt", lambda *a: next(picks))
    track.resolve_duplicates(None, tmp_path / "v.mp4", tracks, prompts)
    assert all(1009 not in {o["object_id"] for o in tf["objects"]} for tf in tracks)


def test_find_goal_validates_spark_answer(tmp_path):
    from playwhy.read import find_goal, players_near_goal

    class Fake:
        def __init__(self, answer): self.answer = answer
        def upload_file(self, _): return "file-1"
        def spark_video_json(self, prompt, file_id): return self.answer

    ok = find_goal(Fake({"goal": True, "t_shot": 20.5, "t_goal": 21.0, "description": "near post"}), tmp_path, 24.0)
    assert ok == {"event_id": "goal_0", "timestamp": 21.0, "t_shot": 20.5, "description": "near post"}
    assert find_goal(Fake({"goal": False, "t_shot": None}), tmp_path, 24.0) is None
    assert find_goal(Fake({"goal": True, "t_shot": 30.0, "t_goal": 31.0}), tmp_path, 24.0) is None  # past the end
    pitch = [{"frame_idx": 5, "players": [
        {"player_id": 1, "team": "A", "x_m": 100.0, "y_m": 34.0},
        {"player_id": 2, "team": "B", "x_m": 104.0, "y_m": 34.0},
        {"player_id": 3, "team": "A", "x_m": 60.0, "y_m": 34.0},
        {"player_id": -1, "team": "ball", "x_m": 105.0, "y_m": 34.0}]}]
    assert players_near_goal(pitch, 5, 105.0, k=2) == [2, 1]
