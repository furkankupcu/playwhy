"""Tests for Meta Model API client and caching."""

import json
from pathlib import Path
import pytest

from goal_explainer.api import MetaAPIClient, get_client, SAM_MODEL, SPARK_MODEL


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


def test_parse_sam_text_frames_ids_boxes():
    from goal_explainer.api import parse_sam_text
    text = (
        "<0f>0<|box;x1=181;y1=365;x2=556;y2=486;w=1516;h=1600|><|mask;x=0;y=0;data=122,376,~ab|>,"
        "3<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10;c=0.9|><|mask;x=0;y=0;data=1,1,!z|>\n"
        "<2f>3<|box;x1=5;y1=6;x2=7;y2=8;w=10;h=10|>\n"
    )
    dets = parse_sam_text(text)
    assert [(d["frame"], d["id"], d["bbox"]) for d in dets] == [
        (0, 0, [181, 365, 556, 486]), (0, 3, [1, 2, 3, 4]), (2, 3, [5, 6, 7, 8]),
    ]
    assert dets[1]["conf"] == 0.9 and dets[0]["conf"] == 1.0
    assert parse_sam_text("") == []


def test_load_dotenv(tmp_path, monkeypatch):
    from goal_explainer.api import load_dotenv
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
