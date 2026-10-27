"""Web API: upload validation, id checks, result shape (pipeline stubbed out)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import server


def _clip(path: Path, seconds: float, fps: int = 10) -> bytes:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (64, 48))
    for i in range(int(seconds * fps)):
        w.write(np.full((48, 64, 3), i % 255, np.uint8))
    w.release()
    return path.read_bytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    started = []
    monkeypatch.setattr(server, "_start", lambda cid, *a, **k: started.append(cid) or True)
    c = TestClient(server.app)
    c.started = started
    return c


def test_upload_starts_pipeline_and_status(client, tmp_path):
    data = _clip(tmp_path / "a.mp4", 2)
    r = client.post("/api/clips", files={"file": ("a.mp4", data, "video/mp4")})
    assert r.status_code == 200
    cid = r.json()["clip_id"]
    assert client.started == [cid]
    st = client.get(f"/api/clips/{cid}/status").json()
    assert [s["status"] for s in st["steps"]] == ["pending"] * 6
    assert client.get(f"/api/clips/{cid}/frame0.jpg").headers["content-type"] == "image/jpeg"


def test_upload_rejects_long_and_garbage(client, tmp_path):
    long = _clip(tmp_path / "b.mp4", 30)
    assert client.post("/api/clips", files={"file": ("b.mp4", long, "video/mp4")}).status_code == 400
    assert client.post("/api/clips", files={"file": ("c.mp4", b"not a video", "video/mp4")}).status_code == 400
    assert client.started == []


def test_bad_ids_are_404(client):
    assert client.get("/api/clips/..%2F..%2Fetc/status").status_code == 404
    assert client.get("/api/clips/0123456789abcdef/status").status_code == 404


def test_result_shape(client, tmp_path):
    data = _clip(tmp_path / "d.mp4", 2)
    cid = client.post("/api/clips", files={"file": ("d.mp4", data, "video/mp4")}).json()["clip_id"]
    run = Path("runs") / cid
    assert client.get(f"/api/clips/{cid}/result").status_code == 409
    (run / "explanation.json").write_text(json.dumps({"claims": [], "summary": "s"}))
    eye = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    (run / "homography.json").write_text(json.dumps([{"reliable": True, "H": eye}, {"reliable": False, "H": eye}]))
    (run / "tracks.json").write_text(json.dumps({"shirts": {"A": "red", "B": "white"}, "frames": []}))
    body = client.get(f"/api/clips/{cid}/result").json()
    assert body["reliable_share"] == 0.5 and body["shirts"]["A"] == "red"
    assert body["duration"] == pytest.approx(2.0, abs=0.2)
    # calibration needs ≥4 points
    pts = [{"px": [1, 2], "pitch": [3, 4]}] * 3
    assert client.post(f"/api/clips/{cid}/calibrate", json={"points": pts}).status_code == 422
    assert client.post(f"/api/clips/{cid}/calibrate", json={"points": pts * 2}).status_code == 200


def test_graph_video_404_until_rendered(client, tmp_path):
    data = _clip(tmp_path / "e.mp4", 1)
    cid = client.post("/api/clips", files={"file": ("e.mp4", data, "video/mp4")}).json()["clip_id"]
    assert client.get(f"/api/clips/{cid}/graphs.mp4").status_code == 404
    (Path("runs") / cid / "graphs.mp4").write_bytes(b"\x00")
    assert client.get(f"/api/clips/{cid}/graphs.mp4").status_code == 200
