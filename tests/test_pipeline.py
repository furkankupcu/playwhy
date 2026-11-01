"""Integration test for pipeline data flow."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from goal_explainer import geometry, graphs, explain


def test_data_pipeline_flow(tmp_path: Path, monkeypatch):
    # Set runs directory to tmp_path / "runs"
    runs_dir = tmp_path / "runs"
    clip_id = "test_clip"
    clip_dir = runs_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(tmp_path)

    # 1. Create synthetic tracks.json
    tracks_data = {
        "fps": 25.0,
        "num_frames": 10,
        "frames": [
            {
                "frame_idx": i,
                "timestamp": i * 0.04,
                "objects": [
                    {
                        "object_id": 1,
                        "cls": "player",
                        "team": "A",
                        "foot_xy": [100 + i * 5, 200],
                        "bbox": [90 + i * 5, 160, 110 + i * 5, 200],
                    },
                    {
                        "object_id": 2,
                        "cls": "player",
                        "team": "B",
                        "foot_xy": [300, 250],
                        "bbox": [290, 210, 310, 250],
                    },
                    {
                        "object_id": -1,
                        "cls": "ball",
                        "team": "ball",
                        "foot_xy": [100 + i * 5, 200],
                        "bbox": [98 + i * 5, 195, 102 + i * 5, 205],
                    },
                ],
            }
            for i in range(10)
        ],
    }
    (clip_dir / "tracks.json").write_text(json.dumps(tracks_data), encoding="utf-8")

    # 2. Create synthetic homography.json
    # H is identity matrix (1px = 1m offset for testing)
    H_eye = np.diag([4.0, 4.0, 1.0]).tolist()  # 4 px per metre: fixture pixels land on the pitch
    homography_data = {
        "fps": 25.0,
        "method": "c_manual",
        "frames": [
            {
                "frame_idx": i,
                "H": H_eye,
                "method": "c_manual",
                "error": 1.2,
                "reliable": True,
            }
            for i in range(10)
        ],
    }
    (clip_dir / "homography.json").write_text(json.dumps(homography_data), encoding="utf-8")

    # 3. Run Step 3: Geometry
    events_path = geometry.run(clip_id)
    assert events_path.is_file()
    assert (clip_dir / "pitch.json").is_file()

    # 4. Check pitch.json structure
    pitch_data = json.loads((clip_dir / "pitch.json").read_text(encoding="utf-8"))
    assert len(pitch_data) == 10

    # 5. Step 6: graphs (computation only; the MP4 render needs a real video)
    g = graphs.compute(clip_id)
    assert g["attacking_team"] in ("A", "B")
    assert len(g["frames"]) == 10

    # 6. Validate explanation logic
    events_content = json.loads((clip_dir / "events.json").read_text(encoding="utf-8"))
    valid_ids = explain._collect_valid_ids(events_content, cues=[])
    assert isinstance(valid_ids, set)


def test_later_shot_overrules_goal_time_and_after_goal_play_is_dropped():
    from goal_explainer.read import drop_after_goal, reconcile_goal
    pitch = [{"frame_idx": 500, "players": [{"player_id": 7, "team": "A", "timestamp": 20.0, "x_m": 90, "y_m": 30}]},
             {"frame_idx": 540, "players": [{"player_id": 0, "team": "ball", "timestamp": 21.6, "x_m": 104, "y_m": 34}]}]
    events = {"passes": [{"team": "A", "timestamp": 23.5}],
              "shots": [{"timestamp": 20.0, "frame_idx": 500, "shooter_id": 7, "goal_xy": [105, 34]}]}
    late = reconcile_goal({"timestamp": 17.0, "t_shot": 16.0}, events, pitch)
    assert late["t_shot"] == 20.0 and late["timestamp"] == 21.6
    # a goal after an earlier shot keeps its time
    assert reconcile_goal({"timestamp": 22.0, "t_shot": 21.0}, events, pitch)["timestamp"] == 22.0
    drop_after_goal(events, late["timestamp"])
    assert events["passes"] == [] and len(events["shots"]) == 1


def test_person_fallback_adds_named_tracks_only_for_a_short_team(monkeypatch, tmp_path):
    import numpy as np
    from goal_explainer import track

    box = lambda x: [x, 10, x + 10, 40]  # noqa: E731
    frames = [{"frame_idx": i, "timestamp": i / 25, "objects": [
        {"object_id": 1000 + k, "cls": "player", "team": "A", "bbox": box(k * 30), "conf": 0.9, "foot_xy": [0, 0]} for k in range(8)
    ]} for i in range(3)]

    class Client:
        def sam(self, prompt, video_file_id=None):
            assert prompt == "person"
            return "stub"

        def spark_json(self, prompt, schema, images=None):
            assert "white shirt" in prompt and images
            return {"people": [{"number": 1, "role": "team_b"}, {"number": 2, "role": "other"}]}

    people = [{"id": 5, "frame": f, "bbox": box(600), "conf": 0.7} for f in range(3)]  # new person
    people += [{"id": 6, "frame": f, "bbox": box(700), "conf": 0.7} for f in range(3)]  # staff
    people += [{"id": 7, "frame": f, "bbox": box(0), "conf": 0.7} for f in range(3)]  # already tracked
    monkeypatch.setattr(track, "parse_sam_text", lambda text: people)
    monkeypatch.setattr(track, "read_frame", lambda path, i: np.zeros((100, 800, 3), np.uint8))
    assert track.fill_missing_people(Client(), tmp_path, frames, "f", {"team_b": "white"}, 5000) == 1
    assert {o["object_id"] for o in frames[0]["objects"] if o["team"] == "B"} == {5005}
    assert not any(o["object_id"] in (5006, 5007) for o in frames[0]["objects"])
    # both teams full: no extra request
    for f in frames:
        f["objects"] += [{"object_id": 2000 + k, "cls": "player", "team": "B", "bbox": box(k), "conf": 1, "foot_xy": [0, 0]} for k in range(7)]
    monkeypatch.setattr(track, "parse_sam_text", lambda text: 1 / 0)
    assert track.fill_missing_people(Client(), tmp_path, frames, "f", {}, 5000) == 0

