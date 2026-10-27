"""Integration test for pipeline data flow."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from goal_explainer import geometry, replay, explain


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

    # 5. Run Step 6: Replay
    replay_path = replay.run(clip_id, fps=25.0)
    assert replay_path.is_file()
    assert replay_path.stat().st_size > 0

    # 6. Validate explanation logic
    events_content = json.loads((clip_dir / "events.json").read_text(encoding="utf-8"))
    valid_ids = explain._collect_valid_ids(events_content, cues=[])
    assert isinstance(valid_ids, set)
