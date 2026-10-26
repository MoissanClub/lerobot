import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lerobot.scripts.lerobot_camera_views import rotate_image, save_views
from lerobot.scripts.lerobot_replay import prepare_recorded_previews, show_replay_frame


def test_view_save_preserves_calibration_and_replay_dataset_match(tmp_path):
    record = {
        "robot": {"calibration_dir": "keep_this", "cameras": {}},
        "dataset": {"repo_id": "local/old", "root": str(tmp_path / "datasets/old")},
    }
    replay = {
        "robot": {"calibration_dir": "keep_this", "cameras": {}},
        "dataset": {"repo_id": "local/old", "root": str(tmp_path / "datasets/old"), "episode": 0},
    }
    for name, cfg in [("record", record), ("replay", replay)]:
        (tmp_path / (name + ".json")).write_text(json.dumps(cfg))
    views = [
        {"index": 2, "rotation": 90, "width": 640, "height": 480},
        {"index": 1, "rotation": 0, "width": 640, "height": 480},
    ]
    cameras = save_views(tmp_path, views)
    assert cameras["camera_0"]["width"] == 480 and cameras["camera_0"]["height"] == 640
    saved = json.loads((tmp_path / "record.json").read_text())
    rp = json.loads((tmp_path / "replay.json").read_text())
    assert saved["robot"]["calibration_dir"] == "keep_this"
    assert saved["dataset"]["video"] and saved["dataset"]["vcodec"] == "h264"
    assert saved["dataset"]["root"] == rp["dataset"]["root"]
    assert rp["display_recorded_cameras"] and rp["display_data"]
    assert len(list(tmp_path.glob("*.before-cameras-*.json"))) == 2
    assert rotate_image(np.zeros((2, 3, 3), np.uint8), 90).shape == (3, 2, 3)


class Dataset:
    meta = SimpleNamespace(video_keys=["observation.images.camera_0", "observation.images.camera_1"])
    hf_dataset = [{"episode_index": 0}, {"episode_index": 1}, {"episode_index": 1}]

    def __getitem__(self, index):
        return {k: np.full((3, 8, 12), index / 3, dtype=np.float32) for k in self.meta.video_keys}


def test_reference_previews_match_selected_episode_and_have_memory_limit():
    previews = prepare_recorded_previews(Dataset(), 1, 6, 100000)
    assert len(previews) == 2 and set(previews[0]) == {"camera_0", "camera_1"}
    image = cv2.imdecode(np.frombuffer(previews[0]["camera_0"], np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (4, 6, 3) and abs(int(image[0, 0, 0]) - 85) <= 2
    with pytest.raises(ValueError, match="memory budget"):
        prepare_recorded_previews(Dataset(), 1, 6, 1)


def test_replay_live_and_reference_views_share_action_frame(monkeypatch):
    from lerobot.scripts import lerobot_replay as replay

    calls = []
    monkeypatch.setattr(replay.rr, "set_time_sequence", lambda *args: calls.append(args))
    monkeypatch.setattr(replay, "log_rerun_data", lambda obs, act, **kwargs: calls.append((obs, act)))
    preview = prepare_recorded_previews(Dataset(), 1, 6, 100000)[0]
    live = {"camera_0": np.zeros((8, 12, 3), np.uint8), "camera_1": np.zeros((8, 12, 3), np.uint8)}
    show_replay_frame(live, {"joint.pos": 2}, preview, 7)
    assert calls[0] == ("replay_frame", 7)
    assert set(calls[1][0]) == {"camera_0", "camera_1", "recorded.camera_0", "recorded.camera_1"}
    assert calls[1][1] == {"joint.pos": 2}
