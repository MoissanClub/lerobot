import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_replay import prepare_recorded_previews
from lerobot.utils.demo_session import DemoSession


def test_two_real_video_episodes_append_transactionally_then_select_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf-cache"))
    # datasets reads this setting at import time; patch the installed config too.
    import datasets.config

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    calibration = tmp_path / "cal"
    calibration.mkdir()
    (calibration / "arm.json").write_text("{}")
    config = {
        "robot": {"id": "arm", "calibration_dir": str(calibration), "cameras": {}},
        "teleop": {"id": "arm", "calibration_dir": str(calibration)},
        "dataset": {"fps": 10, "video": True, "vcodec": "h264"},
    }
    session = DemoSession.create(tmp_path / "session", config, 25)
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": ["joint"] * 6},
        "action": {"dtype": "float32", "shape": (6,), "names": ["joint"] * 6},
    }
    for camera in ["camera_0", "camera_1"]:
        features["observation.images." + camera] = {
            "dtype": "video",
            "shape": (24, 32, 3),
            "names": ["height", "width", "channels"],
        }
    for episode in range(2):
        attempt = session.stage()
        root = attempt / "data"
        if episode == 0:
            ds = LeRobotDataset.create(
                session.meta["repo_id"],
                root=root,
                fps=10,
                features=features,
                robot_type="so_follower",
                vcodec="h264",
                video_backend="pyav",
            )
        else:
            ds = LeRobotDataset(session.meta["repo_id"], root=root, video_backend="pyav", vcodec="h264")
        for frame in range(3):
            sample = {
                "observation.state": np.full(6, frame, np.float32),
                "action": np.full(6, episode * 10 + frame, np.float32),
                "task": "synthetic",
            }
            for camera in ["camera_0", "camera_1"]:
                sample["observation.images." + camera] = np.full((24, 32, 3), frame * 50, np.uint8)
            ds.add_frame(sample)
        ds.save_episode(parallel_encoding=False)
        ds.finalize()
        assert session.commit(attempt, 0)
    assert session.meta["current_episode"] == 1
    loaded = LeRobotDataset(
        session.meta["repo_id"], root=session.current_root, episodes=[1], video_backend="pyav"
    )
    rows = loaded.hf_dataset.filter(lambda r: r["episode_index"] == 1)
    assert [float(v[0]) for v in rows["action"]] == [10, 11, 12]
    previews = prepare_recorded_previews(loaded, 1, 16, 100000)
    assert len(previews) == 3 and set(previews[0]) == {"camera_0", "camera_1"}
    session.close()
