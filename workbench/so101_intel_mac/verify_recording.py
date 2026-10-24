"""Exercise local dataset/video recording using synthetic data, without hardware."""

import tempfile
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    with tempfile.TemporaryDirectory(prefix="so101-recording-") as directory:
        root = Path(directory) / "dataset"
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joint"] * 6,
            },
            "action": {"dtype": "float32", "shape": (6,), "names": ["joint"] * 6},
        }
        for camera in ("camera_0", "camera_1"):
            features[f"observation.images.{camera}"] = {
                "dtype": "video",
                "shape": (48, 64, 3),
                "names": ["height", "width", "channels"],
            }
        dataset = LeRobotDataset.create(
            repo_id="local/recording_check",
            root=root,
            fps=10,
            features=features,
            robot_type="so101_follower",
            vcodec="h264",
            video_backend="pyav",
        )
        for index in range(8):
            frame = {
                "observation.state": np.full(6, index, dtype=np.float32),
                "action": np.full(6, index + 1, dtype=np.float32),
                "task": "Synthetic recording check",
            }
            for camera in ("camera_0", "camera_1"):
                frame[f"observation.images.{camera}"] = np.full(
                    (48, 64, 3), index * 20, dtype=np.uint8
                )
            dataset.add_frame(frame)
        dataset.save_episode()
        dataset.finalize()
        loaded = LeRobotDataset(
            "local/recording_check", root=root, video_backend="pyav"
        )
        assert loaded.num_episodes == 1 and len(loaded) == 8
        sample = loaded[3]
        for camera in ("camera_0", "camera_1"):
            assert sample[f"observation.images.{camera}"].shape == (3, 48, 64)
        assert sample["observation.state"].tolist() == [3.0] * 6
        print(
            "PASS: record, encode two camera videos, finalize, reload, and decode 8 synthetic frames."
        )


if __name__ == "__main__":
    main()
