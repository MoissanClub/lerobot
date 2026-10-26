from unittest.mock import MagicMock

import numpy as np

from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.scripts import lerobot_record as script
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.so_leader import SO101LeaderConfig


def test_manual_record_real_writer_times_out_and_finalizes_without_pynput(tmp_path, monkeypatch):
    import datasets.config

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    joints = [f"joint{i}.pos" for i in range(6)]
    robot = MagicMock()
    robot.name = "so_follower"
    robot.is_connected = True
    robot.action_features = dict.fromkeys(joints, float)
    robot.observation_features = {
        **dict.fromkeys(joints, float),
        "camera_0": (24, 32, 3),
        "camera_1": (24, 32, 3),
    }
    robot.cameras = {"camera_0": object(), "camera_1": object()}
    robot.get_observation.return_value = {
        **dict.fromkeys(joints, 0.0),
        "camera_0": np.zeros((24, 32, 3), np.uint8),
        "camera_1": np.zeros((24, 32, 3), np.uint8),
    }
    robot.send_action.side_effect = lambda action: action
    teleop = MagicMock(spec=Teleoperator)
    teleop.get_action.return_value = dict.fromkeys(joints, 1.0)
    teleop.is_connected = True
    monkeypatch.setattr(script, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(script, "make_teleoperator_from_config", lambda _: teleop)
    monkeypatch.setattr(
        script,
        "init_keyboard_listener",
        lambda: (_ for _ in ()).throw(AssertionError("Global listener must not run")),
    )
    command = tmp_path / "controls.txt"
    command.write_text("s\n")
    cfg = script.RecordConfig(
        robot=SO101FollowerConfig(port="/dev/fake"),
        teleop=SO101LeaderConfig(port="/dev/fake"),
        dataset=script.DatasetRecordConfig(
            repo_id="local/test_manual_entry",
            root=tmp_path / "data",
            single_task="synthetic",
            fps=10,
            num_episodes=1,
            episode_time_s=0.3,
            video=True,
            vcodec="h264",
            push_to_hub=False,
        ),
        manual_control=True,
        manual_control_file=str(command),
        play_sounds=False,
    )
    cfg.catch_up_warning.enabled = False
    dataset = script.record(cfg)
    assert dataset.num_episodes == 1 and dataset.num_frames >= 2
    robot.disconnect.assert_called_once()
    teleop.disconnect.assert_called_once()
    assert robot.send_action.call_count >= 2
