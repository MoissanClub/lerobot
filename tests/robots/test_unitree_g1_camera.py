# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Camera-only CLI mode must never construct hardware motion or simulation backends."""

from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.cameras.frame_channel import read_frame
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.unitree_g1 import g1_motion as module


def config(tmp_path, **overrides):
    values = {
        "mode": "camera",
        "cameras": {"head": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=30)},
        "video_channel": str(tmp_path / "camera.rgb"),
    }
    values.update(overrides)
    return module.UnitreeG1MotionConfig(**values)


def test_camera_frames_and_quit_without_motion(tmp_path, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Camera mode opened a motion backend"))
    for name in ("G1ArmSDK", "G1BaseMotion", "G1VRKinematics"):
        monkeypatch.setattr(module, name, forbidden)
    camera = Mock()
    pixels = np.full((480, 640, 3), 123, dtype=np.uint8)
    camera.async_read.return_value = pixels
    monkeypatch.setattr(module, "make_cameras_from_configs", lambda _: {"head": camera})
    robot = module.UnitreeG1Motion(config(tmp_path))
    robot.connect()
    try:
        assert robot.observation_features == {"head": (480, 640, 3)}
        assert np.array_equal(robot.get_observation()["head"], pixels)
        metadata, frame = read_frame(robot.config.video_channel, 0.5)
        assert np.array_equal(frame, pixels)
        assert metadata["embodiment"] == "g1-29-physical-camera"
        assert robot.send_action({"control.start": True, "base.vx": 1.0}) == {"control.quit": False}
        camera.async_read.side_effect = TimeoutError
        assert robot.get_observation() == {}
        assert read_frame(robot.config.video_channel, 0.5)[0]["sequence"] == metadata["sequence"]
        camera.async_read.side_effect = None
        robot.get_observation()
        assert read_frame(robot.config.video_channel, 0.5)[0]["sequence"] > metadata["sequence"]
        with pytest.raises(KeyboardInterrupt):
            robot.send_action({"control.quit": True})
    finally:
        robot.disconnect()
    camera.disconnect.assert_called_once()
    forbidden.assert_not_called()
    assert not robot.is_connected
    assert not (tmp_path / "camera.rgb").exists()


def test_warmup_failure_closes_camera(tmp_path, monkeypatch):
    camera = Mock()
    camera.async_read.side_effect = TimeoutError
    monkeypatch.setattr(module, "make_cameras_from_configs", lambda _: {"head": camera})
    robot = module.UnitreeG1Motion(config(tmp_path))
    with pytest.raises(TimeoutError):
        robot.connect()
    camera.disconnect.assert_called_once()
    assert not robot.is_connected


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_motion": True},
        {"enable_locomotion": True},
        {"onscreen": True},
        {"cameras": {}},
        {"video_channel": None},
    ],
)
def test_camera_config_rejects_motion_and_missing_camera(tmp_path, overrides):
    with pytest.raises(ValueError):
        config(tmp_path, **overrides)
