# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
import os
import time

import numpy as np
import pytest

from lerobot.cameras.frame_channel import FrameWriter, read_frame
from lerobot.teleoperators.xr_controllers.camera_display import CameraDisplay, VideoConfig
from lerobot.teleoperators.xr_controllers.video_session import VideoControllerSession


def test_frames_restart_ownership_and_copy(tmp_path):
    path = tmp_path / "camera.rgb"
    writer = FrameWriter(path, 8, 6)
    image = np.full((6, 8, 3), 90, dtype=np.uint8)
    try:
        with pytest.raises(BlockingIOError):
            FrameWriter(path, 8, 6)
        assert writer.publish(image, {"captured_monotonic_ns": time.monotonic_ns()})
        info, pixels = read_frame(path)
        assert info["sequence"] == 1
        assert writer.publish(image * 2, {"captured_monotonic_ns": time.monotonic_ns()})
        assert np.all(pixels == 90)
    finally:
        writer.close()
    assert read_frame(path) is None
    replacement = FrameWriter(path, 8, 6)
    try:
        replacement.publish(image, {"captured_monotonic_ns": time.monotonic_ns()})
        assert read_frame(path)[0]["session_id"] != info["session_id"]
    finally:
        replacement.close()


@pytest.mark.parametrize("offset", [-2_000_000_000, 2_000_000_000])
def test_stale_future_frames(tmp_path, offset):
    writer = FrameWriter(tmp_path / "rgb", 8, 6)
    try:
        writer.publish(
            np.zeros((6, 8, 3), dtype=np.uint8), {"captured_monotonic_ns": time.monotonic_ns() + offset}
        )
        assert read_frame(writer.path) is None
    finally:
        writer.close()


@pytest.mark.parametrize("width,height", [(0, 1), (2, -1), (4097, 1), (1.5, 1), (True, 1)])
def test_dimensions(tmp_path, width, height):
    with pytest.raises(ValueError):
        FrameWriter(tmp_path / "rgb", width, height)
    with pytest.raises(ValueError):
        VideoConfig(width=width, height=height)


def test_bad_payload_and_closed_writer(tmp_path):
    writer = FrameWriter(tmp_path / "rgb", 8, 6)
    try:
        with pytest.raises(ValueError):
            writer.publish(np.zeros((6, 8, 3)), {})
        with pytest.raises(ValueError):
            writer.publish(np.zeros((6, 8, 3), dtype=np.uint8), {})
    finally:
        writer.close()
    with pytest.raises(RuntimeError):
        writer.publish(np.zeros((6, 8, 3), dtype=np.uint8), {})


def fake_video_worker(config, video, frames, stop, ready, errors):
    frames.put({"captured_at": 1.0, "left.tracked": True, "right.tracked": True})
    ready.set()
    stop.wait(10)


def failed_video_worker(config, video, frames, stop, ready, errors):
    errors.put("synthetic startup failure")
    ready.set()


def test_worker_lifecycle_stale_timestamp_and_reconnect():
    session = VideoControllerSession(None, None, worker=fake_video_worker, startup_timeout=5)
    for _ in range(2):
        session.connect()
        process = session.process
        try:
            assert session.read()["captured_at"] == 1.0
            assert session.read()["captured_at"] == 1.0
        finally:
            session.close()
        assert not process.is_alive()
    with pytest.raises(RuntimeError):
        session.read()


def test_worker_failure_cleanup():
    session = VideoControllerSession(None, None, worker=failed_video_worker, startup_timeout=5)
    with pytest.raises(RuntimeError):
        session.connect()
    assert session.process is None


@pytest.mark.skipif(
    not os.environ.get("G1_VIDEO_GPU_TESTS"),
    reason="Opt-in installed Isaac Teleop offscreen Vulkan/CUDA test",
)
def test_real_offscreen_delivery_and_recovery(tmp_path):
    channel = tmp_path / "camera.rgb"
    writer = FrameWriter(channel, 64, 48)
    display = None
    try:
        display = CameraDisplay(
            VideoConfig(channel=str(channel), width=320, height=240, expected_source="test", max_age_s=10),
            offscreen=True,
        )
        for brightness in (80, 180):
            writer.publish(
                np.full((48, 64, 3), brightness, dtype=np.uint8),
                {"captured_monotonic_ns": time.monotonic_ns(), "embodiment": "test"},
            )
            display.update_camera()
            display.render()
            np.testing.assert_allclose(display.readback()[120, 160, :3], [brightness] * 3, atol=3)
        writer.close()
        display.update_camera()
        assert display.status != "live"
        writer = FrameWriter(channel, 64, 48)
        writer.publish(
            np.full((48, 64, 3), 120, dtype=np.uint8),
            {"captured_monotonic_ns": time.monotonic_ns(), "embodiment": "wrong"},
        )
        display.update_camera()
        assert display.status == "Camera source mismatch"
        writer.publish(
            np.full((48, 64, 3), 120, dtype=np.uint8),
            {"captured_monotonic_ns": time.monotonic_ns(), "embodiment": "test"},
        )
        display.update_camera()
        display.render()
        assert display.status == "live"
        np.testing.assert_allclose(display.readback()[120, 160, :3], [120] * 3, atol=3)
    finally:
        if display is not None:
            display.close()
        writer.close()
