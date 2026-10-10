# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
import multiprocessing as mp
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.cameras.frame_channel import FrameWriter, read_frame
from lerobot.teleoperators.xr_controllers.camera_display import CameraDisplay, VideoConfig
from lerobot.teleoperators.xr_controllers.latest_slot import LatestSlot
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


def controller_sample(captured_at, full_input=False):
    sample = {"captured_at": captured_at}
    for side, sign in (("left", -1), ("right", 1)):
        sample.update(
            {
                f"{side}.tracked": True,
                f"{side}.grip_pos": np.array([sign, 2.0, 3.0]),
                f"{side}.grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                f"{side}.squeeze": 0.25,
                f"{side}.trigger": 0.75,
            }
        )
        if full_input:
            sample.update(
                {
                    f"{side}.stick_x": 0.1 * sign,
                    f"{side}.stick_y": 0.2 * sign,
                    f"{side}.stick_click": side == "left",
                    f"{side}.primary": side == "right",
                    f"{side}.secondary": False,
                }
            )
    if full_input:
        sample.update(
            {
                "head.tracked": True,
                "head.pos": np.array([4.0, 5.0, 6.0]),
                "head.quat": np.array([0.0, 0.0, 0.0, 1.0]),
            }
        )
    return sample


def fake_video_worker(config, video, latest, stop, ready, errors):
    latest.write(controller_sample(1.0))
    ready.set()
    stop.wait(10)


def latest_video_worker(config, video, latest, stop, ready, errors):
    for sequence in range(1, 101):
        latest.write(controller_sample(float(sequence), config.full_input))
    ready.set()
    stop.wait(10)


def concurrent_video_worker(config, video, latest, stop, ready, errors):
    ready.set()
    for sequence in range(1, 501):
        if stop.is_set():
            return
        sample = controller_sample(float(sequence), config.full_input)
        for key, value in sample.items():
            if key == "captured_at":
                continue
            if isinstance(value, np.ndarray):
                sample[key] = np.full(value.shape, sequence, dtype=np.float64)
            elif isinstance(value, bool):
                sample[key] = bool(sequence % 2)
            else:
                sample[key] = float(sequence)
        latest.write(sample)
        time.sleep(0.0005)
    stop.wait(10)


def failed_video_worker(config, video, latest, stop, ready, errors):
    errors.put("synthetic startup failure")
    ready.set()


@pytest.mark.parametrize("full_input", [False, True])
def test_latest_slot_round_trip(full_input):
    slot = LatestSlot(mp.get_context("spawn"), full_input)
    assert slot.read() is None
    expected = controller_sample(1.5, full_input)
    slot.write(expected)
    actual = slot.read()
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], np.ndarray):
            np.testing.assert_array_equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


def test_latest_slot_rejects_malformed_sample_without_losing_latest():
    slot = LatestSlot(mp.get_context("spawn"), False)
    expected = controller_sample(1.5)
    assert slot.write(expected)
    malformed = controller_sample(2.0)
    malformed["left.grip_pos"] = np.zeros(2)
    with pytest.raises(ValueError, match="left.grip_pos must contain 3 value"):
        slot.write(malformed)
    malformed["left.grip_pos"] = np.zeros(3)
    del malformed["right.trigger"]
    with pytest.raises(KeyError, match="right.trigger"):
        slot.write(malformed)
    assert slot.read()["captured_at"] == expected["captured_at"]


def test_latest_slot_lock_timeout_preserves_liveness():
    slot = LatestSlot(mp.get_context("spawn"), False)
    assert slot.write(controller_sample(1.0))
    assert slot.lock.acquire()
    try:
        assert slot.read() is None
        assert not slot.write(controller_sample(2.0))
    finally:
        slot.lock.release()
    assert slot.read()["captured_at"] == 1.0


@pytest.mark.parametrize("full_input", [False, True])
def test_controller_mailbox_keeps_latest_cross_process_sample(full_input):
    config = SimpleNamespace(full_input=full_input)
    session = VideoControllerSession(config, None, worker=latest_video_worker, startup_timeout=5)
    session.connect()
    try:
        assert session.read()["captured_at"] == 100.0
    finally:
        session.close()


@pytest.mark.parametrize("full_input", [False, True])
def test_controller_mailbox_is_consistent_during_concurrent_access(full_input):
    config = SimpleNamespace(full_input=full_input)
    session = VideoControllerSession(config, None, worker=concurrent_video_worker, startup_timeout=5)
    session.connect()
    samples = []
    deadline = time.monotonic() + 2
    try:
        while len(samples) < 50 and time.monotonic() < deadline:
            sample = session.read()
            sequence = sample["captured_at"]
            if not sequence:
                continue
            for key, value in sample.items():
                if key == "captured_at":
                    continue
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, np.full(value.shape, sequence))
                elif isinstance(value, bool):
                    assert value is bool(int(sequence) % 2)
                else:
                    assert value == sequence
            if not samples or sequence != samples[-1]:
                assert not samples or sequence > samples[-1]
                samples.append(sequence)
    finally:
        session.close()
    assert len(samples) >= 50


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


def test_busy_channel_keeps_only_fresh_displayed_frame(tmp_path, monkeypatch):
    import fcntl
    from unittest.mock import Mock

    from lerobot.teleoperators.xr_controllers import camera_display as module

    writer = FrameWriter(tmp_path / "rgb", 8, 6)
    display = CameraDisplay.__new__(CameraDisplay)
    display.config = VideoConfig(channel=str(writer.path), width=8, height=6, max_age_s=0.5)
    display.status = display.last_key = display.last_captured_ns = None
    display.stats = {"camera_uploads": 0, "placeholder_uploads": 0}
    display._upload = Mock()
    stamp = time.monotonic_ns()
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: stamp)
    try:
        writer.publish(np.zeros((6, 8, 3), dtype=np.uint8), {"captured_monotonic_ns": stamp})
        display.update_camera()
        assert display.status == "live"
        fcntl.flock(writer.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read_frame(writer.path) is None
        with pytest.raises(BlockingIOError):
            read_frame(writer.path, raise_on_busy=True)
        stamp += 100_000_000
        display.update_camera()
        assert display.status == "live"
        assert display.stats["camera_uploads"] == 1
        assert display.stats["placeholder_uploads"] == 0
        stamp += 500_000_000
        display.update_camera()
        assert display.status == "Camera unavailable or stale"
        assert display.stats["placeholder_uploads"] == 1
    finally:
        writer.close()
