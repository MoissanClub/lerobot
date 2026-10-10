# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Read-only XR boundary tests; never connect to physical DDS or camera hardware."""

import io
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from examples.unitree_g1 import validate_xr_readonly as diagnostic


class SequenceClock:
    def __init__(self, *values):
        self.values = iter(values)
        self.last = values[-1]

    def __call__(self):
        self.last = next(self.values, self.last)
        return self.last


def test_no_motion_cli():
    with pytest.raises(SystemExit):
        diagnostic.parser().parse_args(["display", "--output", "out.jsonl", "--enable-motion"])


def test_camera_loss_placeholder_and_recovery_without_gpu(tmp_path):
    from lerobot.cameras.frame_channel import FrameWriter
    from lerobot.teleoperators.xr_controllers.camera_display import CameraDisplay, VideoConfig

    viewer = CameraDisplay.__new__(CameraDisplay)
    viewer.config = VideoConfig(channel=str(tmp_path / "rgb"), expected_source="camera")
    viewer.status = viewer.last_key = None
    viewer.stats = {"camera_uploads": 0, "placeholder_uploads": 0}
    uploads = []
    viewer._upload = lambda pixels: uploads.append(pixels.copy())
    for source, expected in (("camera", "live"), ("wrong", "Camera source mismatch"), ("camera", "live")):
        writer = FrameWriter(viewer.config.channel, 8, 4)
        try:
            writer.publish(
                np.full((4, 8, 3), 123, dtype=np.uint8),
                {"captured_monotonic_ns": time.monotonic_ns(), "embodiment": source},
            )
            viewer.update_camera()
            assert viewer.status == expected
        finally:
            writer.close()
        viewer.update_camera()
        assert viewer.status == "Camera unavailable or stale"
    assert viewer.stats["camera_uploads"] == 2
    assert viewer.stats["placeholder_uploads"] >= 3


def test_camera_producer_fresh_frames_and_cleanup(tmp_path, monkeypatch):
    from lerobot.cameras.frame_channel import read_frame
    from lerobot.cameras.opencv import camera_opencv

    class Camera:
        is_connected = False
        read_completed_ns = None
        buffer_size_set = True
        buffer_size_actual = 1

        def __init__(self, config):
            assert config.index_or_path == 0
            assert config.buffer_size == 1

        def connect(self):
            self.is_connected = True

        def read(self):
            Camera.read_completed_ns = time.monotonic_ns()
            return np.full((4, 8, 3), 127, dtype=np.uint8)

        def disconnect(self):
            self.is_connected = False

    monkeypatch.setattr(camera_opencv, "OpenCVCamera", Camera)

    monkeypatch.setattr(diagnostic.time, "monotonic", SequenceClock(0.0, 0.05, 0.2))
    monkeypatch.setattr(
        diagnostic.time,
        "sleep",
        lambda *_args, **_kwargs: pytest.fail("camera publisher must not add a pacing sleep"),
    )
    args = SimpleNamespace(
        device="0",
        width=8,
        height=4,
        fps=30,
        channel=tmp_path / "rgb",
        source_id="test-camera",
        duration_s=0.1,
    )
    output = io.StringIO()
    original = diagnostic.record

    def inspect(report, **data):
        if data.get("published"):
            metadata, pixels = read_frame(args.channel)
            assert metadata["embodiment"] == "test-camera"
            assert metadata["timestamp_semantics"] == "opencv_delivery"
            assert Camera.read_completed_ns is not None
            assert metadata["captured_monotonic_ns"] >= Camera.read_completed_ns
            assert metadata["captured_monotonic_ns"] <= time.monotonic_ns()
            assert np.all(pixels == 127)
        original(report, **data)

    monkeypatch.setattr(diagnostic, "record", inspect)
    diagnostic.camera(args, output)
    assert not args.channel.exists()
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert rows[0] == {
        "event": "camera_config",
        "buffer_size_requested": 1,
        "buffer_size_set": True,
        "buffer_size_actual": 1,
        "buffer_size_honored": True,
    }
    assert len(rows) == 3
    assert rows[1]["published"]
    assert rows[1]["read_wait_ms"] >= 0
    assert rows[2] == {
        "event": "camera_summary",
        "frames_read": 1,
        "frames_published": 1,
        "backlog_frames_dropped": 0,
        "publish_failures": 0,
        "draining_at_exit": False,
    }


def test_camera_producer_drops_frame_after_slow_read(tmp_path, monkeypatch):
    from lerobot.cameras.opencv import camera_opencv

    class Camera:
        is_connected = False
        buffer_size_set = True
        buffer_size_actual = 1

        def __init__(self, _config):
            pass

        def connect(self):
            self.is_connected = True

        def read(self):
            return np.full((4, 8, 3), 127, dtype=np.uint8)

        def disconnect(self):
            self.is_connected = False

    monkeypatch.setattr(camera_opencv, "OpenCVCamera", Camera)
    monkeypatch.setattr(diagnostic.time, "monotonic", SequenceClock(0.0, 0.05, 0.2))
    monkeypatch.setattr(diagnostic.time, "monotonic_ns", SequenceClock(0, 300_000_000))
    args = SimpleNamespace(
        device="0",
        width=8,
        height=4,
        fps=30,
        channel=tmp_path / "rgb",
        source_id="test-camera",
        duration_s=0.1,
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="No camera frames published"):
        diagnostic.camera(args, output)

    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(rows) == 3
    assert rows[1]["published"] is False
    assert rows[1]["dropped_reason"] == "draining_backlog"
    assert rows[1]["drain_trigger"] == "slow_read"
    assert rows[1]["read_wait_ms"] == 300
    assert rows[2]["frames_published"] == 0
    assert rows[2]["backlog_frames_dropped"] == 1
    assert rows[2]["draining_at_exit"] is True
    assert not args.channel.exists()


def test_camera_producer_detects_fast_read_after_stall(tmp_path, monkeypatch):
    from lerobot.cameras import frame_channel
    from lerobot.cameras.opencv import camera_opencv

    class Camera:
        is_connected = False
        buffer_size_set = True
        buffer_size_actual = 1

        def __init__(self, _config):
            pass

        def connect(self):
            self.is_connected = True

        def read(self):
            return np.full((4, 8, 3), 127, dtype=np.uint8)

        def disconnect(self):
            self.is_connected = False

    class Writer:
        sequence = 0

        def __init__(self, *_args):
            pass

        def publish(self, _pixels, _metadata):
            self.sequence += 1
            return True

        def close(self):
            pass

    monkeypatch.setattr(camera_opencv, "OpenCVCamera", Camera)
    monkeypatch.setattr(frame_channel, "FrameWriter", Writer)
    monkeypatch.setattr(diagnostic.time, "monotonic", SequenceClock(0.0, 0.01, 0.02, 0.2))
    monkeypatch.setattr(
        diagnostic.time,
        "monotonic_ns",
        SequenceClock(0, 33_000_000, 500_000_000, 501_000_000),
    )
    args = SimpleNamespace(
        device="0",
        width=8,
        height=4,
        fps=30,
        channel=tmp_path / "rgb",
        source_id="test-camera",
        duration_s=0.1,
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="ended while draining"):
        diagnostic.camera(args, output)

    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert rows[1]["published"] is True
    assert rows[2]["published"] is False
    assert rows[2]["dropped_reason"] == "draining_backlog"
    assert rows[2]["drain_trigger"] == "fast_read"
    assert rows[2]["loop_gap_ms"] == 467
    assert rows[3]["frames_published"] == 1
    assert rows[3]["backlog_frames_dropped"] == 1
    assert rows[3]["draining_at_exit"] is True


def test_camera_producer_drains_backlog_before_resuming(tmp_path, monkeypatch):
    from lerobot.cameras import frame_channel
    from lerobot.cameras.opencv import camera_opencv

    class Camera:
        is_connected = False
        buffer_size_set = True
        buffer_size_actual = 1

        def __init__(self, _config):
            pass

        def connect(self):
            self.is_connected = True

        def read(self):
            return np.full((4, 8, 3), 127, dtype=np.uint8)

        def disconnect(self):
            self.is_connected = False

    class Writer:
        sequence = 0
        published_metadata = []

        def __init__(self, *_args):
            type(self).published_metadata = []

        def publish(self, _pixels, metadata):
            self.sequence += 1
            type(self).published_metadata.append(metadata)
            return True

        def close(self):
            pass

    monkeypatch.setattr(camera_opencv, "OpenCVCamera", Camera)
    monkeypatch.setattr(frame_channel, "FrameWriter", Writer)
    monkeypatch.setattr(
        diagnostic.time,
        "monotonic",
        SequenceClock(0.0, 0.01, 0.02, 0.03, 0.04, 0.2),
    )
    monkeypatch.setattr(
        diagnostic.time,
        "monotonic_ns",
        SequenceClock(
            0,
            300_000_000,
            301_000_000,
            302_000_000,
            303_000_000,
            304_000_000,
            305_000_000,
            335_000_000,
        ),
    )
    args = SimpleNamespace(
        device="0",
        width=8,
        height=4,
        fps=30,
        channel=tmp_path / "rgb",
        source_id="test-camera",
        duration_s=0.1,
    )
    output = io.StringIO()

    diagnostic.camera(args, output)

    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    camera_rows = [row for row in rows if row["event"] == "camera"]
    assert [row["published"] for row in camera_rows] == [False, False, False, True]
    assert [row["read_wait_ms"] for row in camera_rows] == [300, 1, 1, 30]
    assert Writer.published_metadata[0]["captured_monotonic_ns"] == 335_000_000
    assert rows[-1]["backlog_frames_dropped"] == 3
    assert rows[-1]["frames_published"] == 1
    assert rows[-1]["draining_at_exit"] is False


def test_camera_producer_preserves_failure_and_writes_summary(tmp_path, monkeypatch):
    from lerobot.cameras.opencv import camera_opencv

    class Camera:
        is_connected = False
        buffer_size_set = True
        buffer_size_actual = 1

        def __init__(self, _config):
            pass

        def connect(self):
            self.is_connected = True

        def read(self):
            raise TimeoutError("camera stalled")

        def disconnect(self):
            self.is_connected = False

    monkeypatch.setattr(camera_opencv, "OpenCVCamera", Camera)
    monkeypatch.setattr(diagnostic.time, "monotonic", SequenceClock(0.0, 0.01))
    args = SimpleNamespace(
        device="0",
        width=8,
        height=4,
        fps=30,
        channel=tmp_path / "rgb",
        source_id="test-camera",
        duration_s=0.1,
    )
    output = io.StringIO()

    with pytest.raises(TimeoutError, match="camera stalled"):
        diagnostic.camera(args, output)

    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert rows[-1] == {
        "event": "camera_summary",
        "frames_read": 0,
        "frames_published": 0,
        "backlog_frames_dropped": 0,
        "publish_failures": 0,
        "draining_at_exit": False,
    }
    assert not args.channel.exists()


def test_cli_failure_report_and_no_overwrite(tmp_path):
    output = tmp_path / "report.jsonl"
    cmd = [
        sys.executable,
        "examples/unitree_g1/validate_xr_readonly.py",
        "camera",
        "--device",
        str(tmp_path / "missing.avi"),
        "--output",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["motor_commands_enabled"] is False
    assert rows[-1]["event"] == "failed_or_cancelled"
    contents = output.read_bytes()
    assert subprocess.run(cmd, capture_output=True, timeout=20).returncode != 0
    assert output.read_bytes() == contents


def test_display_only_does_not_create_controller_or_robot(tmp_path, monkeypatch):
    from lerobot.robots.unitree_g1.g1_arm_sdk import G1ArmSDK
    from lerobot.teleoperators.xr_controllers import XRControllers, camera_display

    def forbidden(*args, **kwargs):
        raise AssertionError("Video-only path initialized robot/controller input")

    monkeypatch.setattr(G1ArmSDK, "__init__", forbidden)
    monkeypatch.setattr(XRControllers, "__init__", forbidden)
    closed = []

    class Display:
        def __init__(self, config, offscreen):
            assert config.expected_source == "physical-test"
            self.session = SimpleNamespace(should_close=lambda: False)
            self.stats = {"camera_uploads": 0}
            self.status = "live"

        def update_camera(self):
            self.stats["camera_uploads"] += 1

        def render(self):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(camera_display, "CameraDisplay", Display)
    args = SimpleNamespace(
        channel=tmp_path / "rgb", source_id="physical-test", offscreen=True, duration_s=0.03
    )
    diagnostic.display(args, io.StringIO())
    assert closed == [True]
