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

        def __init__(self, config):
            assert config.index_or_path == 0

        def connect(self):
            self.is_connected = True

        def read(self):
            return np.full((4, 8, 3), 127, dtype=np.uint8)

        def disconnect(self):
            self.is_connected = False

    monkeypatch.setattr(camera_opencv, "OpenCVCamera", Camera)
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
            assert metadata["captured_monotonic_ns"] <= time.monotonic_ns()
            assert np.all(pixels == 127)
        original(report, **data)

    monkeypatch.setattr(diagnostic, "record", inspect)
    diagnostic.camera(args, output)
    assert not args.channel.exists()
    assert json.loads(output.getvalue().splitlines()[0])["published"]


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
