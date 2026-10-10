#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Camera delivery validation. No motor command path."""

import argparse
import hashlib
import json
import signal
import socket
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    camera = sub.add_parser("camera", help="RGB camera producer only; no DDS or XR")
    camera.add_argument("--device", required=True, help="OpenCV camera index or /dev/video path")
    camera.add_argument("--width", type=int, default=640)
    camera.add_argument("--height", type=int, default=480)
    camera.add_argument("--fps", type=int, default=30)
    display = sub.add_parser("display", help="Headset video only; no robot or controller input")
    display.add_argument("--offscreen", action="store_true", help="GPU test only, not headset delivery")
    for command in (display,):
        command.add_argument("--cloudxr-config", type=Path, help="Optionally start a private CloudXR runtime")
        command.add_argument("--accept-cloudxr-eula", action="store_true")
    for command in (camera, display):
        command.add_argument("--channel", type=Path, default=Path("/tmp/lerobot-physical-camera.rgb"))
        command.add_argument("--source-id", default="g1-29-physical-camera")
        command.add_argument("--duration-s", type=float, default=30)
        command.add_argument("--output", type=Path, required=True, help="New JSONL report, never overwritten")
    return p


def record(output, **data):
    output.write(json.dumps(data, allow_nan=False) + "\n")
    output.flush()


def camera(args, output):
    from lerobot.cameras.frame_channel import FrameWriter
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    device = int(args.device) if args.device.isdecimal() else Path(args.device)
    config = OpenCVCameraConfig(
        index_or_path=device, width=args.width, height=args.height, fps=args.fps, buffer_size=1
    )
    with ExitStack() as cleanup:
        source = OpenCVCamera(config)
        cleanup.callback(lambda: source.disconnect() if source.is_connected else None)
        source.connect()
        buffer_size_honored = source.buffer_size_set is True and source.buffer_size_actual == 1
        record(
            output,
            event="camera_config",
            buffer_size_requested=config.buffer_size,
            buffer_size_set=source.buffer_size_set,
            buffer_size_actual=source.buffer_size_actual,
            buffer_size_honored=buffer_size_honored,
        )
        if not buffer_size_honored:
            print(
                "Warning: capture backend did not confirm buffer_size=1; frame freshness is unverified",
                flush=True,
            )
        writer = FrameWriter(args.channel, args.width, args.height)
        cleanup.callback(writer.close)
        print("Camera capture only: no robot connection or motor publisher", flush=True)
        deadline = time.monotonic() + args.duration_s
        frame_period_ns = 1e9 / args.fps
        stall_threshold_ns = max(4 * frame_period_ns, 250e6 if buffer_size_honored else 0)
        minimum_live_wait_ns = frame_period_ns / 2
        previous_delivery_ns = None
        draining_backlog = False
        drain_trigger = None
        frames_read = frames_published = backlog_frames_dropped = publish_failures = 0
        try:
            while time.monotonic() < deadline:
                read_started = time.monotonic_ns()
                had_previous_delivery = previous_delivery_ns is not None
                loop_gap_ns = 0 if previous_delivery_ns is None else read_started - previous_delivery_ns
                pixels = source.read()
                captured = time.monotonic_ns()
                frames_read += 1
                read_wait_ns = captured - read_started
                previous_delivery_ns = captured
                if read_wait_ns > stall_threshold_ns:
                    draining_backlog = True
                    drain_trigger = "slow_read"
                elif had_previous_delivery and read_wait_ns < minimum_live_wait_ns:
                    draining_backlog = True
                    drain_trigger = drain_trigger or "fast_read"
                if draining_backlog and not minimum_live_wait_ns <= read_wait_ns <= stall_threshold_ns:
                    backlog_frames_dropped += 1
                    record(
                        output,
                        event="camera",
                        sequence=writer.sequence,
                        published=False,
                        dropped_reason="draining_backlog",
                        drain_trigger=drain_trigger,
                        read_wait_ms=read_wait_ns / 1e6,
                        loop_gap_ms=loop_gap_ns / 1e6,
                    )
                    continue
                draining_backlog = False
                drain_trigger = None
                # This is OpenCV delivery time, not hardware exposure time. Consequently,
                # camera_age_ms is a lower bound on the frame's true age.
                published = writer.publish(
                    pixels,
                    {
                        "captured_monotonic_ns": captured,
                        "timestamp_semantics": "opencv_delivery",
                        "embodiment": args.source_id,
                    },
                )
                record(
                    output,
                    event="camera",
                    sequence=writer.sequence,
                    published=published,
                    read_wait_ms=read_wait_ns / 1e6,
                    loop_gap_ms=loop_gap_ns / 1e6,
                )
                frames_published += int(published)
                publish_failures += int(not published)
        finally:
            record(
                output,
                event="camera_summary",
                frames_read=frames_read,
                frames_published=frames_published,
                backlog_frames_dropped=backlog_frames_dropped,
                publish_failures=publish_failures,
                draining_at_exit=draining_backlog,
            )
        if not frames_published:
            raise RuntimeError("No camera frames published")
        if draining_backlog:
            raise RuntimeError("Camera ended while draining a capture backlog")


def display(args, output):
    from lerobot.teleoperators.xr_controllers.camera_display import CameraDisplay, VideoConfig

    viewer = CameraDisplay(
        VideoConfig(channel=str(args.channel), expected_source=args.source_id), offscreen=args.offscreen
    )
    try:
        deadline = time.monotonic() + args.duration_s
        while time.monotonic() < deadline:
            if viewer.session.should_close():
                raise RuntimeError("XR session ended; restart display to reconnect")
            viewer.update_camera()
            viewer.render()
            record(output, event="display", status=viewer.status, **viewer.stats)
            time.sleep(1 / 90)
        if not viewer.stats["camera_uploads"]:
            raise RuntimeError("No fresh camera frame displayed")
    finally:
        viewer.close()


def main():
    p = parser()
    args = p.parse_args()
    if not np.isfinite(args.duration_s) or not 0 < args.duration_s <= 3600:
        p.error("duration-s must be in (0, 3600]")
    if args.mode == "camera" and (args.fps <= 0 or args.width <= 0 or args.height <= 0):
        p.error("Camera dimensions and fps must be positive")
    cloudxr_config = getattr(args, "cloudxr_config", None)
    if cloudxr_config and getattr(args, "offscreen", False):
        p.error("CloudXR requires live headset mode")
    if cloudxr_config and not args.accept_cloudxr_eula:
        p.error("Review the CloudXR EULA, then explicitly pass --accept-cloudxr-eula")
    import lerobot

    if Path(lerobot.__file__).resolve().parents[2] != Path(__file__).resolve().parents[2]:
        p.error("Imported LeRobot does not match this checkout; set PYTHONPATH=$PWD/src")

    def terminate(*_):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        with args.output.open("x") as output:
            record(
                output,
                event="start",
                mode=args.mode,
                parameters={
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                source=str(Path(__file__).resolve()),
                diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                motor_commands_enabled=False,
                headset_acceptance="pending_operator_review",
            )
            try:
                with ExitStack() as cleanup:
                    if cloudxr_config:
                        from isaacteleop.cloudxr import CloudXRLauncher

                        for port in (48322, 49100):
                            with socket.socket() as probe:
                                probe.settimeout(0.3)
                                if probe.connect_ex(("127.0.0.1", port)) == 0:
                                    raise RuntimeError(
                                        f"Port {port} in use; do not replace a running CloudXR"
                                    )
                        cleanup.enter_context(
                            CloudXRLauncher(env_config=str(cloudxr_config), accept_eula=True)
                        )
                        print(
                            "CloudXR ready. Connect headset using this host's IP at "
                            "https://nvidia.github.io/IsaacTeleop/client",
                            flush=True,
                        )
                    globals()[args.mode](args, output)
            except BaseException as exc:
                record(output, event="failed_or_cancelled", error=str(exc) or type(exc).__name__)
                raise
            else:
                record(output, event="completed", motor_commands_enabled=False)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
