# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Generic camera-channel presentation through the installed Isaac Teleop Viz API."""

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from lerobot.cameras.frame_channel import default_channel, read_frame, validate_dimensions


@dataclass(frozen=True)
class VideoConfig:
    channel: str = str(default_channel())
    width: int = 640
    height: int = 480
    max_age_s: float = 0.5
    distance_m: float = 1.5
    screen_width_m: float = 1.8
    expected_source: str | None = None

    def __post_init__(self):
        validate_dimensions(self.width, self.height)
        if not all(np.isfinite(x) and x > 0 for x in (self.max_age_s, self.distance_m, self.screen_width_m)):
            raise ValueError("Video timing and geometry must be positive finite")


class CameraDisplay:
    """Own the graphics session and its frame buffers in the XR worker process."""

    def __init__(self, config, required_extensions=(), offscreen=False):
        import torch
        from isaacteleop import viz

        self.viz, self.torch, self.config = viz, torch, config
        self.offscreen = offscreen
        self.session = None
        self.layer = None
        self.gpu = None
        self.last_key = None
        self.status = None
        self.stats = {
            "frame_loops": 0,
            "render_requested": 0,
            "camera_uploads": 0,
            "placeholder_uploads": 0,
            "camera_age_ms": None,
            "camera_sequence": None,
            "camera_session": None,
        }
        try:
            session_config = viz.VizSessionConfig()
            session_config.mode = viz.DisplayMode.kOffscreen if offscreen else viz.DisplayMode.kXr
            session_config.app_name = "LeRobot camera and controllers"
            session_config.required_extensions = list(required_extensions)
            session_config.window_width, session_config.window_height = config.width, config.height
            session_config.xr_system_wait_seconds = 3
            self.session = viz.VizSession.create(session_config)
            layer_config = viz.QuadLayerConfig()
            layer_config.name = "robot_camera"
            layer_config.resolution = viz.Resolution(config.width, config.height)
            layer_config.format = viz.PixelFormat.kRGBA8
            layer_config.stereo = False
            if not offscreen:
                layer_config.placement = viz.QuadLayerPlacement(
                    viz.Pose3D((0.0, 1.4, -config.distance_m), (1.0, 0.0, 0.0, 0.0)),
                    (config.screen_width_m, config.screen_width_m * config.height / config.width),
                )
            self.layer = self.session.add_quad_layer(layer_config)
            self.gpu = torch.empty((config.height, config.width, 4), dtype=torch.uint8, device="cuda:0")
            self.buffer = viz.VizBuffer()
            self.buffer.data = self.gpu.data_ptr()
            self.buffer.width, self.buffer.height = config.width, config.height
            self.buffer.pitch = config.width * 4
            self.buffer.space = viz.MemorySpace.kDevice
            self.buffer.format = viz.PixelFormat.kRGBA8
        except BaseException:
            self.close()
            raise

    def _upload(self, pixels):
        rgba = np.full((self.config.height, self.config.width, 4), 255, dtype=np.uint8)
        # Letterbox without changing aspect ratio when the producer resolution changes.
        image = Image.fromarray(pixels)
        image.thumbnail((self.config.width, self.config.height), Image.Resampling.BILINEAR)
        rgba[:, :, :3] = 0
        x, y = (self.config.width - image.width) // 2, (self.config.height - image.height) // 2
        rgba[y : y + image.height, x : x + image.width, :3] = np.asarray(image)
        self.gpu.copy_(self.torch.from_numpy(rgba))
        self.layer.submit(self.buffer, stream=self.torch.cuda.current_stream().cuda_stream)

    def update_camera(self):
        try:
            frame = read_frame(self.config.channel, self.config.max_age_s)
        except (ValueError, KeyError, OSError):
            frame = None
        status = "Camera unavailable or stale"
        if frame is not None:
            metadata, pixels = frame
            if (
                self.config.expected_source is not None
                and metadata.get("embodiment") != self.config.expected_source
            ):
                status = "Camera source mismatch"
            else:
                status = "live"
                key = metadata["session_id"], metadata["sequence"]
                self.stats["camera_age_ms"] = (time.monotonic_ns() - metadata["captured_monotonic_ns"]) / 1e6
                if key != self.last_key or self.status != status:
                    self._upload(pixels)
                    self.last_key = key
                    self.stats["camera_uploads"] += 1
                    self.stats["camera_sequence"] = metadata["sequence"]
                    self.stats["camera_session"] = metadata["session_id"]
        if status != "live" and status != self.status:
            image = Image.new("RGB", (self.config.width, self.config.height), (35, 35, 35))
            ImageDraw.Draw(image).text((24, self.config.height // 2), status, fill=(255, 205, 80))
            self._upload(np.asarray(image))
            self.stats["placeholder_uploads"] += 1
            self.stats["camera_age_ms"] = None
        if status != self.status:
            print(f"XR video: {status}", flush=True)
        self.status = status

    def render(self):
        if not self.offscreen:
            pose = self.session.head_pose_now()
            if pose is not None:
                # Pose3D uses wxyz. Place a mono monitor in front of the current head;
                # this moves only the display, not the simulator's robot-mounted camera.
                w, x, y, z = pose.orientation
                forward = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
                position = np.asarray(pose.position) - self.config.distance_m * forward
                self.layer.set_placement(
                    self.viz.QuadLayerPlacement(
                        self.viz.Pose3D(position.tolist(), pose.orientation),
                        (
                            self.config.screen_width_m,
                            self.config.screen_width_m * self.config.height / self.config.width,
                        ),
                    )
                )
        info = self.session.render()
        self.stats["frame_loops"] += 1
        self.stats["render_requested"] += int(info.should_render)
        return info

    def readback(self):
        return np.asarray(self.session.readback_to_host()).copy()

    def close(self):
        if self.session is not None:
            try:
                self.session.destroy()
            finally:
                self.session = None
                self.layer = None
                self.gpu = None
