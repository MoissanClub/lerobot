# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Standard Robot interface for isolated simulation and stock-owned G1 motion.

Separate registration deliberately leaves legacy whole-body UnitreeG1 unchanged.
No XR device code, IK action mapping, or teleoperation loop belongs here.
"""

import json
import logging
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from lerobot.cameras import Camera, CameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.frame_channel import FrameWriter
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .g1_arm_sdk import G1ArmSDK, G1ArmSDKConfig
from .g1_base_motion import G1BaseMotion
from .g1_safety_contract import digest, model_limits, validate_arm_model, validate_contract
from .g1_utils import G1_29_JointIndex
from .g1_vr_assets import resolve_g1_vr_assets
from .g1_vr_control import ARM_KEYS, G1VRKinematics
from .g1_vr_processor import CONTROL_KEYS


@RobotConfig.register_subclass("unitree_g1_motion")
@dataclass(kw_only=True)
class UnitreeG1MotionConfig(RobotConfig):
    assets: str | None = None
    mode: str = "simulation"
    contract: str | None = None
    enable_motion: bool = False
    enable_locomotion: bool = False
    onscreen: bool = False
    video_channel: str | None = None
    report_path: str | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    arm_sdk: G1ArmSDKConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode not in ("camera", "simulation", "shadow", "arms", "walk", "combined"):
            raise ValueError("Unknown G1 motion mode")
        if self.contract and self.arm_sdk is not None:
            raise ValueError("Use robot.arm_sdk or a legacy contract, not both")
        if self.arm_sdk is not None and self.mode not in ("shadow", "arms"):
            raise ValueError("Direct arm_sdk configuration supports shadow and arms modes")
        if self.mode == "camera":
            if len(self.cameras) != 1 or not self.video_channel or self.onscreen:
                raise ValueError(
                    "Camera mode requires exactly one camera and a video_channel, without onscreen"
                )
            camera = next(iter(self.cameras.values()))
            if getattr(camera, "color_mode", ColorMode.RGB) != ColorMode.RGB:
                raise ValueError("XR video requires RGB camera output")
        elif self.cameras:
            raise ValueError("Integrated cameras currently require camera mode")
        physical = self.mode in ("arms", "walk", "combined")
        if self.enable_motion != physical:
            raise ValueError("Only physical motion modes require enable_motion=true")
        if self.enable_locomotion != (self.mode in ("walk", "combined")):
            raise ValueError("Only walk/combined require enable_locomotion=true")
        if self.mode not in ("simulation", "camera") and not self.contract and self.arm_sdk is None:
            raise ValueError("Physical feedback/control requires robot.arm_sdk or a legacy contract")
        if self.mode == "arms" and self.arm_sdk is not None:
            if self.arm_sdk.expected_mode_machine is None:
                raise ValueError("Arm motion requires an explicit arm_sdk.expected_mode_machine")
            for name in ("kp", "kd", "torque_limits"):
                values = getattr(self.arm_sdk, name)
                if len(values) != 14 or not np.isfinite(values).all() or min(values) <= 0:
                    raise ValueError(f"Arm motion requires 14 explicit positive arm_sdk.{name} values")
        if self.mode not in ("simulation", "camera") and (self.onscreen or self.video_channel):
            raise ValueError("Physical video uses a separate camera producer, not Robot rendering")


class UnitreeG1Motion(Robot):
    config_class = UnitreeG1MotionConfig
    name = "unitree_g1_motion"

    def __init__(self, config: UnitreeG1MotionConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._used = False
        self._stack = None
        self.sim = self.arm = self.base = self.writer = self.report = None
        self.damp_requested = False
        self.cameras: dict[str, Camera] = {}
        self._camera_read_started = 0
        self._camera_timed_out = False

    @property
    def action_features(self) -> dict:
        if self.config.mode == "camera":
            return {"control.quit": bool}
        return dict.fromkeys((*ARM_KEYS, *CONTROL_KEYS), float)

    @property
    def observation_features(self) -> dict:
        if self.config.mode == "camera":
            return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}
        return {f"{j.name}.{field}": float for j in G1_29_JointIndex for field in ("q", "dq", "tau")}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _record(self, **row) -> None:
        if self.report:
            self.report.write(json.dumps(row, allow_nan=False) + "\n")
            self.report.flush()

    def connect(self, calibrate: bool = True) -> None:
        if self._used:
            raise RuntimeError("Use a fresh G1 session; reconnect is not supported")
        self._used = True
        self._stack = ExitStack()
        try:
            if self.config.mode != "camera" and self.config.report_path is None:
                self.config.report_path = str(Path(tempfile.mkdtemp(prefix="lerobot-g1-")) / "report.jsonl")
                logging.info("G1 session report: %s", self.config.report_path)
            if self.config.report_path:
                self.report = self._stack.enter_context(Path(self.config.report_path).open("x"))  # noqa: SIM115
            self._record(
                event="starting", mode=self.config.mode, hardware_acceptance="pending_operator_review"
            )
            if self.config.mode == "camera":
                self.cameras = make_cameras_from_configs(self.config.cameras)
                for camera in self.cameras.values():
                    camera.connect()
                    self._stack.callback(camera.disconnect)
                    # Drain the initial buffered frame. Subsequent async reads consume new frames.
                    self._camera_read_started = time.monotonic_ns()
                    camera.async_read(timeout_ms=1000)
                camera_config = next(iter(self.config.cameras.values()))
                self.writer = FrameWriter(
                    self.config.video_channel, camera_config.width, camera_config.height
                )
                self._stack.callback(self.writer.close)
                self._connected = True
                self._record(event="ready", mode="camera", motor_publication=False, base_rpc=False)
                return
            assets = Path(resolve_g1_vr_assets(self.config.assets))
            self.gravity = G1VRKinematics(assets)
            if self.config.mode == "simulation":
                from .g1_vr_simulation import G1VRSimulation

                self.sim = G1VRSimulation(
                    assets, onscreen=self.config.onscreen, video=bool(self.config.video_channel)
                )
                self._stack.callback(self.sim.close)
                q = np.zeros(14)
                for _ in range(100):
                    self.sim.step(q, self.gravity.gravity(q))
                if self.config.video_channel:
                    self.writer = FrameWriter(self.config.video_channel, 640, 480)
                    self._stack.callback(self.writer.close)
            else:
                urdf = assets / "assets/g1_body29_hand14.urdf"
                if self.config.arm_sdk is not None:
                    self.arm_config = self._direct_arm_config(urdf)
                    doc = {}
                else:
                    doc = json.loads(Path(self.config.contract).read_text())
                    self.arm_config = validate_contract(doc, urdf, self.config.enable_motion)
                    if self.config.enable_motion and doc.get("vr_pose_mapping_reviewed") is not True:
                        raise ValueError("Physical VR mapping must be reviewed")
                if self.config.enable_locomotion:
                    for flag in ("locomotion_reviewed", "damping_reviewed", "combined_ownership_reviewed"):
                        if doc.get(flag) is not True:
                            raise ValueError(f"Unreviewed locomotion: {flag}")
                    if set(doc.get("base_limits", {})) != {"max_speed", "max_yaw", "timeout"}:
                        raise ValueError("Reviewed base_limits required")
                self.arm = G1ArmSDK(
                    replace(self.arm_config, read_only=self.config.mode in ("shadow", "walk"))
                )
                self._stack.callback(self.arm.close)
                self.arm.connect()
                logging.info("G1 hardware mode (mode_machine, mode_pr): %s", self.arm.mode)
                if self.config.enable_locomotion:
                    self.base = G1BaseMotion(
                        self._base_feedback, domain_id=self.arm_config.domain_id, **doc["base_limits"]
                    )
                    self._stack.callback(lambda: self.base.close(damp=self.damp_requested))
            self._connected = True
            self._record(event="ready", urdf_sha256=digest(assets / "assets/g1_body29_hand14.urdf"))
        except BaseException:
            self.disconnect()
            raise

    def _direct_arm_config(self, urdf: Path) -> G1ArmSDKConfig:
        cfg = self.config.arm_sdk
        if cfg is None:
            raise ValueError("Missing robot.arm_sdk configuration")
        if self.config.mode == "shadow":
            return replace(cfg, read_only=True)
        limits = model_limits(urdf)
        cfg = replace(
            cfg,
            read_only=False,
            lower=cfg.lower or limits["lower"],
            upper=cfg.upper or limits["upper"],
            gravity_urdf=cfg.gravity_urdf or str(urdf),
        )
        validate_arm_model(cfg, urdf)
        return cfg

    def _base_feedback(self) -> dict:
        obs = self.arm.observation()
        if self.arm.mode != (self.arm_config.expected_mode_machine, 0):
            raise RuntimeError("Base mode does not match reviewed contract")
        return obs

    def get_observation(self) -> dict:
        if not self.is_connected:
            raise RuntimeError("G1 not connected")
        if self.config.mode == "camera":
            name, camera = next(iter(self.cameras.items()))
            started = time.monotonic_ns()
            try:
                pixels = camera.async_read(timeout_ms=200)
            except TimeoutError:
                if not self._camera_timed_out:
                    logging.warning("Camera frames timed out; XR will show an unavailable placeholder")
                self._camera_timed_out = True
                return {}
            # Lower bound on acquisition, not sensor exposure time. Never restamp cached images.
            captured, self._camera_read_started = self._camera_read_started, started
            self._camera_timed_out = False
            published = self.writer.publish(
                pixels, {"captured_monotonic_ns": captured, "embodiment": "g1-29-physical-camera"}
            )
            self._record(event="camera", sequence=self.writer.sequence, published=published)
            return {name: pixels}
        observation = self.sim.observation() if self.sim else self.arm.observation()
        if self.writer:
            captured = time.monotonic_ns()
            self.writer.publish(
                self.sim.render(), {"captured_monotonic_ns": captured, "embodiment": "g1-29-simulation"}
            )
        return observation

    def send_action(self, action: dict) -> dict:
        if not self.is_connected:
            raise RuntimeError("G1 not connected")
        if self.config.mode == "camera":
            # XR poses/buttons are observational in camera mode; there is no motor transport.
            if action.get("control.quit", False):
                raise KeyboardInterrupt
            return {"control.quit": False}
        if set(action) != set(self.action_features) or not np.isfinite(list(action.values())).all():
            raise ValueError("G1 motion requires the complete processed action contract")
        if not 0 <= time.monotonic() - action["control.created_at"] <= 0.25:
            raise ValueError("Stale/future processed command")
        for key in ("control.enabled", "control.live", "control.stop", "control.damp"):
            if action[key] not in (0.0, 1.0):
                raise ValueError(f"Invalid boolean control field: {key}")
        if action["control.stop"] or action["control.damp"]:
            self.damp_requested = bool(action["control.damp"])
            self.disconnect()
            raise KeyboardInterrupt
        enabled = bool(action["control.enabled"])
        if enabled and self.config.enable_motion and not action["control.live"]:
            raise ValueError("Replay cannot activate physical hardware")
        q = np.array([action[k] for k in ARM_KEYS])
        velocity = np.array([action[k] for k in ("base.vx", "base.vy", "base.yaw")])
        if not enabled:
            velocity[:] = 0
        if self.sim:
            self.sim.step(q, self.gravity.gravity(q), velocity)
        else:
            if enabled and self.config.mode in ("arms", "combined") and not self.arm.active:
                observation = self.arm.observation()
                measured = np.array([observation[k] for k in ARM_KEYS])
                if max(abs(q - measured)) > self.arm_config.max_displacement:
                    raise ValueError("Initial action exceeds reviewed displacement")
                self.arm.activate()
            if self.arm.active:
                self.arm.send(dict(zip(ARM_KEYS, q.tolist(), strict=True)))
            if enabled and self.base and self.base.thread is None:
                self.base.activate()
            if self.base and self.base.thread is not None:
                self.base.send(velocity)
        observation = self.sim.observation() if self.sim else self.arm.observation()
        self._record(
            event="action",
            enabled=enabled,
            target=q.tolist(),
            velocity=velocity.tolist(),
            motor_publication=bool(self.arm and self.arm.active),
            base_rpc=bool(self.base and self.base.thread),
            measured=[observation[k] for k in ARM_KEYS],
        )
        return action.copy()

    def disconnect(self) -> None:
        self._connected = False
        stack, self._stack = self._stack, None
        if stack:
            try:
                self._record(event="closing", hardware_acceptance="pending_operator_review")
            finally:
                try:
                    stack.close()
                finally:
                    self.report = None
