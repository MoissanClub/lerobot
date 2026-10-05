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
from .g1_voice import G1FollowingVoice
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
    arm_test: bool = False
    arm_test_workspace: str = "auto"
    arm_test_start_pose: str = "zero"
    arm_test_voice: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.arm_test_workspace == "auto":
            self.arm_test_workspace = "front_box" if self.arm_test else "small"
        if self.arm_test_workspace not in ("small", "front_box"):
            raise ValueError("arm_test_workspace must be small or front_box")
        if self.arm_test_workspace != "small" and not self.arm_test:
            raise ValueError("front_box requires arm_test=true")
        if self.arm_test_start_pose not in ("zero", "current"):
            raise ValueError("arm_test_start_pose must be zero or current")
        if self.arm_test_start_pose != "zero" and not self.arm_test:
            raise ValueError("arm_test_start_pose=current requires arm_test=true")
        if self.arm_test and (self.mode not in ("arms", "shadow") or self.arm_sdk is None or self.contract):
            raise ValueError("arm_test requires direct arm_sdk configuration in arms or shadow mode")
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
        if self.mode == "arms" and self.arm_sdk is not None and not self.arm_test:
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
        self.voice: G1FollowingVoice | None = None

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
            row["recorded_at"] = time.monotonic()
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
                event="starting",
                mode=self.config.mode,
                hardware_acceptance="pending_operator_review",
                arm_joint_names=list(ARM_KEYS),
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
                if self.config.arm_test and self.config.mode == "arms" and self.config.arm_test_voice:
                    self.voice = G1FollowingVoice()
                if self.config.arm_test and self.arm.config.expected_mode_machine is None:
                    # This locks the observed hardware byte; it is not an action-mode detector.
                    self.arm.config.expected_mode_machine = self.arm.mode[0]
                    self.arm_config.expected_mode_machine = self.arm.mode[0]
                logging.info("G1 hardware mode (mode_machine, mode_pr): %s", self.arm.mode)
                if self.config.arm_test:
                    logging.info(
                        "Arm test starting reference: %s. Bounds remain centered on the activation pose.",
                        self.config.arm_test_start_pose,
                    )
                    logging.warning(
                        "Bounded arm test: regular action mode must be selected by the operator. "
                        "Joint travel <= %.3f rad; speed <= %.3f rad/s; wrist radius=%s m; workspace=%s. "
                        "kp=%s kd=%s torque trip thresholds=%s. "
                        "r arms pickup, p holds, q releases to stock.",
                        self.arm_config.max_displacement,
                        self.arm_config.max_velocity,
                        self.arm_config.max_wrist_displacement,
                        self.config.arm_test_workspace,
                        self.arm_config.kp,
                        self.arm_config.kd,
                        self.arm_config.torque_limits,
                    )
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
        if self.config.mode == "shadow" and not self.config.arm_test:
            return replace(cfg, read_only=True)
        limits = model_limits(urdf)
        if self.config.arm_test:
            # Gains from Unitree's g1_arm7_sdk_dds_example.py. Torque values are
            # software trip thresholds, not hardware torque saturation settings.
            cfg = replace(
                cfg,
                kp=cfg.kp or [60.0] * 14,
                kd=cfg.kd or [1.5] * 14,
                torque_limits=cfg.torque_limits or [min(15.0, v) for v in limits["effort"]],
                max_velocity=min(cfg.max_velocity, 0.02),
                max_acceleration=min(cfg.max_acceleration, 0.05),
                max_displacement=min(cfg.max_displacement, 0.03),
                max_wrist_displacement=min(cfg.max_wrist_displacement or 0.025, 0.025),
                max_measured_velocity=min(cfg.max_measured_velocity, 0.1),
                max_tracking_error=min(cfg.max_tracking_error, 0.03),
                blend_s=max(cfg.blend_s, 5.0),
                start_position=[0.0] * 14 if self.config.arm_test_start_pose == "zero" else [],
                start_tolerance=min(cfg.start_tolerance, 0.05),
            )
        if self.config.arm_test and self.config.arm_test_workspace == "front_box":
            cfg = replace(
                cfg,
                max_displacement=float(np.max(np.array(limits["upper"]) - limits["lower"])),
                max_wrist_displacement=None,
                workspace_lower=[0.0, -0.5, -0.5],
                workspace_upper=[1.0, 0.5, 0.5],
            )
        cfg = replace(
            cfg,
            read_only=self.config.mode == "shadow",
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
                try:
                    self.arm.activate()
                except (RuntimeError, ValueError):
                    if self.voice is not None:
                        self.voice.say("Arm activation blocked. Check the terminal for the reason.")
                    raise
            if self.arm.active:
                self.arm.send(dict(zip(ARM_KEYS, q.tolist(), strict=True)))
            if enabled and self.base and self.base.thread is None:
                self.base.activate()
            if self.base and self.base.thread is not None:
                self.base.send(velocity)
        observation = self.sim.observation() if self.sim else self.arm.observation()
        self._record(
            event="action",
            created_at=action["control.created_at"],
            enabled=enabled,
            target=q.tolist(),
            velocity=velocity.tolist(),
            motor_publication=bool(self.arm and self.arm.active),
            base_rpc=bool(self.base and self.base.thread),
            measured=[observation[k] for k in ARM_KEYS],
            observation=observation,
        )
        if self.voice is not None:
            self.voice.update(enabled and bool(self.arm and self.arm.active))
        return action.copy()

    def disconnect(self) -> None:
        self._connected = False
        if self.voice is not None:
            self.voice.update(False)
        stack, self._stack = self._stack, None
        if stack:
            try:
                self._record(event="closing", hardware_acceptance="pending_operator_review")
            finally:
                try:
                    stack.close()
                finally:
                    self.report = None
                    if self.voice is not None:
                        self.voice.close()
                        self.voice = None
