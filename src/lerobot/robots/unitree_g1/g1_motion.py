# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Standard Robot interface for isolated simulation and stock-owned G1 motion.

Separate registration deliberately leaves legacy whole-body UnitreeG1 unchanged.
No XR device code, IK action mapping, or teleoperation loop belongs here.
"""

import json
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .g1_arm_sdk import G1ArmSDK
from .g1_base_motion import G1BaseMotion
from .g1_safety_contract import digest, validate_contract
from .g1_utils import G1_29_JointIndex
from .g1_vr_control import ARM_KEYS, G1VRKinematics
from .g1_vr_processor import CONTROL_KEYS


@RobotConfig.register_subclass("unitree_g1_motion")
@dataclass(kw_only=True)
class UnitreeG1MotionConfig(RobotConfig):
    assets: str
    mode: str = "simulation"
    contract: str | None = None
    enable_motion: bool = False
    enable_locomotion: bool = False
    onscreen: bool = False
    video_channel: str | None = None
    report_path: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in ("simulation", "shadow", "arms", "walk", "combined"):
            raise ValueError("Unknown G1 motion mode")
        physical = self.mode in ("arms", "walk", "combined")
        if self.enable_motion != physical:
            raise ValueError("Only physical motion modes require enable_motion=true")
        if self.enable_locomotion != (self.mode in ("walk", "combined")):
            raise ValueError("Only walk/combined require enable_locomotion=true")
        if self.mode != "simulation" and (not self.contract or not self.report_path):
            raise ValueError("Physical feedback/control requires contract and report_path")
        if self.mode != "simulation" and (self.onscreen or self.video_channel):
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

    @property
    def action_features(self) -> dict:
        return dict.fromkeys((*ARM_KEYS, *CONTROL_KEYS), float)

    @property
    def observation_features(self) -> dict:
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
        assets = Path(self.config.assets)
        try:
            if self.config.report_path:
                self.report = self._stack.enter_context(Path(self.config.report_path).open("x"))  # noqa: SIM115
            self._record(
                event="starting", mode=self.config.mode, hardware_acceptance="pending_operator_review"
            )
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
                    from lerobot.cameras.frame_channel import FrameWriter

                    self.writer = FrameWriter(self.config.video_channel, 640, 480)
                    self._stack.callback(self.writer.close)
            else:
                doc = json.loads(Path(self.config.contract).read_text())
                self.arm_config = validate_contract(
                    doc, assets / "assets/g1_body29_hand14.urdf", self.config.enable_motion
                )
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

    def _base_feedback(self) -> dict:
        obs = self.arm.observation()
        if self.arm.mode != (self.arm_config.expected_mode_machine, 0):
            raise RuntimeError("Base mode does not match reviewed contract")
        return obs

    def get_observation(self) -> dict:
        if not self.is_connected:
            raise RuntimeError("G1 not connected")
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
