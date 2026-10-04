# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Serializable VR-to-G1 action processor. No transport or command authority."""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.processor.pipeline import ProcessorStepRegistry, RobotActionProcessorStep, RobotProcessorPipeline

from .g1_vr_assets import resolve_g1_vr_assets
from .g1_vr_control import ARM_KEYS, G1VRKinematics, map_input, stop_buttons

if TYPE_CHECKING:
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

CONTROL_KEYS = (
    "base.vx",
    "base.vy",
    "base.yaw",
    "control.enabled",
    "control.live",
    "control.stop",
    "control.damp",
    "control.created_at",
)


def make_g1_vr_action_processor(
    robot_config: "RobotConfig | None", teleop_config: "TeleoperatorConfig | None"
) -> RobotProcessorPipeline:
    """Select the G1 pose processor for the configured Robot/Teleoperator pair."""
    from lerobot.teleoperators.xr_controllers.config_xr_controllers import XRControllersConfig

    from .g1_motion import UnitreeG1MotionConfig

    if not isinstance(robot_config, UnitreeG1MotionConfig):
        raise ValueError("G1 VR requires unitree_g1_motion")
    if not isinstance(teleop_config, XRControllersConfig) or not teleop_config.full_input:
        raise ValueError("G1 VR requires teleop.type=xr_controllers and teleop.full_input=true")
    if not np.allclose(teleop_config.base_T_anchor, np.eye(4)):
        raise ValueError("G1 VR requires untransformed OpenXR poses (identity base_T_anchor)")
    return RobotProcessorPipeline(
        steps=[G1VRActionProcessor(resolve_g1_vr_assets(robot_config.assets))],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


@ProcessorStepRegistry.register("g1_vr_action")
@dataclass
class G1VRActionProcessor(RobotActionProcessorStep):
    assets: str
    max_step: float = 0.04

    def __post_init__(self):
        if not np.isfinite(self.max_step) or not 0 < self.max_step <= 0.04:
            raise ValueError("max_step must be in (0, 0.04]")
        self.ik = G1VRKinematics(self.assets)
        self.reset()

    def reset(self) -> None:
        self.enabled = False
        self.held = None
        self.last_stamp = -np.inf

    def get_config(self) -> dict:
        return {"assets": self.assets, "max_step": self.max_step}

    def action(self, action: dict) -> dict:
        obs = self.transition[TransitionKey.OBSERVATION]
        measured = np.array([obs[k] for k in ARM_KEYS], dtype=float)
        if not np.isfinite(measured).all():
            raise ValueError("Invalid measured arm state")
        if self.held is None:
            self.held = measured.copy()
        stop, damp = stop_buttons(action)
        stop = stop or bool(action.get("control.quit", False))
        if action.get("control.start", False):
            self.enabled = True
        if action.get("control.pause", False) or stop or damp:
            self.enabled = False
            self.held = measured.copy()
        intent = None
        try:
            intent = map_input(action)
            if action["captured_at"] < self.last_stamp:
                raise ValueError("XR timestamp regressed")
            self.last_stamp = action["captured_at"]
        except (KeyError, ValueError, TypeError) as exc:
            if action.get("control.start", False) or self.enabled:
                logging.warning(
                    "XR tracking paused: %s (head=%s, left=%s, right=%s)",
                    exc,
                    action.get("head.tracked", False),
                    action.get("left.tracked", False),
                    action.get("right.tracked", False),
                )
            self.enabled = False
            self.held = measured.copy()
        velocity = np.zeros(3)
        if self.enabled and intent:
            try:
                self.held, _ = self.ik.solve(intent["wrists"], measured, self.max_step)
                velocity = intent["velocity"]
            except RuntimeError:
                logging.exception("IK failed: paused at measured pose; explicit restart required")
                self.enabled = False
                self.held = measured.copy()
        result = dict(zip(ARM_KEYS, self.held.tolist(), strict=True))
        result.update(
            dict(
                zip(
                    CONTROL_KEYS,
                    [
                        *velocity.tolist(),
                        float(self.enabled),
                        float(action.get("input.live", False)),
                        float(stop or damp),
                        float(damp),
                        time.monotonic(),
                    ],
                    strict=True,
                )
            )
        )
        return result

    def transform_features(self, features: dict) -> dict:
        result = features.copy()
        result[PipelineFeatureType.ACTION] = {
            k: PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for k in (*ARM_KEYS, *CONTROL_KEYS)
        }
        return result
