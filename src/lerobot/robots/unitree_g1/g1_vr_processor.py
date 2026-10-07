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
from .g1_vr_control import (
    ARM_KEYS,
    G1VRKinematics,
    XRStaleInputError,
    XRTrackingUnavailableError,
    map_input,
    pose,
    stop_buttons,
    validate_xr_timestamp,
)
from .g1_vr_front_box import FrontBoxTracking

if TYPE_CHECKING:
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

CONTROL_KEYS = (
    "base.vx",
    "base.vy",
    "base.yaw",
    "control.enabled",
    "control.following",
    "control.preparing",
    "control.live",
    "control.stop",
    "control.damp",
    "control.created_at",
)


def pickup_guidance(side: str, delta: np.ndarray) -> str:
    """Describe robot-minus-VR position error in the mapped forward/left/up frame."""
    distance = float(np.linalg.norm(delta))
    if distance <= 0.05:
        return f"{side} hand: READY, keep still"
    axes = (("forward", "backward"), ("left", "right"), ("up", "down"))
    directions = [
        f"{axes[i][0 if delta[i] > 0 else 1]} {abs(delta[i]) * 100:.1f} cm"
        for i in np.argsort(-np.abs(delta))
        if abs(delta[i]) >= 0.005
    ]
    return f"{side} hand: move " + ", ".join(directions)


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
    if robot_config.arm_controller == "unitree" and teleop_config.replay_path is not None:
        raise ValueError("Unitree reference startup enables arm motion; replay is not supported")
    return RobotProcessorPipeline(
        steps=[
            G1VRActionProcessor(
                resolve_g1_vr_assets(robot_config.assets),
                arm_test=robot_config.arm_test,
                workspace=robot_config.arm_test_workspace,
                sync_distance_m=robot_config.arm_test_sync_distance_m,
                unitree_reference=robot_config.arm_controller == "unitree",
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


@ProcessorStepRegistry.register("g1_vr_action")
@dataclass
class G1VRActionProcessor(RobotActionProcessorStep):
    assets: str
    max_step: float = 0.04
    arm_test: bool = False
    workspace: str = "small"
    sync_distance_m: float = 0.1
    unitree_reference: bool = False

    def __post_init__(self):
        if not np.isfinite(self.max_step) or not 0 < self.max_step <= 0.04:
            raise ValueError("max_step must be in (0, 0.04]")
        if self.workspace not in ("small", "front_box") or (
            self.workspace == "front_box" and not self.arm_test
        ):
            raise ValueError("front_box requires arm_test=true")
        self.ik = G1VRKinematics(self.assets)
        if self.workspace == "front_box" or self.unitree_reference:
            # Initialize Ipopt before DDS feedback timing and arm authority start.
            # Its first solve can otherwise stall the feedback callback on Orin.
            neutral = np.zeros(14)
            self.ik.solve(self.ik.fk(neutral), neutral, 0.001)
        self.reset()

    def reset(self) -> None:
        self.front_box = FrontBoxTracking(self.ik, sync_distance_m=self.sync_distance_m)
        self.enabled = False
        self.held = None
        self.last_stamp = -np.inf
        self.origin = None
        self.controller_anchor = None
        self.waiting_for_pickup = False
        self.pickup_wrists = None
        self.last_pickup_log = -np.inf
        self.pickup_tracking_missing = False
        self.last_boundary_log = -np.inf
        self.outside_workspace = False
        self.reference_started = False
        self.reference_hold_generation = 0
        self.reference_state: str | None = None
        self.tracking_status: tuple[bool, ...] | None = None
        self.last_status_log = -np.inf
        self.reference_reason = "No positional pickup gate in Unitree reference mode."

    def _reference_status(self, action: dict, state: str, reason: str = "") -> None:
        now = time.monotonic()
        stamp = action.get("captured_at", -np.inf)
        fresh = isinstance(stamp, (int, float)) and 0 <= now - stamp <= 0.25
        tracked = tuple(
            bool(fresh and action.get(f"{side}.tracked", False)) for side in ("head", "left", "right")
        )
        changed = tracked != self.tracking_status or state != self.reference_state
        if changed or now - self.last_status_log >= 10:
            logging.info(
                "Teleop state: %s; head=%s, left=%s, right=%s. %s",
                state,
                *tracked,
                reason,
            )
            self.last_status_log = now
        speech = None
        if state != self.reference_state:
            speech = {
                "waiting": "Waiting. Press R to follow.",
                "lost": "Lost track. Holding until tracking returns.",
                "blocked": "Tracking blocked. Check the terminal.",
                "command hold": "Control delayed. Holding until updates return.",
                "exiting": "Lowering. Exiting.",
            }.get(state)
        if tracked != self.tracking_status:
            changes = []
            for index, side in enumerate(("head", "left", "right")):
                if self.tracking_status is None or tracked[index] != self.tracking_status[index]:
                    label = "Headset" if side == "head" else f"{side.capitalize()} controller"
                    changes.append(f"{label} {'tracked' if tracked[index] else 'tracking unavailable'}.")
            logging.info("XR tracking change: %s", " ".join(changes))
            if speech is None and state != "following":
                speech = " ".join(changes)
        if speech:
            logging.info("Teleop voice: %s", speech, extra={"teleop_voice": speech})
        self.reference_state, self.tracking_status = state, tracked
        self.reference_reason = reason

    def get_config(self) -> dict:
        return {
            "assets": self.assets,
            "max_step": self.max_step,
            "arm_test": self.arm_test,
            "workspace": self.workspace,
            "sync_distance_m": self.sync_distance_m,
            "unitree_reference": self.unitree_reference,
        }

    def _test_target(self, action: dict, measured: np.ndarray) -> np.ndarray:
        """Relative translation only; resuming never moves the session's bounds."""
        controllers = [
            pose(action[f"{side}.grip_pos"], action[f"{side}.grip_quat"]) for side in ("left", "right")
        ]
        if self.controller_anchor is None:
            self.outside_workspace = False
            if self.origin is None:
                self.origin = measured.copy()
                self.origin_wrists = self.ik.fk(measured)
            self.controller_anchor = controllers
            self.robot_anchor = self.ik.fk(measured)
            head = pose(action["head.pos"], action["head.quat"])
            forward = head[:3, 0].copy()
            forward[2] = 0
            forward /= np.linalg.norm(forward)
            self.anchor_yaw = np.column_stack((forward, np.cross([0.0, 0.0, 1.0], forward), [0.0, 0.0, 1.0]))
            # Warm up IK before the returned action can activate the SDK watchdog.
            # Discard the result: the initial target must hold the measured pose.
            self.ik.solve(self.robot_anchor, measured, min(self.max_step, 0.005))
            return measured.copy()
        wrists = [w.copy() for w in self.robot_anchor]
        moving = []
        requested_wrists = []
        held_wrists = self.ik.fk(self.held)
        for i, (current, anchor) in enumerate(zip(controllers, self.controller_anchor, strict=True)):
            delta = 0.2 * self.anchor_yaw.T @ (current[:3, 3] - anchor[:3, 3])
            delta += self.robot_anchor[i][:3, 3] - self.origin_wrists[i][:3, 3]
            requested_wrists.append(self.origin_wrists[i][:3, 3] + delta)
            delta *= min(1.0, 0.02 / max(np.linalg.norm(delta), 1e-12))
            wrists[i][:3, 3] = self.origin_wrists[i][:3, 3] + delta
            moving.append(np.linalg.norm(wrists[i][:3, 3] - held_wrists[i][:3, 3]) > 0.0005)
        if not any(moving):
            self._boundary_guidance(requested_wrists, held_wrists, self.held)
            return self.held.copy()
        target, _ = self.ik.solve(wrists, measured, min(self.max_step, 0.005))
        target = np.clip(target, self.origin - 0.025, self.origin + 0.025)
        for i, active in enumerate(moving):
            if not active:
                target[i * 7 : (i + 1) * 7] = self.held[i * 7 : (i + 1) * 7]

        # Joint clipping can change FK. Find the boundary along the segment
        # from the last accepted target, instead of pulling back toward origin.
        def inside(q: np.ndarray) -> bool:
            return all(
                np.linalg.norm(w[:3, 3] - o[:3, 3]) <= 0.02
                for w, o in zip(self.ik.fk(q), self.origin_wrists, strict=True)
            )

        if not inside(target):
            safe, outside = self.held.copy(), target.copy()
            for _ in range(20):
                midpoint = (safe + outside) * 0.5
                if inside(midpoint):
                    safe = midpoint
                else:
                    outside = midpoint
            target = safe
        self._boundary_guidance(requested_wrists, self.ik.fk(target), target)
        return target

    def _boundary_guidance(
        self, requested: list[np.ndarray], bounded: tuple | list, target: np.ndarray
    ) -> None:
        """Guide hands back into the fixed workspace without disabling motion."""
        differences = [(w[:3, 3] - raw) / 0.2 for raw, w in zip(requested, bounded, strict=True)]
        # Ignore IK convergence residuals: announce workspace/joint saturation,
        # rather than ordinary slow motion toward an attainable target.
        outside = any(
            np.linalg.norm(raw - origin[:3, 3]) > 0.02 + 1e-6
            for raw, origin in zip(requested, self.origin_wrists, strict=True)
        ) or (
            np.max(np.abs(target - self.origin)) >= 0.025 - 1e-6
            and max(np.linalg.norm(d) for d in differences) > 0.05
        )
        if outside:
            if not self.outside_workspace or time.monotonic() - self.last_boundary_log >= 2.0:
                logging.info(
                    "VR workspace boundary: move hands back to pick up the robot arms. "
                    "Targets stop at the bounded workspace; directions use the frame at pickup.\n  %s\n  %s",
                    pickup_guidance("Left", differences[0]),
                    pickup_guidance("Right", differences[1]),
                )
                self.last_boundary_log = time.monotonic()
        elif self.outside_workspace:
            logging.info("VR workspace regained: hands are back inside the bounded workspace.")
        self.outside_workspace = outside

    def action(self, action: dict) -> dict:
        obs = self.transition[TransitionKey.OBSERVATION]
        measured = np.array([obs[k] for k in ARM_KEYS], dtype=float)
        if not np.isfinite(measured).all():
            raise ValueError("Invalid measured arm state")
        if self.unitree_reference:
            return self._reference_action(action, measured, obs)
        if self.workspace == "front_box":
            q, enabled, stop, damp = self.front_box.step(action, measured)
            result = dict(zip(ARM_KEYS, q.tolist(), strict=True))
            result.update(
                dict(
                    zip(
                        CONTROL_KEYS,
                        [
                            0.0,
                            0.0,
                            0.0,
                            float(enabled),
                            float(self.front_box.phase == "synced"),
                            float(self.front_box.phase == "initial"),
                            float(action.get("input.live", False)),
                            float(stop),
                            float(damp),
                            time.monotonic(),
                        ],
                        strict=True,
                    )
                )
            )
            return result
        if self.held is None:
            self.held = measured.copy()
        stop, damp = stop_buttons(action)
        stop = stop or bool(action.get("control.quit", False))
        if action.get("control.start", False):
            if self.arm_test:
                if not self.enabled and not self.waiting_for_pickup:
                    self.waiting_for_pickup = True
                    self.pickup_wrists = self.ik.fk(measured)
                    self.last_pickup_log = -np.inf
                    self.held = measured.copy()
                    logging.info("Start to get picked up: waiting for both VR hands near the robot wrists.")
            else:
                self.enabled = True
        if action.get("control.pause", False) or stop or damp:
            if self.enabled or self.waiting_for_pickup:
                logging.info("VR pickup cancelled: following paused.")
            self.enabled = False
            self.held = measured.copy()
            self.controller_anchor = None
            self.waiting_for_pickup = False
        intent = None
        try:
            intent = map_input(action)
            if action["captured_at"] < self.last_stamp:
                raise ValueError("XR timestamp regressed")
            self.last_stamp = action["captured_at"]
            if self.pickup_tracking_missing:
                self.last_pickup_log = -np.inf
                self.pickup_tracking_missing = False
        except (KeyError, ValueError, TypeError) as exc:
            initial_tracking_wait = (
                self.waiting_for_pickup
                and self.origin is None
                and isinstance(exc, XRTrackingUnavailableError)
            )
            if initial_tracking_wait:
                self.pickup_tracking_missing = True
                if time.monotonic() - self.last_pickup_log >= 2.0:
                    missing = ", ".join(
                        side for side in ("head", "left", "right") if not action.get(f"{side}.tracked", False)
                    )
                    logging.info(
                        "Waiting for tracking: %s. No valid pose, so directional guidance is unavailable. "
                        "Enter VR with Play, wake both controllers and keep them visible to the headset. "
                        "Pickup remains armed; p cancels. No arm authority acquired.",
                        missing,
                    )
                    self.last_pickup_log = time.monotonic()
            elif action.get("control.start", False) or self.enabled or self.waiting_for_pickup:
                logging.warning(
                    "XR tracking paused: %s (head=%s, left=%s, right=%s)",
                    exc,
                    action.get("head.tracked", False),
                    action.get("left.tracked", False),
                    action.get("right.tracked", False),
                )
            self.enabled = False
            self.held = measured.copy()
            self.controller_anchor = None
            self.waiting_for_pickup = initial_tracking_wait
        if self.waiting_for_pickup and intent:
            differences = [
                start[:3, 3] - wrist[:3, 3]
                for wrist, start in zip(intent["wrists"], self.pickup_wrists, strict=True)
            ]
            if max(np.linalg.norm(delta) for delta in differences) <= 0.05:
                self.waiting_for_pickup = False
                self.enabled = True
                logging.info("VR pickup reached: requesting following without a target jump.")
            elif time.monotonic() - self.last_pickup_log >= 2.0:
                logging.info(
                    "Waiting for VR pickup (both hands within 5 cm; p cancels). "
                    "Directions follow your head-facing frame; forward is away from you.\n  %s\n  %s",
                    pickup_guidance("Left", differences[0]),
                    pickup_guidance("Right", differences[1]),
                )
                self.last_pickup_log = time.monotonic()
        velocity = np.zeros(3)
        if self.enabled and intent:
            try:
                if self.arm_test:
                    self.held = self._test_target(action, measured)
                else:
                    self.held, _ = self.ik.solve(intent["wrists"], measured, self.max_step)
                    velocity = intent["velocity"]
            except RuntimeError:
                logging.exception("IK failed: paused at measured pose; explicit restart required")
                self.enabled = False
                self.held = measured.copy()
                self.controller_anchor = None
                self.waiting_for_pickup = False
        result = dict(zip(ARM_KEYS, self.held.tolist(), strict=True))
        result.update(
            dict(
                zip(
                    CONTROL_KEYS,
                    [
                        *velocity.tolist(),
                        float(self.enabled),
                        float(self.enabled),
                        0.0,
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

    def _reference_action(self, action: dict, measured: np.ndarray, observation: dict) -> dict:
        """Reference startup is zero posture; VR mapping starts only on r."""
        stop, damp = stop_buttons(action)
        stop = stop or bool(action.get("control.quit", False))
        if self.held is None:
            self.held = np.zeros(14)
        raw_generation = observation.get("arm.hold_generation", 0)
        raw_holding = observation.get("arm.command_hold", False)
        if (
            not np.isfinite(raw_generation)
            or raw_generation < self.reference_hold_generation
            or int(raw_generation) != raw_generation
            or raw_holding not in (0, 1)
        ):
            raise ValueError("Invalid arm command-hold feedback")
        generation, holding = int(raw_generation), bool(raw_holding)
        if generation != self.reference_hold_generation:
            self.held = measured.copy()
            self.reference_hold_generation = generation
        if action.get("control.start", False):
            self.enabled = True
        if action.get("control.pause", False) or stop or damp:
            self.enabled = False
            self.held = measured.copy()
        following = False
        status = "waiting"
        reason = "No positional pickup gate in Unitree reference mode."
        if self.reference_state in ("lost", "blocked") and not action.get("control.pause", False):
            status, reason = self.reference_state, self.reference_reason
        if holding and self.enabled:
            status, reason = (
                "command hold",
                "Caller deadline missed. Waiting for fresh valid tracking targets.",
            )
        if self.enabled:
            try:
                intent = map_input(action)
                if action["captured_at"] < self.last_stamp:
                    raise ValueError("XR timestamp regressed")
                self.last_stamp = action["captured_at"]
                candidate, _ = self.ik.solve(intent["wrists"], measured, max_step=None)
                # IK can exceed the input's freshness budget. Never revive a
                # stale pose just because the resulting command is newly stamped.
                validate_xr_timestamp(float(action["captured_at"]))
                self.held = candidate
                self.reference_started = following = True
                status = "resuming" if holding else "following"
                reason = (
                    "Valid controller targets accepted; resume requested." if holding else "Robot following."
                )
            except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                tracking_lost = isinstance(exc, (XRTrackingUnavailableError, XRStaleInputError))
                # Keep the operator's follow intent through transient tracking
                # loss. Explicit pause and non-tracking failures still disarm it.
                self.enabled = tracking_lost
                status = "lost" if tracking_lost else "blocked"
                if self.reference_started and self.reference_state != status:
                    self.held = measured.copy()
                recovery = "Resumes when valid tracking returns." if tracking_lost else "Press r to resume."
                reason = f"{exc}. Holding arm target. {recovery}"
                if self.reference_state != status or self.reference_reason != reason:
                    logging.warning("VR following stopped: %s", reason)
        if stop or damp:
            status, reason = "exiting", "Quit/stop requested; returning home and releasing authority."
        self._reference_status(action, status, reason)
        result = dict(zip(ARM_KEYS, self.held.tolist(), strict=True))
        # A fresh solve using the observed hold generation acknowledges the hold.
        # Targets computed before a newer timeout cannot clear that newer hold.
        result["control.resume_generation"] = float(generation) if following else -1.0
        result.update(
            dict(
                zip(
                    CONTROL_KEYS,
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        float(following),
                        float(not self.reference_started),
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
        if self.unitree_reference:
            result[PipelineFeatureType.ACTION]["control.resume_generation"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return result
