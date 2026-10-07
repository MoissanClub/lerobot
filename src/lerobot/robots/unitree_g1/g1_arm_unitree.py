# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Unitree G1_29_ArmController command behavior behind the LeRobot SDK adapter.

Reference: unitreerobotics/xr_teleoperate/teleop/robot_control/robot_arm.py.
Keeps transport freshness, exclusive ownership and cleanup from G1ArmSDK, but
does not inherit the experimental workspace, speed, posture or torque trips.
Only arm slots are owned; leg/waist commands remain inactive in motion mode.
"""

import logging
import time
from typing import NamedTuple

import numpy as np

from .g1_arm_sdk import ARM_KEYS, G1ArmSDK, G1ArmSDKConfig, arm_command


class _MotorFeedback(NamedTuple):
    q: float
    dq: float
    tau_est: float


class _IMUFeedback(NamedTuple):
    rpy: tuple[float, ...]


class _Feedback(NamedTuple):
    motor_state: tuple[_MotorFeedback, ...]
    imu_state: _IMUFeedback
    mode_machine: int
    mode_pr: int
    tick: int


class UnitreeArmSDK(G1ArmSDK):
    """250 Hz measured-position clipping, explicit feedforward, full authority."""

    def __init__(self, config: G1ArmSDKConfig, **kwargs):
        super().__init__(config, **kwargs)
        if config.command_fault_timeout_s <= config.command_timeout_s:
            raise ValueError("command_fault_timeout_s must exceed command_timeout_s")
        self.command_hold = False
        self.hold_generation = 0
        self.returning_home = False
        self.release_weight = 1.0
        self.command_count = 0
        self.first_publish_at: float | None = None
        self.last_publish_at: float | None = None
        self.max_publish_gap = 0.0

    def _check_pose(self, q: np.ndarray) -> None:
        if np.shape(q) != (14,) or not np.isfinite(q).all():
            raise ValueError("Arm targets require 14 finite joint positions")

    def _copy_state(self, state) -> _Feedback:
        # Snapshot only consumed scalar fields, once per DDS callback. Immutable
        # snapshots can be shared by readers without deep-copying the full IDL.
        if isinstance(state, _Feedback):
            return state
        return _Feedback(
            tuple(_MotorFeedback(float(m.q), float(m.dq), float(m.tau_est)) for m in state.motor_state[:29]),
            _IMUFeedback(tuple(float(v) for v in state.imu_state.rpy)),
            int(state.mode_machine),
            int(state.mode_pr),
            int(state.tick),
        )

    def _check_state(self, state) -> np.ndarray:
        q = self._q(state)
        self._check_pose(q)
        return q

    def _check_wrist_displacement(self, q: np.ndarray) -> None:
        # Cartesian restrictions belong to an optional action processor.
        pass

    def _activation_target(self, measured: np.ndarray) -> np.ndarray:
        # Called before starting the publisher thread, after DDS discovery.
        self.use_feedforward = False
        return np.zeros(14)

    def send(
        self, action: dict, *, use_feedforward: bool = True, resume_generation: int | None = None
    ) -> dict:
        with self.lock:
            if not self.active or self.fault:
                raise RuntimeError(self.fault or "Arm authority not activated")
            if set(action) != set(ARM_KEYS):
                raise ValueError("Each command must contain exactly the 14 arm targets")
            if resume_generation is not None and (
                type(resume_generation) is not int or resume_generation < 0
            ):
                raise ValueError("Resume generation must be a nonnegative integer")
            q = np.array([action[key] for key in ARM_KEYS], dtype=float)
            self._check_pose(q)
            state = self._snapshot()
            # Check here too: a late caller must not hide a missed deadline if
            # the publisher itself was delayed by scheduling.
            self._check_caller(self.clock(), self._q(state))
            if self.command_hold and resume_generation == self.hold_generation:
                self.command_hold = False
                logging.info("Teleop state: resumed; fresh valid target acknowledged command hold")
            if not self.command_hold:
                self.target = q
                self.use_feedforward = use_feedforward
            self.last_command = self.clock()
            return dict(zip(ARM_KEYS, self.target.tolist(), strict=True))

    def _check_caller(self, now: float, measured: np.ndarray) -> None:
        if self.returning_home:
            return
        age = now - self.last_command
        if age > self.config.command_fault_timeout_s:
            self.fault = f"Caller command timeout: {age:.3f}s without updates"
            raise RuntimeError(self.fault)
        if age > self.config.command_timeout_s and not self.command_hold:
            self.command_hold = True
            self.hold_generation += 1
            self.target = measured.copy()
            self.use_feedforward = True
            logging.warning(
                "Teleop state: command hold; caller gap %.1f ms. Holding measured pose; "
                "authority retained. Waiting for fresh valid targets. Generation %d.",
                age * 1000,
                self.hold_generation,
                extra={"teleop_voice": "Control delayed. Holding until updates return."},
            )

    def observation(self) -> dict:
        with self.lock:
            result = super().observation()
            result["arm.hold_generation"] = float(self.hold_generation)
            result["arm.command_hold"] = float(self.command_hold)
            return result

    def _step(self) -> None:
        with self.lock:
            now = self.clock()
            state = self._snapshot()
            measured = self._q(state)
            self._check_caller(now, measured)
            delta = self.target - measured
            scale = max(float(np.max(np.abs(delta))) / (30.0 / 250.0), 1.0)
            command = measured + delta / scale
            tau = (
                self.gravity.gravity(self.target)
                if self.use_feedforward and self.gravity is not None
                else np.zeros(14)
            )
            if not np.isfinite(tau).all():
                raise RuntimeError("Nonfinite feedforward torque")
            self.weight = self.release_weight
            self.transport.write(arm_command(command, tau, self.weight, state.mode_machine, self.config))
            published = self.clock()
            if self.last_publish_at is not None:
                self.max_publish_gap = max(self.max_publish_gap, published - self.last_publish_at)
            else:
                self.first_publish_at = published
            self.last_publish_at = published
            self.command_count += 1
            self.previous = command
            self.last_step = now

    def close(self) -> None:
        """Normal exit homes first, then fades authority over two seconds."""
        try:
            if self.active and not self.fault and not self.returning_home:
                with self.lock:
                    self.returning_home = True
                    self.target = np.zeros(14)
                deadline = self.clock() + 5.0
                while self.active and not self.fault:
                    with self.lock:
                        measured = self._q(self._snapshot())
                    if np.all(np.abs(measured) < 0.05):
                        break
                    if self.clock() >= deadline:
                        raise RuntimeError("Unitree home return timed out before authority release")
                    time.sleep(0.05)
                if self.active and not self.fault:
                    for weight in np.linspace(1.0, 0.0, 101):
                        with self.lock:
                            self.release_weight = float(weight)
                        if self.fault:
                            break
                        time.sleep(0.02)
        finally:
            try:
                super().close()
            finally:
                logging.info(
                    "Unitree feedback: %d samples, maximum callback gap %.1f ms, "
                    "maximum feedback lock wait %.1f ms",
                    self.feedback_count,
                    self.max_feedback_gap_s * 1000,
                    self.max_feedback_lock_wait_s * 1000,
                )
                first, last = self.first_publish_at, self.last_publish_at
                if self.command_count > 1 and first is not None and last is not None and last > first:
                    logging.info(
                        "Unitree arm publisher: %.1f Hz, maximum interval %.1f ms, %d commands",
                        (self.command_count - 1) / (last - first),
                        1000 * self.max_publish_gap,
                        self.command_count,
                    )

    def _run(self) -> None:
        try:
            while not self.stop.is_set():
                started = self.clock()
                self._step()
                self.stop.wait(max(0.0, 1 / 250 - (self.clock() - started)))
        except Exception as exc:
            with self.lock:
                self.fault = str(exc)
            logging.error(
                "Teleop state: fault; releasing arm authority: %s",
                exc,
                extra={"teleop_voice": "Arm control fault. Releasing. Check the terminal."},
            )
        finally:
            self._release_authority()
