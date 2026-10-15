# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Opt-in G1-29 arm SDK execution; never takes whole-body command authority.

Run on the robot-side host. The thread watchdog covers lost caller updates, not
process death, host failure or DDS loss; those require a verified firmware response
and the operator stop arrangement. No MotionSwitcher commands are issued here.
"""

import copy
import logging
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .g1_utils import G1_29_JointArmIndex, G1_29_JointIndex

ARM_JOINTS = tuple(G1_29_JointArmIndex)
ARM_KEYS = tuple(f"{joint.name}.q" for joint in ARM_JOINTS)
ARM_SLOTS = tuple(joint.value for joint in ARM_JOINTS)
WEIGHT_SLOT = 29


@dataclass
class G1ArmSDKConfig:
    network_interface: str
    read_only: bool = True
    # Unitree motion/return gates only; experimental watchdogs and motion trips are disabled.
    unitree_gating: bool = False
    domain_id: int = 0
    expected_mode_machine: int | None = None
    state_timeout_s: float = 5.0
    max_state_age_s: float = 0.1
    command_timeout_s: float = 0.1
    # Unitree reference profile: release after a prolonged caller outage.
    command_fault_timeout_s: float = 1.0
    period_s: float = 0.02
    blend_s: float = 2.0
    return_velocity: float = 0.5
    return_tolerance: float = 0.05
    return_timeout_s: float = 5.0
    return_fade_s: float = 2.0
    ramp_authority: bool = True
    measured_target_clipping: bool = False
    max_velocity: float = 0.1
    max_acceleration: float = 0.5
    max_displacement: float = 0.05
    max_wrist_displacement: float | None = None
    workspace_lower: list[float] = field(default_factory=list)
    workspace_upper: list[float] = field(default_factory=list)
    start_position: list[float] = field(default_factory=list)
    start_tolerance: float = 0.05
    max_tracking_error: float = 0.1
    max_preparation_tracking_error: float | None = None
    max_measured_velocity: float = 0.5
    max_preparation_velocity: float | None = None
    max_feedforward: float = 5.0
    max_tilt_rad: float = 0.05
    gravity_urdf: str | None = None
    kp: list[float] = field(default_factory=list)
    kd: list[float] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    torque_limits: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.unitree_gating, bool):
            raise ValueError("unitree_gating must be boolean")
        if not isinstance(self.read_only, bool):
            raise ValueError("read_only must be boolean")
        if not isinstance(self.ramp_authority, bool):
            raise ValueError("ramp_authority must be boolean")
        if not isinstance(self.measured_target_clipping, bool):
            raise ValueError("measured_target_clipping must be boolean")
        if not self.network_interface or self.network_interface == "lo":
            raise ValueError("Arm SDK requires an explicit non-loopback interface")
        if isinstance(self.domain_id, bool) or not isinstance(self.domain_id, int) or self.domain_id < 0:
            raise ValueError("domain_id must be a nonnegative integer")
        if self.expected_mode_machine is not None and (
            type(self.expected_mode_machine) is not int or not 0 <= self.expected_mode_machine <= 255
        ):
            raise ValueError("expected_mode_machine must be an integer byte")
        for name in (
            "state_timeout_s",
            "max_state_age_s",
            "command_timeout_s",
            "command_fault_timeout_s",
            "period_s",
            "blend_s",
            "return_velocity",
            "return_tolerance",
            "return_timeout_s",
            "return_fade_s",
            "max_velocity",
            "max_acceleration",
            "max_displacement",
            "max_tracking_error",
            "max_measured_velocity",
            "max_feedforward",
            "max_tilt_rad",
            "start_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.period_s > min(self.command_timeout_s, self.max_state_age_s) / 2:
            raise ValueError("Control period must be at most half the freshness bounds")
        if self.max_wrist_displacement is not None and (
            not math.isfinite(self.max_wrist_displacement) or self.max_wrist_displacement <= 0
        ):
            raise ValueError("max_wrist_displacement must be positive and finite")
        if self.max_preparation_tracking_error is not None and (
            not math.isfinite(self.max_preparation_tracking_error) or self.max_preparation_tracking_error <= 0
        ):
            raise ValueError("max_preparation_tracking_error must be positive and finite")
        if self.max_preparation_velocity is not None and (
            not math.isfinite(self.max_preparation_velocity) or self.max_preparation_velocity <= 0
        ):
            raise ValueError("max_preparation_velocity must be positive and finite")
        if (self.workspace_lower or self.workspace_upper) and (
            len(self.workspace_lower) != 3
            or len(self.workspace_upper) != 3
            or not np.isfinite([self.workspace_lower, self.workspace_upper]).all()
            or np.any(np.array(self.workspace_lower) >= self.workspace_upper)
        ):
            raise ValueError("Workspace bounds must be finite ordered xyz triples")
        if self.start_position and (
            len(self.start_position) != 14 or not np.isfinite(self.start_position).all()
        ):
            raise ValueError("start_position requires 14 finite arm joint values")
        for name in ("kp", "kd", "lower", "upper", "torque_limits"):
            values = getattr(self, name)
            if (values or not self.read_only) and (len(values) != 14 or not np.isfinite(values).all()):
                raise ValueError(f"{name} requires 14 finite values in arm DDS order")
        if not self.read_only and (min(self.kp) <= 0 or min(self.kd) <= 0 or min(self.torque_limits) <= 0):
            raise ValueError("Explicit positive kp/kd required")
        if self.lower and (not self.upper or np.any(np.asarray(self.lower) >= self.upper)):
            raise ValueError("Joint bounds must be ordered")


class ArmSDKTransport:
    """Lazy SDK adapter. Read-only connect creates no publisher or RPC client."""

    def connect(self, config, callback):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        if config.network_interface not in dict(socket.if_nameindex()).values():
            raise ValueError("Unknown network interface")
        self.publisher = None
        self.subscriber = None
        self.timeout = config.period_s
        self.startup_timeout = config.state_timeout_s
        ChannelFactoryInitialize(config.domain_id, config.network_interface)
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(callback, 1)

    def enable_writer(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        self.crc = CRC()
        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()

    def write(self, command):
        command.crc = self.crc.Crc(command)
        if not self.publisher.Write(command, self.timeout):
            raise RuntimeError("Arm SDK write failed")

    def wait_ready(self, disabled_command):
        # SDK timeout waits for a matched subscriber (not a hardware execution ACK).
        disabled_command.crc = self.crc.Crc(disabled_command)
        if not self.publisher.Write(disabled_command, self.startup_timeout):
            raise RuntimeError("No matched arm SDK subscriber before timeout")

    def close(self):
        errors = []
        for name in ("publisher", "subscriber"):
            channel = getattr(self, name, None)
            if channel is not None:
                try:
                    channel.Close()
                    setattr(self, name, None)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("Arm SDK transport cleanup failed", errors)


def arm_command(q, tau, weight, mode_machine, config):
    """All unowned slots stay zero; slot 29 is a protocol weight, not a joint."""
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

    msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr = 0
    msg.mode_machine = mode_machine
    for i, slot in enumerate(ARM_SLOTS):
        cmd = msg.motor_cmd[slot]
        cmd.mode = 1
        cmd.q, cmd.dq, cmd.tau = float(q[i]), 0.0, float(tau[i])
        cmd.kp, cmd.kd = config.kp[i], config.kd[i]
    msg.motor_cmd[WEIGHT_SLOT].q = float(weight)
    return msg


class G1ArmSDK:
    def __init__(self, config, *, transport=None, clock=time.monotonic):
        self.config = copy.deepcopy(config)
        self.transport = transport if transport is not None else ArmSDKTransport()
        self.clock = clock
        self.lock = threading.RLock()
        # Lock order is command lock -> feedback lock. The callback only takes
        # feedback_lock, so holding the command lock cannot block the callback.
        self.feedback_lock = threading.RLock()
        self.feedback_count = 0
        self.max_feedback_gap_s = 0.0
        self.max_feedback_lock_wait_s = 0.0
        self.stop = threading.Event()
        self.return_lock = threading.Lock()
        self.thread = None
        self.state = None
        self.state_at = None
        self.tick = None
        self.mode = None
        self.fault = None
        self.connected = False
        self.used = False
        self.active = False
        self.writer = False
        self.weight = 0.0
        self.use_feedforward = True
        self.last_feedforward = np.zeros(14)
        self.preparing = config.max_preparation_tracking_error is not None
        self.gravity = None
        self.release_sent = False
        self.returning_home = False
        self.return_attempted = False
        self.return_succeeded = False
        self.handover_phase = "inactive"
        self.lease = None

    def _receive(self, msg):
        received_at = self.clock()
        with self.feedback_lock:
            lock_wait = self.clock() - received_at
            self.max_feedback_lock_wait_s = max(self.max_feedback_lock_wait_s, lock_wait)
            try:
                values = np.array([(m.q, m.dq, m.tau_est) for m in msg.motor_state[:29]])
                if values.shape != (29, 3) or not np.isfinite(values).all():
                    raise ValueError("Invalid body state")
                if not np.isfinite(msg.imu_state.rpy).all():
                    raise ValueError("Invalid IMU state")
                mode = (int(msg.mode_machine), int(msg.mode_pr))
                if not self.config.unitree_gating and self.mode is not None and mode != self.mode:
                    raise ValueError("Robot mode changed")
                tick = int(msg.tick)
                if not self.config.unitree_gating and self.tick is not None:
                    delta = (tick - self.tick) % (2**32)
                    if delta == 0:
                        return  # Repeated snapshots cannot refresh feedback freshness.
                    if delta > 2**31:
                        raise ValueError("Robot tick regressed")
                gap = 0.0 if self.state_at is None else received_at - self.state_at
                self.max_feedback_gap_s = max(self.max_feedback_gap_s, gap)
                if not self.config.unitree_gating and self.active and gap > self.config.max_state_age_s:
                    raise ValueError(
                        "Feedback receive gap exceeded: "
                        f"callback gap={gap * 1000:.1f} ms, lock wait={lock_wait * 1000:.1f} ms, "
                        f"limit={self.config.max_state_age_s * 1000:.1f} ms"
                    )
                self.state, self.state_at, self.tick, self.mode = (
                    self._copy_state(msg),
                    received_at,
                    tick,
                    mode,
                )
                self.feedback_count += 1
            except Exception as exc:
                self.fault = str(exc)

    def _snapshot(self):
        with self.feedback_lock:
            if self.fault:
                raise RuntimeError(self.fault)
            if self.state_at is None or (
                not self.config.unitree_gating
                and not 0 <= self.clock() - self.state_at <= self.config.max_state_age_s
            ):
                raise RuntimeError("Missing or stale feedback")
            if self.config.lower:
                self._check_pose(self._q(self.state))
            return self._copy_state(self.state)

    def _copy_state(self, state):
        return copy.deepcopy(state)

    def connect(self):
        if self.used:
            raise RuntimeError("Use a fresh process/session; reconnect is not supported")
        self.used = True
        try:
            # Load optional model code before feedback timing or command authority starts.
            if not self.config.read_only and self.config.gravity_urdf:
                from .g1_vr_control import G1VRKinematics

                self.gravity = G1VRKinematics(Path(self.config.gravity_urdf).absolute().parents[1])
                if np.any(np.asarray(self.config.lower) < self.gravity.lower) or np.any(
                    np.asarray(self.config.upper) > self.gravity.upper
                ):
                    raise ValueError("Configured bounds exceed gravity model limits")
            self.transport.connect(self.config, self._receive)
            deadline = self.clock() + self.config.state_timeout_s
            while self.state is None:
                if self.fault or self.clock() >= deadline:
                    raise RuntimeError(self.fault or "Timed out waiting for feedback")
                time.sleep(0.005)
            with self.lock:
                self._snapshot()
            self.connected = True
        except BaseException:
            self.transport.close()
            raise

    def observation(self):
        with self.lock:
            if not self.connected:
                raise RuntimeError("Arm SDK session is disconnected")
            state = self._snapshot()
            obs = {}
            for joint in G1_29_JointIndex:
                motor = state.motor_state[joint.value]
                obs.update(
                    {
                        f"{joint.name}.q": motor.q,
                        f"{joint.name}.dq": motor.dq,
                        f"{joint.name}.tau": motor.tau_est,
                    }
                )
            return obs

    def activate(self):
        """Explicit torque-enabling operation, never called by connect/send_action."""
        import fcntl

        with self.lock:
            if self.config.read_only or not self.connected or self.writer:
                raise RuntimeError("Motion requires connected, motion-enabled, unused session")
            state = self._snapshot()
            if not self.config.unitree_gating and (
                self.config.expected_mode_machine is None
                or state.mode_machine != self.config.expected_mode_machine
                or state.mode_pr != 0
            ):
                raise RuntimeError("Motion requires reviewed expected mode_machine matching feedback")
            self.initial = self._q(state)
            if (
                not self.config.unitree_gating
                and self.config.start_position
                and np.max(np.abs(self.initial - self.config.start_position)) >= self.config.start_tolerance
            ):
                raise RuntimeError(
                    "Arms are outside the starting pose tolerance; no publisher opened. "
                    f"Measured arm positions: {self.initial.tolist()}"
                )
            if (
                not self.config.unitree_gating
                and (self.config.max_wrist_displacement is not None or self.config.workspace_lower)
                and self.gravity is None
            ):
                raise RuntimeError("Wrist displacement checks require the gravity/kinematics model")
            self._check_pose(self.initial)
            self.target = self.initial.copy()
            self.previous = self.initial.copy()
            self.velocity = np.zeros(14)
            self._check_state(state)
            self.lease = open(  # noqa: SIM115 - session lifetime
                f"/tmp/lerobot-g1-arm-{self.config.domain_id}.lock",  # nosec B108
                "a",
            )
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.transport.enable_writer()
            self.writer = True
            self.transport.wait_ready(
                arm_command(self.initial, np.zeros(14), 0.0, state.mode_machine, self.config)
            )
            # DDS endpoint discovery may briefly delay callbacks. No active
            # command has been sent; wait for fresh feedback before takeover.
            deadline = self.clock() + self.config.state_timeout_s
            while self.state_at is None or (
                not self.config.unitree_gating and self.clock() - self.state_at > self.config.max_state_age_s
            ):
                if self.fault or self.clock() >= deadline:
                    raise RuntimeError(self.fault or "Timed out waiting for fresh activation feedback")
                time.sleep(0.002)
            with self.lock:
                state = self._snapshot()
                self.initial = self._q(state)
                self.target = self._activation_target(self.initial)
                self.previous = self.initial.copy()
                self._check_state(state)
                self.last_command = self.clock()
                self.last_step = self.clock()
                self.active = True
                self.thread = threading.Thread(target=self._run, name="g1-arm-sdk-watchdog", daemon=True)
                self.thread.start()
        except BaseException:
            self.active = False
            self.lease.close()
            self.lease = None
            raise

    @staticmethod
    def _q(state):
        return np.array([state.motor_state[slot].q for slot in ARM_SLOTS])

    def _activation_target(self, measured: np.ndarray) -> np.ndarray:
        return measured.copy()

    def _check_pose(self, q):
        if self.config.unitree_gating:
            return
        if np.any(q < self.config.lower) or np.any(q > self.config.upper):
            raise ValueError("Arm target exceeds configured bounds")

    def _check_state(self, state):
        q = self._q(state)
        if self.config.unitree_gating:
            return q
        self._check_pose(q)
        if np.max(np.abs(q - self.initial)) > self.config.max_displacement:
            raise RuntimeError("Measured displacement limit exceeded")
        self._check_wrist_displacement(q)
        fastest = max(ARM_JOINTS, key=lambda joint: abs(state.motor_state[joint.value].dq))
        measured_velocity = state.motor_state[fastest.value].dq
        velocity_limit = (
            self.config.max_preparation_velocity
            if self.preparing and self.config.max_preparation_velocity is not None
            else self.config.max_measured_velocity
        )
        if abs(measured_velocity) > velocity_limit:
            raise RuntimeError(
                "Measured arm velocity limit exceeded: "
                f"{fastest.name}={measured_velocity:.4f} rad/s, "
                f"limit={velocity_limit:.4f} rad/s, blend={self.weight:.3f}"
            )
        error_limit = (
            self.config.max_preparation_tracking_error
            if self.preparing and self.config.max_preparation_tracking_error is not None
            else self.config.max_tracking_error
        )
        worst = int(np.argmax(np.abs(q - self.previous)))
        error = float(abs(q[worst] - self.previous[worst]))
        if error > error_limit:
            phase = "preparation" if self.preparing else "following/hold"
            raise RuntimeError(
                f"Tracking error limit exceeded during {phase}: "
                f"{ARM_JOINTS[worst].name} error={error:.4f} rad, limit={error_limit:.4f} rad"
            )
        if (
            self.gravity is not None
            and max(abs(v) for v in state.imu_state.rpy[:2]) > self.config.max_tilt_rad
        ):
            posture = dict(
                zip(
                    ("IMU roll", "IMU pitch"),
                    state.imu_state.rpy[:2],
                    strict=True,
                )
            )
            exceeded = ", ".join(
                f"{name}={value:.4f} rad ({np.degrees(value):.2f} deg)"
                for name, value in posture.items()
                if abs(value) > self.config.max_tilt_rad
            )
            raise RuntimeError(
                "Reduced gravity model requires upright torso. "
                f"Outside posture limit: {exceeded}. "
                f"Limit: +/-{self.config.max_tilt_rad:.4f} rad "
                f"({np.degrees(self.config.max_tilt_rad):.2f} deg). "
                "Return the robot to upright posture using its normal controls."
            )
        return q

    def _check_wrist_displacement(self, q: np.ndarray) -> None:
        if self.config.unitree_gating:
            return
        if self.config.workspace_lower:
            for side, wrist in zip(("left", "right"), self.gravity.fk(q), strict=True):
                if np.any(wrist[:3, 3] < self.config.workspace_lower) or np.any(
                    wrist[:3, 3] > self.config.workspace_upper
                ):
                    raise RuntimeError(
                        "Robot wrist outside configured Cartesian workspace: "
                        f"{side} xyz={wrist[:3, 3].round(4).tolist()}, "
                        f"lower={self.config.workspace_lower}, upper={self.config.workspace_upper}, "
                        f"blend={self.weight:.3f}"
                    )
        if self.config.max_wrist_displacement is None:
            return
        origin = self.gravity.fk(self.initial)
        current = self.gravity.fk(q)
        if any(
            np.linalg.norm(pose[:3, 3] - start[:3, 3]) > self.config.max_wrist_displacement
            for pose, start in zip(current, origin, strict=True)
        ):
            raise RuntimeError("Wrist displacement limit exceeded")

    def send(self, action, *, use_feedforward: bool = True, preparing: bool = False):
        with self.lock:
            if not self.active or self.fault or self.returning_home:
                raise RuntimeError(self.fault or "Arm authority not explicitly activated")
            try:
                if set(action) != set(ARM_KEYS):
                    raise ValueError("Each command must contain all and only the 14 arm targets")
                q = np.array([action[key] for key in ARM_KEYS], dtype=float)
                if q.shape != (14,) or not np.isfinite(q).all():
                    raise ValueError("Arm targets must be finite scalars")
                self._snapshot()
                self._check_pose(q)
                if (
                    not self.config.unitree_gating
                    and np.max(np.abs(q - self.initial)) > self.config.max_displacement
                ):
                    raise ValueError("Requested displacement limit exceeded")
                self._check_wrist_displacement(q)
                self.target = q
                self.use_feedforward = use_feedforward
                self.preparing = preparing
                self.last_command = self.clock()
            except Exception as exc:
                self.fault = str(exc)
                raise
        return action.copy()

    def _step(self):
        with self.lock:
            now = self.clock()
            elapsed = now - self.last_step
            if not self.config.unitree_gating and (elapsed < 0 or elapsed > self.config.command_timeout_s):
                raise RuntimeError("Control scheduling deadline missed")
            if not self.config.unitree_gating and now - self.last_command > self.config.command_timeout_s:
                raise RuntimeError("Caller command timeout")
            dt = min(elapsed, self.config.period_s)
            state = self._snapshot()
            q = self._check_state(state)
            # A stopping-distance envelope limits speed before reaching a target.
            delta = self.target - self.previous
            desired = np.sign(delta) * np.minimum(
                self.config.max_velocity, np.sqrt(2 * self.config.max_acceleration * np.abs(delta))
            )
            velocity = self.velocity + np.clip(
                desired - self.velocity, -self.config.max_acceleration * dt, self.config.max_acceleration * dt
            )
            command = self.previous + velocity * dt
            if self.config.measured_target_clipping:
                # Same proportional, measured-position-relative clipping as
                # G1_29_ArmController.clip_arm_q_target (all axes share a scale).
                delta = self.target - q
                scale = max(
                    float(np.max(np.abs(delta))) / (self.config.max_velocity * self.config.period_s), 1.0
                )
                command = q + delta / scale
            self._check_pose(command)
            if (
                not self.config.unitree_gating
                and np.max(np.abs(command - self.initial)) > self.config.max_displacement
            ):
                raise RuntimeError("Slew-limited displacement limit exceeded")
            self._check_wrist_displacement(command)
            tau = (
                np.zeros(14)
                if self.gravity is None or not self.use_feedforward
                else self.gravity.gravity(command)
            )
            if not np.isfinite(tau).all() or (
                not self.config.unitree_gating and np.max(np.abs(tau)) > self.config.max_feedforward
            ):
                raise RuntimeError("Invalid or excessive gravity feedforward")
            dq = np.array([state.motor_state[s].dq for s in ARM_SLOTS])
            estimated = np.asarray(self.config.kp) * (command - q) - np.asarray(self.config.kd) * dq + tau
            if not self.config.unitree_gating and (
                np.any(np.abs(estimated) > self.config.torque_limits)
                or any(
                    abs(state.motor_state[s].tau_est) > self.config.torque_limits[i]
                    for i, s in enumerate(ARM_SLOTS)
                )
            ):
                raise RuntimeError("Estimated arm torque limit exceeded")
            # Unitree's arm SDK motion-mode controller takes full authority;
            # trajectory speed is bounded independently of this protocol weight.
            self.weight = (
                min(1.0, self.weight + dt / self.config.blend_s) if self.config.ramp_authority else 1.0
            )
            self.transport.write(arm_command(command, tau, self.weight, state.mode_machine, self.config))
            self.last_feedforward = tau.copy()
            self.previous, self.velocity, self.last_step = command, velocity, now

    def _run(self):
        try:
            while not self.stop.wait(self.config.period_s):
                self._step()
        except Exception as exc:
            with self.lock:
                self.fault = str(exc)
            logging.error("G1 arm control fault; attempting stock-controller return: %s", exc)
        finally:
            self.return_to_stock()

    def _return_snapshot(self):
        """Read feedback during cleanup without masking it with the initiating fault."""
        with self.feedback_lock:
            if self.state_at is None or (
                not self.config.unitree_gating
                and not 0 <= self.clock() - self.state_at <= self.config.max_state_age_s
            ):
                raise RuntimeError("Missing or stale feedback during stock-controller return")
            state = self._copy_state(self.state)
        q = self._q(state)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise RuntimeError("Invalid arm feedback during stock-controller return")
        return state, q

    def return_to_stock(self) -> bool:
        """Best-effort zero-pose return followed by a gradual authority fade.

        Keep measured-relative target clipping and the last published feedforward
        throughout the zero-pose return and authority fade, as in xr_teleoperate.
        The clipping bounds target error, not measured joint speed. Cleanup failures are recorded but
        never escape this method, so every exit path can call it safely.
        """
        with self.return_lock:
            if self.return_attempted:
                return self.return_succeeded
            self.return_attempted = True
            if not self.active or not self.writer:
                return False

            original_fault = self.fault
            self.returning_home = True
            self.handover_phase = "return"
            self.release_sent = False
            zero = np.zeros(14)
            self.target = zero.copy()
            return_weight = self.weight
            return_tau = self.last_feedforward.copy()
            try:
                logging.info("Returning G1 arms to zero pose before restoring stock-controller authority")
                deadline = self.clock() + self.config.return_timeout_s
                arrived = False
                while self.clock() < deadline:
                    state, measured = self._return_snapshot()
                    last_command = self._return_target(measured)
                    self.transport.write(
                        arm_command(last_command, return_tau, return_weight, state.mode_machine, self.config)
                    )
                    self.previous = last_command
                    if np.all(np.abs(measured) < self.config.return_tolerance):
                        arrived = True
                        break
                    remaining = deadline - self.clock()
                    if remaining > 0:
                        time.sleep(min(self.config.period_s, remaining))
                if not arrived:
                    raise RuntimeError("Timed out before measured arms reached the return pose")

                # Keep publishing at the control cadence while authority changes.
                # Arrival tolerance must not bypass clipping at the fade boundary.
                fade_steps = max(1, math.ceil(self.config.return_fade_s / self.config.period_s)) + 1
                fade_period = self.config.return_fade_s / (fade_steps - 1)
                time.sleep(self.config.period_s)
                self.handover_phase = "fade"
                fade_start = self.clock()
                for index in range(fade_steps):
                    delay = fade_start + index * fade_period - self.clock()
                    if delay > 0:
                        time.sleep(delay)
                    fraction = min(max((self.clock() - fade_start) / self.config.return_fade_s, 0.0), 1.0)
                    weight = return_weight * (1.0 - fraction) if index < fade_steps - 1 else 0.0
                    state, measured = self._return_snapshot()
                    last_command = self._return_target(measured)
                    self.transport.write(
                        arm_command(last_command, return_tau, float(weight), state.mode_machine, self.config)
                    )
                    self.previous = last_command
                    self.weight = float(weight)
                    if weight == 0.0:
                        break
                self.release_sent = True
                self.return_succeeded = True
                self.handover_phase = "complete"
                logging.info("G1 arm authority returned to the stock controller")
            except BaseException as exc:
                self.handover_phase = "failed"
                message = f"Stock-controller return failed: {exc}"
                self.fault = f"{original_fault}; {message}" if original_fault else message
                logging.exception(message)
            finally:
                self.active = False
                self.stop.set()
            return self.return_succeeded

    def _return_target(self, measured: np.ndarray) -> np.ndarray:
        """Clip the zero-pose goal relative to feedback, including during fade."""
        scale = max(
            float(np.max(np.abs(measured))) / (self.config.return_velocity * self.config.period_s),
            1.0,
        )
        return measured - measured / scale

    def close(self):
        self.stop.set()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join(timeout=self.config.return_timeout_s + self.config.return_fade_s + 2)
            if self.thread.is_alive():
                raise RuntimeError("Arm publisher did not stop; operator intervention required")
        elif self.active:
            self.return_to_stock()
        self.connected = False
        try:
            self.transport.close()
        finally:
            if self.lease is not None:
                self.lease.close()
                self.lease = None
        if self.fault:
            raise RuntimeError(self.fault)
