# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Opt-in G1-29 arm SDK execution; never takes whole-body command authority.

Run on the robot-side host. The thread watchdog covers lost caller updates, not
process death, host failure or DDS loss; those require a verified firmware response
and the operator stop arrangement. No MotionSwitcher commands are issued here.
"""

import copy
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
    domain_id: int = 0
    expected_mode_machine: int | None = None
    state_timeout_s: float = 5.0
    max_state_age_s: float = 0.1
    command_timeout_s: float = 0.1
    period_s: float = 0.02
    blend_s: float = 2.0
    max_velocity: float = 0.1
    max_acceleration: float = 0.5
    max_displacement: float = 0.05
    max_tracking_error: float = 0.1
    max_measured_velocity: float = 0.5
    max_feedforward: float = 5.0
    max_tilt_rad: float = 0.05
    gravity_urdf: str | None = None
    kp: list[float] = field(default_factory=list)
    kd: list[float] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    torque_limits: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.read_only, bool):
            raise ValueError("read_only must be boolean")
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
            "period_s",
            "blend_s",
            "max_velocity",
            "max_acceleration",
            "max_displacement",
            "max_tracking_error",
            "max_measured_velocity",
            "max_feedforward",
            "max_tilt_rad",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.period_s > min(self.command_timeout_s, self.max_state_age_s) / 2:
            raise ValueError("Control period must be at most half the freshness bounds")
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
        self.stop = threading.Event()
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
        self.gravity = None
        self.release_sent = False
        self.lease = None

    def _receive(self, msg):
        with self.lock:
            if self.stop.is_set():
                return
            try:
                values = np.array([(m.q, m.dq, m.tau_est) for m in msg.motor_state[:29]])
                if values.shape != (29, 3) or not np.isfinite(values).all():
                    raise ValueError("Invalid body state")
                if not np.isfinite(msg.imu_state.rpy).all():
                    raise ValueError("Invalid IMU state")
                mode = (int(msg.mode_machine), int(msg.mode_pr))
                if self.mode is not None and mode != self.mode:
                    raise ValueError("Robot mode changed")
                tick = int(msg.tick)
                if self.tick is not None:
                    delta = (tick - self.tick) % (2**32)
                    if delta == 0:
                        return  # Repeated snapshots cannot refresh feedback freshness.
                    if delta > 2**31:
                        raise ValueError("Robot tick regressed")
                if self.state_at is not None and self.clock() - self.state_at > self.config.max_state_age_s:
                    raise ValueError("Feedback receive gap exceeded")
                self.state, self.state_at, self.tick, self.mode = copy.deepcopy(msg), self.clock(), tick, mode
            except Exception as exc:
                self.fault = str(exc)

    def _snapshot(self):
        if self.fault:
            raise RuntimeError(self.fault)
        if self.state_at is None or not 0 <= self.clock() - self.state_at <= self.config.max_state_age_s:
            raise RuntimeError("Missing or stale feedback")
        if self.config.lower:
            self._check_pose(self._q(self.state))
        return copy.deepcopy(self.state)

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
            if (
                self.config.expected_mode_machine is None
                or state.mode_machine != self.config.expected_mode_machine
                or state.mode_pr != 0
            ):
                raise RuntimeError("Motion requires reviewed expected mode_machine matching feedback")
            self.initial = self._q(state)
            self._check_pose(self.initial)
            self.target = self.initial.copy()
            self.previous = self.initial.copy()
            self.velocity = np.zeros(14)
            self._check_state(state)
            self.lease = open(f"/tmp/lerobot-g1-arm-{self.config.domain_id}.lock", "a")  # noqa: SIM115 - session lifetime
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.transport.enable_writer()
            self.writer = True
            self.transport.wait_ready(
                arm_command(self.initial, np.zeros(14), 0.0, state.mode_machine, self.config)
            )
            with self.lock:
                state = self._snapshot()
                self.initial = self._q(state)
                self.target = self.initial.copy()
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

    def _check_pose(self, q):
        if np.any(q < self.config.lower) or np.any(q > self.config.upper):
            raise ValueError("Arm target exceeds configured bounds")

    def _check_state(self, state):
        q = self._q(state)
        self._check_pose(q)
        if np.max(np.abs(q - self.initial)) > self.config.max_displacement:
            raise RuntimeError("Measured displacement limit exceeded")
        if max(abs(state.motor_state[s].dq) for s in ARM_SLOTS) > self.config.max_measured_velocity:
            raise RuntimeError("Measured arm velocity limit exceeded")
        if np.max(np.abs(q - self.previous)) > self.config.max_tracking_error:
            raise RuntimeError("Tracking error limit exceeded")
        if self.gravity is not None and (
            max(abs(v) for v in state.imu_state.rpy[:2]) > self.config.max_tilt_rad
            or max(abs(state.motor_state[s].q) for s in (12, 13, 14)) > self.config.max_tilt_rad
        ):
            raise RuntimeError("Reduced gravity model requires upright torso and near-neutral waist")
        return q

    def send(self, action):
        with self.lock:
            if not self.active or self.fault:
                raise RuntimeError(self.fault or "Arm authority not explicitly activated")
            try:
                if set(action) != set(ARM_KEYS):
                    raise ValueError("Each command must contain all and only the 14 arm targets")
                q = np.array([action[key] for key in ARM_KEYS], dtype=float)
                if q.shape != (14,) or not np.isfinite(q).all():
                    raise ValueError("Arm targets must be finite scalars")
                self._snapshot()
                self._check_pose(q)
                if np.max(np.abs(q - self.initial)) > self.config.max_displacement:
                    raise ValueError("Requested displacement limit exceeded")
                self.target = q
                self.last_command = self.clock()
            except Exception as exc:
                self.fault = str(exc)
                raise
        return action.copy()

    def _step(self):
        with self.lock:
            now = self.clock()
            elapsed = now - self.last_step
            if elapsed < 0 or elapsed > self.config.command_timeout_s:
                raise RuntimeError("Control scheduling deadline missed")
            if now - self.last_command > self.config.command_timeout_s:
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
            self._check_pose(command)
            if np.max(np.abs(command - self.initial)) > self.config.max_displacement:
                raise RuntimeError("Slew-limited displacement limit exceeded")
            tau = np.zeros(14) if self.gravity is None else self.gravity.gravity(command)
            if not np.isfinite(tau).all() or np.max(np.abs(tau)) > self.config.max_feedforward:
                raise RuntimeError("Invalid or excessive gravity feedforward")
            dq = np.array([state.motor_state[s].dq for s in ARM_SLOTS])
            estimated = np.asarray(self.config.kp) * (command - q) - np.asarray(self.config.kd) * dq + tau
            if np.any(np.abs(estimated) > self.config.torque_limits) or any(
                abs(state.motor_state[s].tau_est) > self.config.torque_limits[i]
                for i, s in enumerate(ARM_SLOTS)
            ):
                raise RuntimeError("Estimated arm torque limit exceeded")
            self.weight = min(1.0, self.weight + dt / self.config.blend_s)
            self.transport.write(arm_command(command, tau, self.weight, state.mode_machine, self.config))
            self.previous, self.velocity, self.last_step = command, velocity, now

    def _run(self):
        try:
            while not self.stop.wait(self.config.period_s):
                self._step()
        except Exception as exc:
            with self.lock:
                self.fault = str(exc)
        finally:
            with self.lock:
                self.active = False
                self.stop.set()
                # Explicitly selected release-to-stock contract, not a verified physical stop.
                try:
                    for _ in range(3):
                        self.transport.write(
                            arm_command(self.previous, np.zeros(14), 0.0, self.mode[0], self.config)
                        )
                        time.sleep(self.config.period_s)
                    self.release_sent = True
                except Exception as exc:
                    self.fault = f"{self.fault or 'Shutdown'}; release write failed: {exc}"

    def close(self):
        self.stop.set()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("Arm publisher did not stop; operator intervention required")
        self.connected = False
        try:
            self.transport.close()
        finally:
            if self.lease is not None:
                self.lease.close()
                self.lease = None
        if self.fault:
            raise RuntimeError(self.fault)
