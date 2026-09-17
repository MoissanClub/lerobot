# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""BrainCo Revo2 Modbus adapter for bc-stark-sdk 2.0.2.

Hardware access is explicit and disabled by default. This synchronous adapter
owns an asyncio runner and must be used from one non-async control thread.
"""

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .hand_system import HandConfig, HandSystem

MOTORS = tuple(f"motor_{i}" for i in range(6))
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@HandConfig.register_subclass("brainco_revo2")
@dataclass(kw_only=True)
class BrainCoHandConfig(HandConfig):
    end_effector: ClassVar[str] = "brainco"
    port: str
    device_id: int | None = None
    allow_hardware: bool = False
    hardware_type: str = "Revo2Basic"
    tactile: bool = False
    timeout_s: float = 1.0
    speed: float = 0.2
    max_rate: float = 0.5
    max_interval_s: float = 0.1
    upper: list[float] = field(default_factory=lambda: [0.5, 0.5, 1.0, 1.0, 1.0, 1.0])

    def __post_init__(self):
        super().__post_init__()
        if not self.port or not self.port.startswith("/dev/"):
            raise ValueError("Explicit /dev/ serial port is required; no discovery")
        if self.device_id is None:
            self.device_id = 0x7E if self.side == "left" else 0x7F
        if type(self.device_id) is not int or not 1 <= self.device_id <= 247:
            raise ValueError("Modbus device_id must be in [1, 247]")
        if self.hardware_type not in ("Revo2Basic", "Revo2Touch"):
            raise ValueError("Only Revo2Basic and capacitive Revo2Touch are supported")
        if self.tactile and self.hardware_type != "Revo2Touch":
            raise ValueError("Tactile data requires Revo2Touch")
        if (
            not all(
                np.isfinite(v) and v > 0
                for v in (self.timeout_s, self.speed, self.max_rate, self.max_interval_s)
            )
            or not 0.001 <= self.speed <= 1
        ):
            raise ValueError("Timing/rate must be positive finite; speed must be in (0, 1]")
        upper = np.asarray(self.upper, dtype=float)
        if upper.shape != (6,) or not np.all(np.isfinite(upper)) or np.any(upper <= 0) or np.any(upper > 1):
            raise ValueError("upper must contain six normalized limits in (0, 1]")
        if not np.allclose(upper * 1000, np.rint(upper * 1000), rtol=0, atol=1e-8):
            raise ValueError("Limits must match the SDK's 0.001 position resolution")


def load_sdk():
    from importlib.metadata import version

    if version("bc-stark-sdk") != "2.0.2":
        raise ImportError("This adapter is verified against bc-stark-sdk==2.0.2")
    from bc_stark_sdk import main_mod

    return main_mod


class BrainCoHand(HandSystem):
    def __init__(self, config, *, sdk_factory=load_sdk, clock=time.monotonic):
        self.config, self._factory, self._clock = config, sdk_factory, clock
        self.sdk = self.client = self.runner = None
        self._connected = False
        self._target = None

    @property
    def is_connected(self):
        return self._connected

    @property
    def action_features(self):
        return {f"{name}.pos": float for name in MOTORS}

    @property
    def observation_features(self):
        result = dict(self.action_features)
        # Raw SDK feedback units are named explicitly, not mislabeled as SI units.
        for name in MOTORS:
            result.update({f"{name}.speed_raw": float, f"{name}.current_raw": float, f"{name}.state": int})
        if self.config.tactile:
            for finger in FINGERS:
                result.update(
                    {
                        f"{finger}.touch_status": int,
                        f"{finger}.normal_force_raw": float,
                        f"{finger}.tangential_force_raw": float,
                        f"{finger}.proximity_raw": float,
                    }
                )
        return result

    def _call(self, method, *args):
        async def invoke():
            result = method(*args)
            return await result if inspect.isawaitable(result) else result

        async def bounded():
            return await asyncio.wait_for(invoke(), timeout=self.config.timeout_s)

        return self.runner.run(bounded())

    def connect(self):
        if self.runner is not None:
            raise RuntimeError("Already connected or cleanup required")
        if not self.config.allow_hardware:
            raise PermissionError(
                "Hardware access disabled; explicitly set allow_hardware after physical preflight"
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("BrainCoHand is synchronous; use a dedicated non-async thread")
        self.runner = asyncio.Runner()
        try:
            self.sdk = self._factory()
            self.client = self._call(self.sdk.modbus_open, self.config.port, self.sdk.Baudrate.Baud460800)
            if self.client is None:
                raise RuntimeError("SDK did not open a serial client")
            identity = self._call(self.client.get_device_info, self.config.device_id)
            expected_side = getattr(self.sdk.HandType, self.config.side.capitalize())
            expected_model = getattr(self.sdk.StarkHardwareType, self.config.hardware_type)
            if (
                identity is None
                or identity.hand_type != expected_side
                or identity.get_hardware_type() != expected_model
            ):
                raise ValueError("Connected hand side/model does not match configuration")
            self._call(
                self.client.set_finger_unit_mode, self.config.device_id, self.sdk.FingerUnitMode.Normalized
            )
            mode = self._call(self.client.get_finger_unit_mode, self.config.device_id)
            if mode != self.sdk.FingerUnitMode.Normalized:
                raise ValueError("Hand did not confirm normalized units")
            if self.config.tactile:
                if not identity.uses_revo2_touch_api():
                    raise ValueError("Connected hand does not expose Revo2 capacitive touch")
                self._call(self.client.touch_sensor_setup, self.config.device_id, 0x1F)
                if self._call(self.client.get_touch_sensor_enabled, self.config.device_id) & 0x1F != 0x1F:
                    raise ValueError("Tactile sensors are not all enabled")
            status = self._call(self.client.get_motor_status, self.config.device_id)
            self._target = self._positions(status)
            if np.any(self._target > self.config.upper):
                raise ValueError("Measured hand pose exceeds configured limits; no automatic motion")
            self._last_command_at = self._clock()
            self._connected = True
        except BaseException as exc:
            try:
                self.disconnect()
            except Exception as cleanup:
                exc.add_note(f"Hand cleanup failed: {cleanup}")
            raise

    @staticmethod
    def _vector(values, name):
        result = np.asarray(values, dtype=float)
        if result.shape != (6,) or not np.all(np.isfinite(result)):
            raise ValueError(f"Malformed SDK {name}")
        return result

    def _positions(self, status):
        values = self._vector(status.positions, "positions") / 1000
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("SDK position outside normalized range")
        return values

    def _require_connected(self):
        if not self.is_connected:
            raise RuntimeError("Hand not connected")

    def closure_action(self, closure):
        if not np.isscalar(closure) or not np.isfinite(closure) or not 0 <= closure <= 1:
            raise ValueError("Closure must be finite in [0, 1]")
        return {
            f"{name}.pos": float(closure * limit)
            for name, limit in zip(MOTORS, self.config.upper, strict=True)
        }

    def validate_action(self, action):
        self._require_connected()
        if set(action) - set(self.action_features):
            raise ValueError("Unknown hand motor")
        requested = self._target.copy()
        for i, name in enumerate(MOTORS):
            key = f"{name}.pos"
            if key in action:
                value = action[key]
                if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= self.config.upper[i]:
                    raise ValueError("Hand target exceeds configured normalized limits")
                requested[i] = value
        elapsed = np.clip(self._clock() - self._last_command_at, 0, self.config.max_interval_s)
        delta = np.floor(elapsed * self.config.max_rate * 1000) / 1000
        bounded = np.clip(requested, self._target - delta, self._target + delta)
        return {f"{name}.pos": float(value) for name, value in zip(MOTORS, bounded, strict=True)}

    def send_action(self, action):
        bounded = self.validate_action(action)
        positions = np.rint(np.array(list(bounded.values())) * 1000).astype(int).tolist()
        try:
            self._call(
                self.client.set_finger_positions_and_speeds,
                self.config.device_id,
                positions,
                [round(self.config.speed * 1000)] * 6,
            )
        except BaseException as exc:
            self._connected = False
            try:
                self.disconnect()
            except Exception as cleanup:
                exc.add_note(f"Hand cleanup failed: {cleanup}")
            raise
        self._target = np.array(positions, dtype=float) / 1000
        self._last_command_at = self._clock()
        return {f"{name}.pos": float(value) for name, value in zip(MOTORS, self._target, strict=True)}

    def get_observation(self):
        self._require_connected()
        try:
            status = self._call(self.client.get_motor_status, self.config.device_id)
            positions = self._positions(status)
            speeds = self._vector(status.speeds, "speeds")
            currents = self._vector(status.currents, "currents")
            states = self._vector(status.states, "states")
            if np.any(states != states.astype(int)):
                raise ValueError("Malformed SDK motor state")
            result = {}
            for i, name in enumerate(MOTORS):
                result.update(
                    {
                        f"{name}.pos": float(positions[i]),
                        f"{name}.speed_raw": float(speeds[i]),
                        f"{name}.current_raw": float(currents[i]),
                        f"{name}.state": int(states[i]),
                    }
                )
            if self.config.tactile:
                fingers = self._call(self.client.get_touch_sensor_status, self.config.device_id)
                if len(fingers) != 5:
                    raise ValueError("Expected five tactile fingers")
                for name, item in zip(FINGERS, fingers, strict=True):
                    raw = [item.normal_force1, item.tangential_force1, item.self_proximity1]
                    if item.status != 0 or not np.all(np.isfinite(raw)):
                        raise ValueError("Tactile feedback reports a fault or nonfinite sample")
                    result.update(
                        {
                            f"{name}.touch_status": int(item.status),
                            f"{name}.normal_force_raw": float(raw[0]),
                            f"{name}.tangential_force_raw": float(raw[1]),
                            f"{name}.proximity_raw": float(raw[2]),
                        }
                    )
            return result
        except BaseException as exc:
            self._connected = False
            try:
                self.disconnect()
            except Exception as cleanup:
                exc.add_note(f"Hand cleanup failed: {cleanup}")
            raise

    def disconnect(self):
        self._connected = False
        try:
            if self.client is not None:
                self._call(self.sdk.modbus_close, self.client)
        finally:
            self.client = None
            if self.runner is not None:
                self.runner.close()
                self.runner = None
            self._target = None
