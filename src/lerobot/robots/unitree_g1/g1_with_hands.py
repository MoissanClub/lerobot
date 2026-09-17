# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Compatibility wrapper; new G1 integrations should use UnitreeG1Config.hands."""

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .config_unitree_g1 import UnitreeG1Config
from .hand_system import HandConfig, make_hand
from .unitree_g1 import UnitreeG1


@RobotConfig.register_subclass("unitree_g1_with_hands")
@dataclass(kw_only=True)
class G1WithHandsConfig(RobotConfig):
    body: UnitreeG1Config = field(default_factory=UnitreeG1Config)
    hands: dict[str, HandConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        for side, config in self.hands.items():
            if side not in ("left", "right") or config.side != side:
                raise ValueError("Hand dictionary key must match configured left/right side")


class G1WithHands(Robot):
    config_class = G1WithHandsConfig
    name = "unitree_g1_with_hands"

    def __init__(self, config, *, body_factory=UnitreeG1, hand_factory=make_hand):
        if body_factory is UnitreeG1 and config.hands and config.body.is_simulation:
            raise ValueError("Configured physical hand drivers cannot be used in simulation")
        if config.body.hands:
            raise ValueError("Configure hands on the body or compatibility wrapper, not both")
        self._connected = False
        super().__init__(config)
        self.config = config
        self.body = body_factory(config.body)
        self.hands = {side: hand_factory(cfg) for side, cfg in config.hands.items()}
        for name in ("action_features", "observation_features"):
            if any(key.startswith("hands.") for key in getattr(self.body, name)):
                raise ValueError("Body uses reserved hands namespace")

    def _features(self, name):
        result = dict(getattr(self.body, name))
        for side, hand in self.hands.items():
            result.update({f"hands.{side}.{key}": value for key, value in getattr(hand, name).items()})
        return result

    @property
    def action_features(self):
        return self._features("action_features")

    @property
    def observation_features(self):
        return self._features("observation_features")

    @property
    def is_connected(self):
        return (
            self._connected
            and self.body.is_connected
            and all(hand.is_connected for hand in self.hands.values())
        )

    @property
    def is_calibrated(self):
        return self.body.is_calibrated

    @property
    def cameras(self):
        return self.body.cameras

    def calibrate(self):
        self.body.calibrate()

    def configure(self):
        self.body.configure()

    def connect(self, calibrate=True):
        if (
            self._connected
            or self.body.is_connected
            or any(hand.is_connected for hand in self.hands.values())
        ):
            raise RuntimeError("Already connected or partially connected")
        try:
            self.body.connect(calibrate=calibrate)
            for hand in self.hands.values():
                hand.connect()
            self._connected = True
        except BaseException as exc:
            self._cleanup_after_failure(exc)
            raise

    def _cleanup_after_failure(self, exc):
        try:
            self.disconnect()
        except Exception as cleanup_error:
            exc.add_note(f"Cleanup also failed: {cleanup_error}")

    def disconnect(self):
        self._connected = False
        errors = []
        for child in [*reversed(list(self.hands.values())), self.body]:
            try:
                child.disconnect()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("One or more devices failed to disconnect", errors)

    def _require_connected(self):
        if not self.is_connected:
            raise RuntimeError("Body and configured hands must all be connected")

    def get_observation(self):
        self._require_connected()
        try:
            result = dict(self.body.get_observation())
            for side, hand in self.hands.items():
                values = hand.get_observation()
                if set(values) != set(hand.observation_features):
                    raise ValueError("Hand observation does not match declared features")
                result.update({f"hands.{side}.{key}": value for key, value in values.items()})
            return result
        except BaseException as exc:
            self._cleanup_after_failure(exc)
            raise

    def send_action(self, action):
        self._require_connected()
        if set(action) - set(self.action_features):
            raise ValueError("Unknown action keys")
        body_action = {key: value for key, value in action.items() if not key.startswith("hands.")}
        hand_actions = {}
        # Validate every hand before any body or hand write; transport writes are not atomic.
        for side, hand in self.hands.items():
            prefix = f"hands.{side}."
            local = {key[len(prefix) :]: value for key, value in action.items() if key.startswith(prefix)}
            if local:
                hand_actions[side] = hand.validate_action(local)
        try:
            result = dict(self.body.send_action(body_action)) if body_action or not self.hands else {}
            for side, local in hand_actions.items():
                sent = self.hands[side].send_action(local)
                result.update({f"hands.{side}.{key}": value for key, value in sent.items()})
            return result
        except BaseException as exc:
            self._cleanup_after_failure(exc)
            raise
