# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Namespaced hand composition without another robot identity or body publisher."""

from .hand_system import make_hand


class HandCollection:
    def __init__(self, configs):
        self.devices = {side: make_hand(config) for side, config in configs.items()}

    def features(self, kind):
        return {
            f"hands.{side}.{key}": value
            for side, hand in self.devices.items()
            for key, value in getattr(hand, kind).items()
        }

    @property
    def is_connected(self):
        return all(hand.is_connected for hand in self.devices.values())

    def connect(self):
        for hand in self.devices.values():
            hand.connect()

    def disconnect(self):
        errors = []
        for hand in reversed(list(self.devices.values())):
            try:
                hand.disconnect()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Hand disconnect failed", errors)

    def prepare(self, action):
        unknown = {key for key in action if key.startswith("hands.")} - self.features(
            "action_features"
        ).keys()
        if unknown:
            raise ValueError(f"Unknown hand action keys: {sorted(unknown)}")
        prepared = {}
        for side, hand in self.devices.items():
            prefix = f"hands.{side}."
            local = {key[len(prefix) :]: value for key, value in action.items() if key.startswith(prefix)}
            if local:
                prepared[side] = hand.validate_action(local)
        return prepared

    def send(self, prepared):
        result = {}
        for side, action in prepared.items():
            sent = self.devices[side].send_action(action)
            result.update({f"hands.{side}.{key}": value for key, value in sent.items()})
        return result

    def observation(self):
        result = {}
        for side, hand in self.devices.items():
            observation = hand.get_observation()
            if set(observation) != set(hand.observation_features):
                raise ValueError("Hand observation does not match declared features")
            result.update({f"hands.{side}.{key}": value for key, value in observation.items()})
        return result
