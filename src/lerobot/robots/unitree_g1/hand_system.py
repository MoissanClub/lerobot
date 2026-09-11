# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Vendor-independent optional hand contract. No XR or SDK imports."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import draccus

from lerobot.utils.import_utils import make_device_from_device_class


@dataclass(kw_only=True)
class HandConfig(draccus.ChoiceRegistry):
    side: str

    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError("Hand side must be left or right")


class HandSystem(ABC):
    """Hand-local features are prefixed by the owner, not the hand implementation.

    validate_action must be side-effect free. It returns the actual bounded action
    accepted for dispatch. Read failures must raise, never return fabricated data.
    """

    @property
    @abstractmethod
    def action_features(self) -> dict: ...

    @property
    @abstractmethod
    def observation_features(self) -> dict: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def validate_action(self, action: dict) -> dict: ...

    @abstractmethod
    def send_action(self, action: dict) -> dict: ...

    @abstractmethod
    def get_observation(self) -> dict: ...


def make_hand(config: HandConfig) -> HandSystem:
    hand = make_device_from_device_class(config)
    if not isinstance(hand, HandSystem):
        raise TypeError("Hand implementation must implement HandSystem")
    return hand
