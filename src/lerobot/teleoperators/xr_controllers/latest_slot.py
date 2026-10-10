# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Fixed-layout shared-memory storage for the latest XR controller sample."""

from typing import Literal

import numpy as np

_LOCK_TIMEOUT_S = 0.001
FieldKind = Literal["bool", "float", "array"]
Field = tuple[str, int, FieldKind]


def _fields(full_input: bool) -> tuple[Field, ...]:
    fields: list[Field] = [("captured_at", 1, "float")]
    for side in ("left", "right"):
        fields.extend(
            (
                (f"{side}.tracked", 1, "bool"),
                (f"{side}.grip_pos", 3, "array"),
                (f"{side}.grip_quat", 4, "array"),
                (f"{side}.squeeze", 1, "float"),
                (f"{side}.trigger", 1, "float"),
            )
        )
        if full_input:
            fields.extend(
                (
                    (f"{side}.stick_x", 1, "float"),
                    (f"{side}.stick_y", 1, "float"),
                    (f"{side}.stick_click", 1, "bool"),
                    (f"{side}.primary", 1, "bool"),
                    (f"{side}.secondary", 1, "bool"),
                )
            )
    if full_input:
        fields.extend(
            (
                ("head.tracked", 1, "bool"),
                ("head.pos", 3, "array"),
                ("head.quat", 4, "array"),
            )
        )
    return tuple(fields)


class LatestSlot:
    """One lock-protected shared-memory controller sample.

    The lock is held only while copying a fixed-size array (21 or 39 doubles),
    avoiding queue serialization and feeder-thread delays.
    """

    def __init__(self, context, full_input: bool):
        self.full_input = full_input
        self.fields = _fields(full_input)
        self.size = sum(width for _, width, _ in self.fields)
        self.values = context.Array("d", self.size, lock=False)
        self.sequence = context.Value("Q", 0, lock=False)
        self.lock = context.Lock()

    def write(self, sample: dict) -> bool:
        encoded: np.ndarray = np.empty(self.size, dtype=np.float64)
        offset = 0
        for key, width, _ in self.fields:
            value = np.asarray(sample[key], dtype=np.float64)
            if value.size != width:
                raise ValueError(f"{key} must contain {width} value(s)")
            encoded[offset : offset + width] = value.reshape(-1)
            offset += width
        if not self.lock.acquire(timeout=_LOCK_TIMEOUT_S):
            return False
        try:
            np.frombuffer(self.values, dtype=np.float64)[:] = encoded
            self.sequence.value += 1
        finally:
            self.lock.release()
        return True

    def read(self) -> dict | None:
        if not self.lock.acquire(timeout=_LOCK_TIMEOUT_S):
            return None
        try:
            if self.sequence.value == 0:
                return None
            encoded = np.frombuffer(self.values, dtype=np.float64).copy()
        finally:
            self.lock.release()
        sample: dict[str, bool | float | np.ndarray] = {}
        offset = 0
        for key, width, kind in self.fields:
            values = encoded[offset : offset + width]
            if kind == "bool":
                sample[key] = bool(values[0])
            elif kind == "float":
                sample[key] = float(values[0])
            else:
                sample[key] = values.copy()
            offset += width
        return sample
