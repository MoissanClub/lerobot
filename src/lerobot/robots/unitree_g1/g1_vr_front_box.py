# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Slow arm tracking inside a fixed robot-model workspace."""

import logging
import time
from typing import Any

import numpy as np

from .g1_vr_control import map_input, stop_buttons

WORKSPACE_LOWER = np.array([0.0, -0.5, -0.5])
WORKSPACE_UPPER = np.array([1.0, 0.5, 0.5])


class FrontBoxTracking:
    def __init__(self, ik: Any):
        self.ik = ik
        self.enabled = False
        self.deadline: float | None = None
        self.target: np.ndarray | None = None
        self.last_time: float | None = None
        self.last_stamp = -np.inf
        self.last_warning = -np.inf

    def _lost(self, now: float) -> None:
        if now - self.last_warning >= 10.0:
            logging.info("Lost track. Slow down.")
            self.last_warning = now

    def _inside(self, q: np.ndarray) -> bool:
        return all(
            np.all(w[:3, 3] >= WORKSPACE_LOWER) and np.all(w[:3, 3] <= WORKSPACE_UPPER) for w in self.ik.fk(q)
        )

    def step(self, action: dict, measured: np.ndarray) -> tuple[np.ndarray, bool, bool, bool]:
        now = time.monotonic()
        stop, damp = stop_buttons(action, now=now)
        stop = stop or damp or bool(action.get("control.quit", False))
        if stop or action.get("control.pause", False):
            self.enabled = False
            self.deadline = None
            self.target = measured.copy()
            return self.target, False, stop, damp
        if action.get("control.start", False) and not self.enabled and self.deadline is None:
            self.deadline = now + 5.0
            logging.info("Robot start tracking in 5 second")
        try:
            intent = map_input(action, now=now)
            if action["captured_at"] < self.last_stamp:
                raise ValueError("XR timestamp regressed")
            self.last_stamp = action["captured_at"]
        except (KeyError, ValueError, TypeError):
            if self.enabled or self.deadline is not None:
                self._lost(now)
            # Never resume automatically after tracking is lost while moving.
            if self.enabled:
                self.deadline = None
            self.enabled = False
            self.target = measured.copy()
            return self.target, False, stop, damp
        if not self.enabled:
            self.target = measured.copy()
            if self.deadline is None or now < self.deadline:
                return self.target, False, stop, damp
            self.deadline = None
            if not self._inside(measured):
                logging.warning(
                    "Tracking blocked: starting wrists are outside the front workspace. Press r to retry."
                )
                return self.target, False, stop, damp
            self.wrists = self.ik.fk(measured)
            # Warm up before requesting motor authority; discard the solution.
            self.ik.solve(self.wrists, measured, 0.001)
            self.enabled = True
            self.last_time = now
            return self.target, True, stop, damp
        dt = min(max(now - self.last_time, 0.0), 0.05)
        self.last_time = now
        goals = [w.copy() for w in self.wrists]
        for goal, requested in zip(goals, intent["wrists"], strict=True):
            goal[:3, 3] = np.clip(requested[:3, 3], WORKSPACE_LOWER, WORKSPACE_UPPER)
        actual = self.ik.fk(measured)
        if max(np.linalg.norm(g[:3, 3] - a[:3, 3]) for g, a in zip(goals, actual, strict=True)) > 0.1:
            self._lost(now)
        try:
            candidate, _ = self.ik.solve(goals, measured, max(0.02 * dt, 1e-9))
            candidate = np.clip(candidate, self.target - 0.02 * dt, self.target + 0.02 * dt)
            if not self._inside(candidate):
                safe, outside = self.target.copy(), candidate
                for _ in range(20):
                    middle = (safe + outside) * 0.5
                    if self._inside(middle):
                        safe = middle
                    else:
                        outside = middle
                candidate = safe
            self.target = candidate
        except RuntimeError:
            self._lost(now)
            # Unreachable IK goals hold the last accepted target.
        return self.target.copy(), True, stop, damp
