# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Initial -> unsynced -> synced -> exit, with robot-only preparation/return."""

import logging
import time
from typing import Any

import numpy as np

from .g1_vr_control import map_input, stop_buttons

# Shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw, left then right.
READY_ARM_Q = np.array([-0.3, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0, -0.3, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0])
LOWERED_ARM_Q = np.zeros(14)
WORKSPACE_LOWER = np.array([0.0, -0.5, -0.5])
WORKSPACE_UPPER = np.array([1.0, 0.5, 0.5])
PREPARATION_LOWER = np.array([-0.02, -0.5, -0.5])


class FrontBoxTracking:
    def __init__(self, ik: Any, sync_distance_m: float = 0.1):
        if not np.isfinite(sync_distance_m) or sync_distance_m <= 0:
            raise ValueError("sync_distance_m must be positive and finite")
        self.ik = ik
        self.sync_distance_m = sync_distance_m
        self.unsync_distance_m = 1.2 * sync_distance_m
        self.phase = "initial"
        self.enabled = False
        self.paused = False
        self.target: np.ndarray | None = None
        self.last_time: float | None = None
        self.last_stamp = -np.inf
        self.last_prompt = -np.inf
        self.sync_since: float | None = None
        self.motion_goal: np.ndarray | None = None
        self.motion_origin: np.ndarray | None = None
        self.motion_after = 0.0
        self.motion_progress = 0.0
        self.ready_reached = False
        self.homed = False

    def _inside(self, q: np.ndarray, *, preparation: bool = False) -> bool:
        lower = PREPARATION_LOWER if preparation else WORKSPACE_LOWER
        return all(np.all(w[:3, 3] >= lower) and np.all(w[:3, 3] <= WORKSPACE_UPPER) for w in self.ik.fk(q))

    def _begin_motion(self, goal: np.ndarray, now: float, delay: float) -> None:
        self.motion_goal = goal.copy()
        self.motion_origin = None
        self.motion_after = now + delay
        self.motion_progress = 0.0
        self.sync_since = None
        self.paused = False

    def _move(self, measured: np.ndarray, now: float, dt: float) -> bool:
        """Robot-feedback-only trajectory; return true after measured arrival."""
        if now < self.motion_after:
            return False
        if self.motion_origin is None:
            if self.phase == "exit" and np.max(np.abs(measured - self.motion_goal)) <= 0.02:
                self.target = measured.copy()
                return True
            if any(
                not self._inside(q, preparation=True) for q in np.linspace(measured, self.motion_goal, 101)
            ):
                raise RuntimeError(
                    "Preparation/return path leaves the Cartesian workspace; no new motion started"
                )
            self.motion_origin = measured.copy()
            self.target = measured.copy()
            self.enabled = True
            # First command holds feedback, including when opening the publisher.
            return False
        delta = self.motion_goal - self.motion_origin
        self.motion_progress = min(
            1.0, self.motion_progress + 0.02 * dt / max(float(np.max(np.abs(delta))), 1e-9)
        )
        self.target = self.motion_origin + self.motion_progress * delta
        return self.motion_progress == 1.0 and np.max(np.abs(measured - self.motion_goal)) <= 0.02

    def _unsync(self, measured: np.ndarray, *, lost: bool = False) -> None:
        self.phase = "unsynced"
        self.target = measured.copy()
        self.wrists = self.ik.fk(measured)
        self.sync_since = None
        self.last_prompt = -np.inf
        if lost:
            logging.info("Lost track.")

    def _guidance(self, differences: list[np.ndarray]) -> None:
        directions = (("forward", "backward"), ("left", "right"), ("up", "down"))
        lines = []
        for hand, delta in zip(("Left", "Right"), differences, strict=True):
            if np.linalg.norm(delta) <= self.sync_distance_m:
                lines.append(f"{hand} hand: READY, keep still")
            else:
                moves = [
                    f"{directions[i][0 if delta[i] > 0 else 1]} {abs(delta[i]) * 100:.1f} cm"
                    for i in np.argsort(-np.abs(delta))
                    if abs(delta[i]) >= 0.005
                ]
                lines.append(f"{hand} hand: move " + ", ".join(moves))
        logging.info("Waiting for VR pickup\n  %s\n  %s", *lines)

    def step(self, action: dict, measured: np.ndarray) -> tuple[np.ndarray, bool, bool, bool]:
        now = time.monotonic()
        dt = 0.0 if self.last_time is None else min(max(now - self.last_time, 0.0), 0.05)
        self.last_time = now
        if self.target is None:
            self.target = measured.copy()
        stop, damp = stop_buttons(action, now=now)
        if stop or damp:  # Emergency controller buttons bypass graceful lowering.
            return measured.copy(), False, True, damp
        if action.get("control.quit", False) and (self.phase != "exit" or self.paused):
            logging.info("Lowering arms and quitting. Keep clear.")
            self.phase = "exit"
            self.target = measured.copy()
            self._begin_motion(LOWERED_ARM_Q, now, 1.0)
        elif action.get("control.pause", False):
            self.target = measured.copy()
            self.paused = True
            self.sync_since = None
            self.motion_goal = None
            if self.phase == "synced":
                self._unsync(measured)
            logging.info("Arm motion paused. Press r to resume, or q to lower and quit.")
        elif action.get("control.start", False) and self.phase != "exit":
            if self.phase == "initial" and (self.motion_goal is None or self.paused):
                logging.info("Robot arms will rise to the ready position in five seconds. Keep clear.")
                self._begin_motion(READY_ARM_Q if self.homed else LOWERED_ARM_Q, now, 5.0)
            elif self.paused:
                self.paused = False
                self._unsync(measured)
        if self.paused:
            return self.target.copy(), self.enabled, False, False
        # These states do not inspect VR poses. Only robot feedback is required.
        if self.phase == "exit":
            done = self._move(measured, now, dt)
            if done:
                logging.info("Arms lowered. Exiting.")
            return self.target.copy(), self.enabled, done, False
        if self.phase == "initial":
            if self.motion_goal is not None and self._move(measured, now, dt):
                if not self.homed:
                    self.homed = True
                    logging.info("Zero arm posture reached. Raising to the ready position.")
                    self._begin_motion(READY_ARM_Q, now, 0.0)
                    return self.target.copy(), self.enabled, False, False
                self.ready_reached = True
                self.motion_goal = None
                self._unsync(measured)
                logging.info("Ready pose held. Waiting for VR controllers.")
            return self.target.copy(), self.enabled, False, False
        try:
            intent = map_input(action, now=now)
            if action["captured_at"] < self.last_stamp:
                raise ValueError("XR timestamp regressed")
            self.last_stamp = action["captured_at"]
        except (KeyError, ValueError, TypeError):
            if self.phase == "synced":
                self._unsync(measured, lost=True)
            self.sync_since = None
            if now - self.last_prompt >= 10.0:
                logging.info("VR controllers unavailable. Connect, press Play, and wake both controllers.")
                self.last_prompt = now
            return self.target.copy(), self.enabled, False, False
        if self.phase == "unsynced":
            actual = self.ik.fk(measured)
            differences = [w[:3, 3] - r[:3, 3] for w, r in zip(actual, intent["wrists"], strict=True)]
            if max(np.linalg.norm(d) for d in differences) <= self.sync_distance_m:
                if self.sync_since is None:
                    self.sync_since = now
                    logging.info("Hold and start tracking in 3, 2, 1.")
                elif now - self.sync_since >= 3.0:
                    self.offsets = differences
                    self.target = measured.copy()
                    self.wrists = actual
                    self.phase = "synced"
                    logging.info("VR synchronized. Robot starts to follow.")
            else:
                self.sync_since = None
                if now - self.last_prompt >= 10.0:
                    self._guidance(differences)
                    self.last_prompt = now
            return self.target.copy(), self.enabled, False, False
        goals = [w.copy() for w in self.wrists]
        requested = [r[:3, 3] + offset for r, offset in zip(intent["wrists"], self.offsets, strict=True)]
        actual = self.ik.fk(measured)
        # Entry and exit use the same raw mapped VR-to-measured-wrist metric.
        # Neither pickup offsets nor clipping may conceal an excessive gap.
        if (
            max(np.linalg.norm(r[:3, 3] - a[:3, 3]) for r, a in zip(intent["wrists"], actual, strict=True))
            > self.unsync_distance_m
        ):
            self._unsync(measured, lost=True)
            return self.target.copy(), self.enabled, False, False
        for goal, position in zip(goals, requested, strict=True):
            goal[:3, 3] = np.clip(position, WORKSPACE_LOWER, WORKSPACE_UPPER)
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
            self._unsync(measured, lost=True)
        return self.target.copy(), self.enabled, False, False
