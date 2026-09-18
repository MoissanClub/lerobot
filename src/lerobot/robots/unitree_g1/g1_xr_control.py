# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Independent dual-arm clutching over shared bounded G1 kinematics."""

import time

import numpy as np

from lerobot.processor import TransitionKey
from lerobot.teleoperators.xr_controllers.xr_controllers import controller_pose

from .g1_action_processor import G1CartesianActionProcessor


class G1XRControl:
    def __init__(self, kinematics, *, threshold=0.5, max_age=0.25, translation_scale=1.0):
        if not (np.isfinite(threshold) and 0 < threshold <= 1):
            raise ValueError("threshold must be in (0, 1]")
        if not all(np.isfinite(x) and x > 0 for x in (max_age, translation_scale)):
            raise ValueError("max_age and translation_scale must be positive finite")
        self.ik = kinematics
        self.processor = G1CartesianActionProcessor(
            embodiment=getattr(getattr(kinematics, "embodiment", None), "name", "g1_29")
        )
        self.processor._kinematics = kinematics
        self.threshold, self.max_age, self.scale = threshold, max_age, translation_scale
        self.reset()

    def reset(self):
        self.origins = [None, None]
        self.targets = None
        self.last_stamp = -np.inf
        self.last_result = None

    @property
    def engaged(self):
        return tuple(origin is not None for origin in self.origins)

    def action(self, sample, measured, *, now=None):
        now = time.monotonic() if now is None else now
        current = self.ik.fk(measured)
        if self.targets is None:
            self.targets = [pose.copy() for pose in current]
        try:
            stamp = float(sample["captured_at"])
            fresh = (
                np.isfinite(stamp)
                and np.isfinite(now)
                and 0 <= now - stamp <= self.max_age
                and stamp >= self.last_stamp
            )
        except (KeyError, TypeError, ValueError):
            fresh = False
        if fresh:
            self.last_stamp = stamp
        for i, side in enumerate(("left", "right")):
            pose = controller_pose(sample, side) if fresh else None
            if pose is None or sample[f"{side}.squeeze"] < self.threshold:
                self.origins[i] = None
                self.targets[i] = current[i].copy()
                continue
            if self.origins[i] is None:
                self.origins[i] = (pose.copy(), current[i].copy())
            origin, robot_origin = self.origins[i]
            self.targets[i][:3, 3] = robot_origin[:3, 3] + self.scale * (pose[:3, 3] - origin[:3, 3])
            self.targets[i][:3, :3] = pose[:3, :3] @ origin[:3, :3].T @ robot_origin[:3, :3]
        if not any(self.engaged):
            return self.ik.arm_action(measured)
        # The processor is also usable outside XR (recording, policy or replay).
        observation = self.ik.arm_action(measured)
        self.processor(
            {
                TransitionKey.ACTION: {"left.ee_pose": self.targets[0], "right.ee_pose": self.targets[1]},
                TransitionKey.OBSERVATION: observation,
            }
        )
        result = self.processor.last_result
        self.last_result = result
        q = result.q.copy()
        half = self.ik.size // 2
        # An inactive hand must not move due to optimizer regularization of its pose.
        for i, active in enumerate(self.engaged):
            if not active:
                q[i * half : (i + 1) * half] = measured[i * half : (i + 1) * half]
        return self.ik.arm_action(q)
