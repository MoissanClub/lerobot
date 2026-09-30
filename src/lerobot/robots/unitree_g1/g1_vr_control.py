# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""G1-29 VR intent mapping and an adapter around upstream G1_29_ArmIK.

Reference: unitreerobotics/xr_teleoperate 817fb00c, televuer 766de45e.
No SDK transport, controller authority or physical side effects in this module.
"""

import copy
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .g1_utils import G1_29_JointArmIndex

ARM_KEYS = tuple(f"{j.name}.q" for j in G1_29_JointArmIndex)
BASIS = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
MODEL_REVISION = "68459ed68f6f68e1f661091dfcb6ebce44681aec"
POLICY_REVISION = "7bb8a672f5a4213c9261ea5ac1f3f034f5078638"


def stop_buttons(sample: dict, *, now: float | None = None, max_age: float = 0.25) -> tuple[bool, bool]:
    """Fresh stop buttons remain usable when pose tracking is invalid."""
    now = time.monotonic() if now is None else now
    stamp = float(sample.get("captured_at", float("nan")))
    if not np.isfinite(stamp) or not 0 <= now - stamp <= max_age:
        return False, False
    return (
        bool(sample.get("right.primary", False)),
        bool(sample.get("left.stick_click", False) and sample.get("right.stick_click", False)),
    )


def pose(position, quaternion) -> np.ndarray:
    p, q = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid tracked pose")
    out = np.eye(4)
    out[:3, :3] = BASIS @ Rotation.from_quat(q).as_matrix() @ BASIS.T
    out[:3, 3] = BASIS @ p
    return out


def map_input(sample: dict, *, now: float | None = None, max_age: float = 0.25) -> dict:
    """Raw OpenXR poses/sticks -> head-yaw-relative wrists and base velocity.

    OpenXR stick +Y is forward; Unitree's WebXR input uses the opposite Y sign.
    Returned hand values are intent only, in reference slot order, not hardware units.
    """
    now = time.monotonic() if now is None else now
    stamp = float(sample["captured_at"])
    if not np.isfinite(stamp) or not 0 <= now - stamp <= max_age:
        raise ValueError("Stale/future XR input")
    if not all(sample.get(f"{s}.tracked", False) for s in ("head", "left", "right")):
        raise ValueError("Head or controller tracking lost")
    head = pose(sample["head.pos"], sample["head.quat"])
    forward = head[:3, 0].copy()
    forward[2] = 0
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("Head yaw is undefined")
    forward /= np.linalg.norm(forward)
    yaw = np.column_stack((forward, np.cross([0.0, 0.0, 1.0], forward), [0.0, 0.0, 1.0]))
    wrists, hands = [], []
    for side in ("left", "right"):
        wrist = pose(sample[f"{side}.grip_pos"], sample[f"{side}.grip_quat"])
        wrist[:3, 3] = yaw.T @ (wrist[:3, 3] - head[:3, 3]) + [0.15, 0, 0.45]
        wrist[:3, :3] = yaw.T @ wrist[:3, :3]
        wrists.append(wrist)
        trigger, squeeze = [float(sample[f"{side}.{k}"]) for k in ("trigger", "squeeze")]
        if (
            not np.isfinite([trigger, squeeze]).all()
            or not 0 <= min(trigger, squeeze) <= max(trigger, squeeze) <= 1
        ):
            raise ValueError("Invalid finger input")
        hands.append(
            [
                float(np.clip((trigger - 0.5) / 0.5, 0, 0.98)),
                float(np.clip(trigger / 0.5, 0, 0.7)),
                min(squeeze, 0.98),
                min(trigger, 0.98),
                min(trigger, 0.98),
                min(trigger, 0.98),
            ]
        )
    sticks = np.array(
        [sample[f"{side}.stick_{axis}"] for side, axis in (("left", "x"), ("left", "y"), ("right", "x"))],
        dtype=float,
    )
    if not np.isfinite(sticks).all() or np.any(abs(sticks) > 1):
        raise ValueError("Invalid joystick input")
    damp = bool(sample["left.stick_click"] and sample["right.stick_click"])
    quit_requested = bool(sample["right.primary"])
    velocity = np.array([sticks[1], -sticks[0], -sticks[2]]) * 0.3
    if damp or quit_requested:
        velocity[:] = 0
    return {"wrists": wrists, "velocity": velocity, "hands": hands, "damp": damp, "quit": quit_requested}


class G1VRKinematics:
    """Reuse the upstream optimizer; make its Pinocchio/motor-order boundary explicit."""

    def __init__(self, assets: str | Path):
        from .g1_kinematics import G1_29_ArmIK

        self.assets = Path(assets)
        self.solver = G1_29_ArmIK(assets_path=self.assets)
        self.model = self.solver.reduced_robot.model
        self.pin = self.solver._pin
        self.data = self.model.createData()
        self.to_pin = self.solver._arm_reorder_g1_to_pin
        self.to_motor = self.solver._arm_reorder_pin_to_g1
        self.lower = self.model.lowerPositionLimit[self.to_motor].copy()
        self.upper = self.model.upperPositionLimit[self.to_motor].copy()

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, ...]:
        self.pin.framesForwardKinematics(self.model, self.data, np.asarray(q)[self.to_pin])
        return tuple(
            self.data.oMf[i].homogeneous.copy() for i in (self.solver.L_hand_id, self.solver.R_hand_id)
        )

    def gravity(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.pin.rnea(self.model, self.data, q[self.to_pin], np.zeros(14), np.zeros(14))[
            self.to_motor
        ].copy()

    def solve(self, wrists, measured: np.ndarray, max_step: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(max_step) or max_step <= 0:
            raise ValueError("IK step bound must be positive and finite")
        if len(wrists) != 2 or any(np.asarray(w).shape != (4, 4) or not np.isfinite(w).all() for w in wrists):
            raise ValueError("IK requires two finite homogeneous wrist poses")
        q = np.asarray(measured, dtype=float)
        if q.shape != (14,) or not np.isfinite(q).all() or np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("Invalid measured arm seed")
        previous_filter = copy.deepcopy(self.solver.smooth_filter)
        candidate, _ = self.solver.solve_ik(*wrists, q[self.to_pin])
        if not self.solver.opti.stats().get("success", False):
            self.solver.smooth_filter = previous_filter
            raise RuntimeError("Upstream IK did not converge")
        candidate = np.asarray(candidate, dtype=float)[self.to_motor]
        if not np.isfinite(candidate).all():
            raise RuntimeError("Nonfinite upstream IK result")
        target = np.clip(q + np.clip(candidate - q, -max_step, max_step), self.lower, self.upper)
        return target, self.gravity(target)


def synthetic_input(wrists, step: int = 0, velocity=(0.0, 0.0, 0.0)) -> dict:
    """Invert the mapping for deterministic simulation replay, never hardware input."""
    sample = {
        "captured_at": time.monotonic(),
        "head.tracked": True,
        "head.pos": [0.0, 1.5, 0.0],
        "head.quat": [0.0, 0.0, 0.0, 1.0],
    }
    for side, wrist in zip(("left", "right"), wrists, strict=True):
        p = wrist[:3, 3] - [0.15, 0, 0.45]
        p[2] += 0.015 * np.sin(step / 30)
        sample.update(
            {
                f"{side}.tracked": True,
                f"{side}.grip_pos": BASIS.T @ p + sample["head.pos"],
                f"{side}.grip_quat": Rotation.from_matrix(BASIS.T @ wrist[:3, :3] @ BASIS).as_quat(),
                f"{side}.trigger": 0.0,
                f"{side}.squeeze": 0.0,
                f"{side}.stick_x": 0.0,
                f"{side}.stick_y": 0.0,
                f"{side}.stick_click": False,
                f"{side}.primary": False,
                f"{side}.secondary": False,
            }
        )
    sample["left.stick_y"], sample["left.stick_x"], sample["right.stick_x"] = (
        velocity[0] / 0.3,
        -velocity[1] / 0.3,
        -velocity[2] / 0.3,
    )
    return sample
