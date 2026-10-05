# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Offline countdown, workspace and synchronization tests; no DDS."""

import logging

import numpy as np
import pytest

from lerobot.robots.unitree_g1 import g1_vr_front_box as module
from lerobot.robots.unitree_g1.g1_vr_control import synthetic_input


class Kinematics:
    def fk(self, q):
        poses = [np.eye(4), np.eye(4)]
        for i, p in enumerate(poses):
            p[0, 3] = 0.2 + q[7 * i]
        return poses

    def solve(self, goals, measured, max_step):
        q = measured.copy()
        q[0], q[7] = goals[0][0, 3] - 0.2, goals[1][0, 3] - 0.2
        return np.clip(q, measured - max_step, measured + max_step), np.zeros(14)


@pytest.fixture
def tracking(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    tracker = module.FrontBoxTracking(Kinematics())
    q = np.zeros(14)
    sample = synthetic_input(tracker.ik.fk(q))
    return tracker, now, q, sample


def tick(tracking, seconds, **changes):
    tracker, now, q, sample = tracking
    now[0] += seconds
    sample.update(changes)
    sample["captured_at"] = now[0]
    return tracker.step(sample, q)


def test_countdown_no_jump_slow_motion_boundary_and_lag(tracking, caplog):
    caplog.set_level(logging.INFO)
    tracker, now, q, sample = tracking
    assert not tick(tracking, 0, **{"control.start": True})[1]
    assert "Robot start tracking in 5 second" in caplog.text
    assert not tick(tracking, 4.99, **{"control.start": False})[1]
    target, enabled, _, _ = tick(tracking, 0.01)
    assert enabled
    np.testing.assert_array_equal(target, q)
    original = sample["left.grip_pos"].copy()
    sample["left.grip_pos"][2] -= 2.0
    for _ in range(900):
        target, enabled, _, _ = tick(tracking, 0.05)
        assert enabled
        assert np.max(np.abs(target - q)) <= 0.001 + 1e-9
        assert tracker._inside(target)
        q[:] = target
    assert target[0] == pytest.approx(0.8)  # x=1 meter boundary
    assert target[7] == 0  # untouched right arm
    assert "Lost track. Slow down." in caplog.text
    sample["left.grip_pos"] = original
    target, enabled, _, _ = tick(tracking, 0.05)
    assert enabled and target[0] < q[0]


@pytest.mark.parametrize("when", [2.0, 5.0])
def test_pause_cancels_countdown_or_following(tracking, when):
    tick(tracking, 0, **{"control.start": True})
    tick(tracking, when, **{"control.start": False})
    assert not tick(tracking, 0, **{"control.pause": True})[1]
    assert not tick(tracking, 10, **{"control.pause": False})[1]


def test_tracking_loss_never_resumes_without_new_start(tracking):
    tick(tracking, 0, **{"control.start": True})
    assert tick(tracking, 5, **{"control.start": False})[1]
    assert not tick(tracking, 0.05, **{"left.tracked": False})[1]
    assert not tick(tracking, 0.05, **{"left.tracked": True})[1]


def test_robot_catches_up_to_hand_already_far_at_start(tracking):
    tracker, now, q, sample = tracking
    sample["left.grip_pos"][2] -= 0.3
    assert not tick(tracking, 0, **{"control.start": True})[1]
    target, enabled, _, _ = tick(tracking, 5, **{"control.start": False})
    assert enabled
    np.testing.assert_array_equal(target, q)
    # Controller stays still: the robot must still approach it, not re-anchor.
    for _ in range(10):
        target, enabled, _, _ = tick(tracking, 0.05)
        assert enabled and 0 < target[0] - q[0] <= 0.001 + 1e-9
        q[:] = target
    assert target[0] == pytest.approx(0.01)
    assert target[7] == 0
