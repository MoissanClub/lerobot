# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Offline four-state tracking tests; never contact DDS or hardware."""

import logging

import numpy as np
import pytest

from lerobot.robots.unitree_g1 import g1_vr_front_box as module
from lerobot.robots.unitree_g1.g1_vr_control import synthetic_input


class Kinematics:
    def fk(self, q):
        poses = [np.eye(4), np.eye(4)]
        for i, p in enumerate(poses):
            p[0, 3] = 0.5 + q[7 * i]
        return poses

    def solve(self, goals, measured, max_step):
        q = measured.copy()
        q[0], q[7] = goals[0][0, 3] - 0.5, goals[1][0, 3] - 0.5
        return np.clip(q, measured - max_step, measured + max_step), np.zeros(14)


@pytest.fixture
def tracking(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    tracker = module.FrontBoxTracking(Kinematics())
    q = np.zeros(14)
    sample = synthetic_input(tracker.ik.fk(q))
    sample.update({"left.tracked": False, "right.tracked": False})
    return tracker, now, q, sample


def tick(tracking, seconds=0.05, **changes):
    tracker, now, q, sample = tracking
    now[0] += seconds
    sample.update(changes)
    sample["captured_at"] = now[0]
    return tracker.step(sample, q)


def reach_ready(tracking):
    tracker, _, q, _ = tracking
    assert not tick(tracking, 0, **{"control.start": True})[1]
    assert not tick(tracking, 4.99, **{"control.start": False})[1]
    target, enabled, stop, _ = tick(tracking, 0.01)
    assert enabled and not stop
    np.testing.assert_array_equal(target, q)
    for _ in range(1010):
        target, enabled, stop, _ = tick(tracking)
        assert enabled and not stop
        assert np.max(np.abs(target - q)) <= 0.001 + 1e-9
        q[:] = target
    assert tracker.phase == "unsynced"
    np.testing.assert_allclose(q, module.READY_ARM_Q)


def near(tracking):
    tracker, _, q, sample = tracking
    sample.update(synthetic_input(tracker.ik.fk(q)))


def test_preparation_reaches_zero_before_ready_without_vr(tracking):
    tracker, _, q, _ = tracking
    q[3] = 0.1
    tick(tracking, 0, **{"control.start": True})
    tick(tracking, 5, **{"control.start": False})
    assert not tracker.homed
    np.testing.assert_array_equal(tracker.motion_goal, module.LOWERED_ARM_Q)
    for _ in range(150):
        target, enabled, stop, _ = tick(tracking)
        assert enabled and not stop
        if tracker.homed:
            assert np.max(np.abs(q)) <= 0.02
            np.testing.assert_array_equal(tracker.motion_goal, module.READY_ARM_Q)
            break
        q[:] = target
    else:
        pytest.fail("Initial zero-pose stage did not complete")


def sync(tracking):
    tracker, _, q, _ = tracking
    near(tracking)
    before = q.copy()
    tick(tracking)
    assert tracker.phase == "unsynced"
    tick(tracking, 2.99)
    assert tracker.phase == "unsynced"
    tick(tracking, 0.01)
    assert tracker.phase == "synced"
    np.testing.assert_array_equal(tracker.target, before)


def test_raise_without_controllers_then_three_second_pickup(tracking, caplog):
    caplog.set_level(logging.INFO)
    tracker, _, q, _ = tracking
    reach_ready(tracking)
    assert "Robot arms will rise" in caplog.text
    tick(tracking)
    assert "VR controllers unavailable" in caplog.text
    np.testing.assert_array_equal(tracker.target, q)
    sync(tracking)
    assert "Hold and start tracking in 3, 2, 1" in caplog.text


@pytest.mark.parametrize("lost", [False, True])
def test_pickup_requires_continuously_close_tracked_hands(tracking, lost):
    tracker, _, _, sample = tracking
    reach_ready(tracking)
    near(tracking)
    tick(tracking)
    if lost:
        sample["right.tracked"] = False
    else:
        sample["right.grip_pos"][2] -= 0.2
    tick(tracking, 2)
    assert tracker.sync_since is None
    near(tracking)
    tick(tracking, 2)
    assert tracker.phase == "unsynced"
    tick(tracking, 3)
    assert tracker.phase == "synced"


@pytest.mark.parametrize("lost", [False, True])
def test_lost_sync_holds_and_can_resync_without_r(tracking, lost, caplog):
    caplog.set_level(logging.INFO)
    tracker, _, q, sample = tracking
    reach_ready(tracking)
    sync(tracking)
    if lost:
        sample["left.tracked"] = False
    else:
        sample["left.grip_pos"][2] -= 0.2
    target, enabled, stop, _ = tick(tracking)
    assert enabled and not stop and tracker.phase == "unsynced"
    np.testing.assert_array_equal(target, q)
    assert "Lost track." in caplog.text
    sync(tracking)


def test_synced_small_motion_is_slow(tracking):
    tracker, _, q, sample = tracking
    reach_ready(tracking)
    sync(tracking)
    sample["left.grip_pos"][2] -= 0.05
    for _ in range(20):
        target, enabled, _, _ = tick(tracking)
        assert enabled and tracker.phase == "synced"
        assert 0 < target[0] - q[0] <= 0.001 + 1e-9
        q[:] = target


@pytest.mark.parametrize("phase", ["initial", "raising", "unsynced", "synced"])
def test_q_lowers_without_controllers_before_exit(tracking, phase, caplog):
    caplog.set_level(logging.INFO)
    tracker, _, q, sample = tracking
    if phase in ("unsynced", "synced"):
        reach_ready(tracking)
        if phase == "synced":
            sync(tracking)
    elif phase == "raising":
        tick(tracking, 0, **{"control.start": True})
        tick(tracking, 5, **{"control.start": False})
        for _ in range(200):
            q[:] = tick(tracking)[0]
    sample.update({"left.tracked": False, "right.tracked": False})
    assert not tick(tracking, 0, **{"control.quit": True})[2]
    assert tracker.phase == "exit"
    assert "Lowering arms and quitting" in caplog.text
    # Repeated q/terminal EOF must not restart the lowering countdown.
    for _ in range(1200):
        target, _, stop, _ = tick(tracking, **{"control.quit": True})
        assert np.max(np.abs(target - q)) <= 0.001 + 1e-9
        q[:] = target
        if stop:
            break
    assert stop
    np.testing.assert_allclose(q, module.LOWERED_ARM_Q, atol=1e-9)


def test_exit_waits_for_measured_lowered_pose(tracking):
    tracker, _, q, _ = tracking
    reach_ready(tracking)
    tick(tracking, 0, **{"control.quit": True})
    for _ in range(1200):
        assert not tick(tracking)[2]  # No measured motion, despite target reaching zero.
    q[:] = module.LOWERED_ARM_Q
    assert tick(tracking)[2]


def test_pause_cancels_sync_countdown(tracking):
    tracker, _, _, _ = tracking
    reach_ready(tracking)
    near(tracking)
    tick(tracking)
    tick(tracking, **{"control.pause": True})
    tick(tracking, 5, **{"control.pause": False})
    assert tracker.phase == "unsynced" and tracker.sync_since is None
    tick(tracking, **{"control.start": True})
    assert tracker.phase == "unsynced"


def test_emergency_button_bypasses_lowering(tracking):
    reach_ready(tracking)
    target, enabled, stop, damp = tick(tracking, **{"right.primary": True})
    assert stop and not enabled and not damp


def test_initial_q_when_already_lowered_never_enables_motors(tracking):
    tick(tracking, 0, **{"control.quit": True})
    _, enabled, stop, _ = tick(tracking, 1)
    assert stop and not enabled


def test_processor_delays_stop_until_lowered_without_vr(monkeypatch):
    from lerobot.lerobot_types import TransitionKey
    from lerobot.robots.unitree_g1 import g1_vr_processor
    from lerobot.robots.unitree_g1.g1_vr_control import ARM_KEYS

    monkeypatch.setattr(g1_vr_processor, "G1VRKinematics", lambda _: Kinematics())
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    processor = g1_vr_processor.G1VRActionProcessor("unused", arm_test=True, workspace="front_box")
    q = module.READY_ARM_Q.copy()
    processor._current_transition = {TransitionKey.OBSERVATION: dict(zip(ARM_KEYS, q, strict=True))}
    action = processor.action({"control.quit": True, "input.live": True})
    assert not action["control.stop"] and not action["control.following"]
    now[0] += 1
    for _ in range(1700):
        now[0] += 0.05
        action = processor.action({"input.live": True})
        assert not action["control.following"]
        if action["control.stop"]:
            break
        q = np.array([action[k] for k in ARM_KEYS])
        processor._current_transition = {TransitionKey.OBSERVATION: dict(zip(ARM_KEYS, q, strict=True))}
    assert action["control.stop"]
    np.testing.assert_allclose(q, module.LOWERED_ARM_Q, atol=0.02)


@pytest.mark.parametrize("distance", [0.1, 0.2])
@pytest.mark.parametrize("side", ["left", "right"])
def test_sync_hysteresis_uses_same_distance_on_entry_and_exit(tracking, distance, side):
    old, now, q, sample = tracking
    tracker = module.FrontBoxTracking(old.ik, sync_distance_m=distance)
    tracking = tracker, now, q, sample
    tracker.enabled = True
    tracker._unsync(q)
    base = synthetic_input(tracker.ik.fk(q))

    def at_gap(gap):
        sample.update(base)
        sample[f"{side}.grip_pos"] = base[f"{side}.grip_pos"].copy()
        sample[f"{side}.grip_pos"][2] -= gap

    at_gap(0.99 * distance)
    tick(tracking)
    tick(tracking, 3)
    assert tracker.phase == "synced"
    # This also verifies the pickup offset cannot hide a large raw hand gap.
    at_gap(1.1 * distance)
    for _ in range(20):
        tick(tracking)
        assert tracker.phase == "synced"
    at_gap(1.21 * distance)
    tick(tracking)
    assert tracker.phase == "unsynced"
    at_gap(1.1 * distance)
    tick(tracking, 4)
    assert tracker.phase == "unsynced" and tracker.sync_since is None
    at_gap(0.99 * distance)
    tick(tracking)
    tick(tracking, 3)
    assert tracker.phase == "synced"


@pytest.mark.parametrize("distance", [0, -0.1, float("nan"), float("inf")])
def test_sync_distance_must_be_positive_and_finite(distance):
    with pytest.raises(ValueError, match="positive and finite"):
        module.FrontBoxTracking(Kinematics(), sync_distance_m=distance)


def test_preparation_allows_small_rear_offset_but_tracking_box_does_not(tracking):
    tracker, _, q, _ = tracking
    q[0] = -0.508  # Fake FK: x=-8 mm, matching the reported real-model failure.
    assert not tracker._inside(q)
    assert tracker._inside(q, preparation=True)
    tick(tracking, 0, **{"control.start": True})
    assert tick(tracking, 5, **{"control.start": False})[1]
    for _ in range(1700):
        target, enabled, stop, _ = tick(tracking)
        assert enabled and not stop and tracker._inside(target, preparation=True)
        q[:] = target
    assert tracker.phase == "unsynced"


def test_preparation_rejects_pose_beyond_rear_margin(tracking):
    tracker, _, q, _ = tracking
    q[0] = -0.53
    tick(tracking, 0, **{"control.start": True})
    with pytest.raises(RuntimeError, match="Cartesian workspace"):
        tick(tracking, 5, **{"control.start": False})
    assert not tracker.enabled
