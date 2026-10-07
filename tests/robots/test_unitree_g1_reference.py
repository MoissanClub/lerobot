# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Command-level reference checks. No DDS or hardware is opened."""

import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from lerobot.lerobot_types import TransitionKey
from lerobot.robots.unitree_g1 import g1_vr_processor
from lerobot.robots.unitree_g1.g1_arm_sdk import ARM_KEYS, ARM_SLOTS, G1ArmSDKConfig
from lerobot.robots.unitree_g1.g1_arm_unitree import UnitreeArmSDK
from lerobot.robots.unitree_g1.g1_vr_control import synthetic_input


def make_backend():
    state = SimpleNamespace(
        motor_state=[SimpleNamespace(q=0.0, dq=0.0, tau_est=0.0) for _ in range(35)],
        imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        mode_machine=5,
        mode_pr=0,
        tick=1,
    )
    for slot in ARM_SLOTS:
        state.motor_state[slot].q = 0.2
    transport = Mock()
    transport.connect.side_effect = lambda cfg, callback: callback(state)
    config = G1ArmSDKConfig(
        network_interface="test",
        domain_id=1_000_000 + os.getpid(),  # Fake transport; never contend with the live robot lease.
        read_only=False,
        expected_mode_machine=5,
        kp=[80.0] * 4 + [40.0] * 3 + [80.0] * 4 + [40.0] * 3,
        kd=[3.0] * 4 + [1.5] * 3 + [3.0] * 4 + [1.5] * 3,
        lower=[-3.0] * 14,
        upper=[3.0] * 14,
        torque_limits=[15.0] * 14,
        period_s=1 / 250,
        measured_target_clipping=True,
        ramp_authority=False,
    )
    now = [1.0]
    robot = UnitreeArmSDK(config, transport=transport, clock=lambda: now[0])
    robot.connect()
    with patch("threading.Thread.start"):
        robot.activate()
    robot.thread = None
    return robot, transport, now, state


@pytest.fixture
def backend():
    robot, transport, now, state = make_backend()
    yield robot, transport, now, state
    robot.active = False
    robot.close()


def test_first_command_is_zero_target_clipped_from_feedback_with_zero_torque(backend):
    robot, transport, _, _ = backend
    robot._step()
    cmd = transport.write.call_args.args[0]
    assert cmd.motor_cmd[29].q == 1
    for slot in ARM_SLOTS:
        # Upstream: 0.2 + (0 - 0.2) / (0.2 / (30 / 250)) = 0.08.
        assert cmd.motor_cmd[slot].q == pytest.approx(0.08)
        assert cmd.motor_cmd[slot].dq == 0
        assert cmd.motor_cmd[slot].tau == 0
    assert cmd.motor_cmd[15].kp == 80 and cmd.motor_cmd[19].kp == 40
    assert all(cmd.motor_cmd[s].mode == 0 for s in range(15))


def test_reference_target_and_feedforward_are_not_experimental_slew_limited(backend):
    robot, transport, _, _ = backend
    robot.gravity = SimpleNamespace(gravity=lambda q: q * 2)
    target = np.full(14, 0.25)
    robot.send(dict(zip(ARM_KEYS, target, strict=True)))
    robot._step()
    cmd = transport.write.call_args.args[0]
    assert cmd.motor_cmd[15].q == pytest.approx(0.25)
    assert cmd.motor_cmd[15].tau == pytest.approx(0.5)


def test_reference_keeps_freshness_check(backend):
    robot, _, now, _ = backend
    now[0] += 0.2
    with pytest.raises(RuntimeError, match="stale feedback"):
        robot._step()


def advance_feedback(robot, now, state, duration):
    for _ in range(round(duration / 0.05)):
        now[0] += 0.05
        state.tick += 1
        robot._receive(state)


def test_short_caller_gap_holds_and_requires_current_resume_token(backend):
    robot, transport, now, state = backend
    advance_feedback(robot, now, state, 0.15)
    robot._step()
    assert robot.command_hold and robot.hold_generation == 1
    assert transport.write.call_args.args[0].motor_cmd[29].q == 1
    np.testing.assert_allclose(robot.target, 0.2)
    delayed = dict.fromkeys(ARM_KEYS, 0.6)
    robot.send(delayed)
    np.testing.assert_allclose(robot.target, 0.2)
    robot.send(delayed, resume_generation=0)
    assert robot.command_hold
    robot.send(delayed, resume_generation=1)
    assert not robot.command_hold
    np.testing.assert_allclose(robot.target, 0.6)
    advance_feedback(robot, now, state, 0.15)
    robot.send(delayed, resume_generation=1)
    assert robot.command_hold and robot.hold_generation == 2


def test_prolonged_caller_gap_still_faults_with_fresh_feedback(backend):
    robot, transport, now, state = backend
    advance_feedback(robot, now, state, 1.05)
    robot._run()
    assert "Caller command timeout" in robot.fault
    assert not robot.active and robot.release_sent
    assert len(transport.write.call_args_list) >= 3
    assert all(call.args[0].motor_cmd[29].q == 0 for call in transport.write.call_args_list[-3:])
    robot.fault = None  # fixture cleanup; no live publisher in this test


def test_normal_exit_homes_then_ramps_weight(monkeypatch, backend):
    robot, _, now, state = backend
    weights = []
    for slot in ARM_SLOTS:
        state.motor_state[slot].q = 0.0
    robot._receive(SimpleNamespace(**{**state.__dict__, "tick": 2}))

    def sleep(dt):
        weights.append(robot.release_weight)
        now[0] += dt

    monkeypatch.setattr("lerobot.robots.unitree_g1.g1_arm_unitree.time.sleep", sleep)
    robot.close()
    np.testing.assert_array_equal(robot.target, np.zeros(14))
    assert len(weights) == 101
    assert weights[0] == 1.0 and weights[-1] == 0.0
    assert all(a >= b for a, b in zip(weights, weights[1:], strict=False))
    assert sum([0.02] * len(weights)) == pytest.approx(2.02)
    robot.active = False


def test_reference_cli_selects_processor_without_guarded_profile(monkeypatch, caplog):
    caplog.set_level("INFO")
    import draccus

    from lerobot.robots.unitree_g1 import g1_vr_processor
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig

    cfg = draccus.parse(
        TeleoperateConfig,
        args=[
            "--robot.type=unitree_g1_motion",
            "--robot.mode=arms",
            "--robot.arm_controller=unitree",
            "--robot.enable_motion=true",
            "--robot.arm_sdk.network_interface=test",
            "--teleop.type=xr_controllers",
            "--teleop.full_input=true",
            "--teleop.terminal_control=true",
        ],
    )
    assert not cfg.robot.arm_test
    ik = Mock()
    ik.fk.return_value = (np.eye(4), np.eye(4))
    ik.solve.return_value = (np.zeros(14), np.zeros(14))
    monkeypatch.setattr(g1_vr_processor, "resolve_g1_vr_assets", lambda _: "unused")
    monkeypatch.setattr(g1_vr_processor, "G1VRKinematics", lambda _: ik)
    pipeline = g1_vr_processor.make_g1_vr_action_processor(cfg.robot, cfg.teleop)
    processor = pipeline.steps[0]
    assert processor.unitree_reference
    from lerobot.lerobot_types import TransitionKey
    from lerobot.robots.unitree_g1.g1_vr_control import synthetic_input

    processor._current_transition = {TransitionKey.OBSERVATION: dict.fromkeys(ARM_KEYS, 0.2)}
    initial = processor.action({"input.live": True})
    assert all(initial[key] == 0 for key in ARM_KEYS)
    assert initial["control.preparing"] and not initial["control.following"]
    action = synthetic_input((np.eye(4), np.eye(4)))
    action.update({"input.live": True, "control.start": True})
    following = processor.action(action)
    assert following["control.following"] and not following["control.preparing"]
    assert ik.solve.call_args.kwargs["max_step"] is None
    assert "Teleop state: following" in caplog.text
    lost = dict(action, **{"control.start": False, "left.tracked": False})
    held = processor.action(lost)
    assert not held["control.following"]
    assert "Teleop state: lost" in caplog.text
    assert "Left controller tracking unavailable" in caplog.text
    count = len(caplog.records)
    processor.action(lost)
    assert len(caplog.records) == count
    healthy = dict(action, **{"control.start": False})
    recovered = processor.action(healthy)
    assert recovered["control.following"]
    assert processor.reference_state == "following"
    processor._current_transition[TransitionKey.OBSERVATION].update(
        {"arm.hold_generation": 1.0, "arm.command_hold": 1.0}
    )
    held = processor.action(lost)
    assert not held["control.following"] and held["control.resume_generation"] == -1
    resumed = processor.action(healthy)
    assert resumed["control.following"] and resumed["control.resume_generation"] == 1
    processor._current_transition[TransitionKey.OBSERVATION]["arm.command_hold"] = 0.0
    paused = processor.action({"input.live": True, "control.pause": True})
    assert not paused["control.following"]
    assert processor.reference_state == "waiting"
    assert all(paused[key] == 0.2 for key in ARM_KEYS)
    assert not processor.action(healthy)["control.following"]
    processor._current_transition[TransitionKey.OBSERVATION].update(
        {"arm.hold_generation": 2.0, "arm.command_hold": 1.0}
    )
    still_waiting = processor.action(healthy)
    assert not still_waiting["control.following"]
    assert still_waiting["control.resume_generation"] == -1
    assert processor.reference_state == "waiting"
    quitting = processor.action({"input.live": True, "control.quit": True})
    assert quitting["control.stop"]


def test_feedback_snapshot_is_detached_and_immutable(backend):
    robot, _, _, state = backend
    snapshot = robot._snapshot()
    state.motor_state[15].q = 99.0
    state.imu_state.rpy[0] = 99.0
    assert snapshot.motor_state[15].q == 0.2
    assert snapshot.imu_state.rpy[0] == 0.0
    assert robot._copy_state(snapshot) is snapshot
    with pytest.raises(AttributeError):
        snapshot.motor_state[15].q = 1.0


@pytest.fixture
def reference_processor(monkeypatch):
    ik = Mock()
    ik.fk.return_value = (np.eye(4), np.eye(4))
    ik.solve.return_value = (np.full(14, 0.3), np.zeros(14))
    monkeypatch.setattr(g1_vr_processor, "G1VRKinematics", lambda _: ik)
    processor = g1_vr_processor.G1VRActionProcessor("unused", unitree_reference=True)
    processor._current_transition = {
        TransitionKey.OBSERVATION: {
            **dict.fromkeys(ARM_KEYS, 0.2),
            "arm.command_hold": 0.0,
            "arm.hold_generation": 0.0,
        }
    }
    return processor, ik


def test_slow_ik_cannot_revive_expired_tracking(monkeypatch, reference_processor):
    processor, ik = reference_processor
    now = [100.0]
    monkeypatch.setattr("lerobot.robots.unitree_g1.g1_vr_control.time.monotonic", lambda: now[0])
    action = synthetic_input((np.eye(4), np.eye(4)))
    action.update({"input.live": True, "control.start": True})
    assert processor.action(action)["control.following"]

    def slow_solve(*args, **kwargs):
        now[0] += 0.3
        return np.full(14, 0.8), np.zeros(14)

    ik.solve.side_effect = slow_solve
    action["control.start"] = False
    held = processor.action(action)
    assert not held["control.following"] and held["control.resume_generation"] == -1
    assert processor.reference_state == "lost"
    np.testing.assert_allclose([held[key] for key in ARM_KEYS], 0.2)
    ik.solve.side_effect = None
    action["captured_at"] = now[0]
    assert processor.action(action)["control.following"]


def test_ik_failure_stays_blocked_until_explicit_retry(reference_processor):
    processor, ik = reference_processor
    action = synthetic_input((np.eye(4), np.eye(4)))
    action["control.start"] = True
    ik.solve.side_effect = RuntimeError("IK did not converge")
    assert not processor.action(action)["control.following"]
    assert processor.reference_state == "blocked"
    ik.solve.side_effect = None
    action["control.start"] = False
    assert not processor.action(action)["control.following"]
    action["control.start"] = True
    assert processor.action(action)["control.following"]


def test_pause_before_first_follow_holds_measured_pose(reference_processor):
    processor, _ = reference_processor
    paused = processor.action({"control.pause": True})
    assert processor.reference_state == "waiting"
    np.testing.assert_allclose([paused[key] for key in ARM_KEYS], 0.2)


@pytest.mark.parametrize("generation", [-1, 0.5, float("nan"), float("inf")])
def test_invalid_hold_feedback_rejected(reference_processor, generation):
    processor, _ = reference_processor
    processor._current_transition[TransitionKey.OBSERVATION]["arm.hold_generation"] = generation
    with pytest.raises(ValueError, match="command-hold feedback"):
        processor.action({})


@pytest.mark.parametrize("token", [-1, 0.5, True])
def test_backend_rejects_invalid_resume_tokens(backend, token):
    robot, _, _, _ = backend
    with pytest.raises(ValueError, match="Resume generation"):
        robot.send(dict.fromkeys(ARM_KEYS, 0.2), resume_generation=token)


def test_feedback_fault_releases_authority_even_during_command_hold(backend):
    robot, transport, now, state = backend
    advance_feedback(robot, now, state, 0.15)
    robot._step()
    assert robot.command_hold
    now[0] += 0.2
    robot._run()
    assert "stale feedback" in robot.fault
    assert robot.release_sent and not robot.active
    assert transport.write.call_args.args[0].motor_cmd[29].q == 0
    robot.fault = None


def test_release_write_failure_is_reported(backend):
    robot, transport, now, _ = backend
    now[0] += 0.2
    transport.write.side_effect = RuntimeError("DDS write failed")
    robot._run()
    assert not robot.release_sent and not robot.active
    assert "release write failed" in robot.fault
    assert transport.write.call_count == 3
    robot.fault = None


def test_release_retries_after_transient_write_failure(backend):
    robot, transport, now, _ = backend
    now[0] += 0.2
    transport.write.side_effect = [RuntimeError("DDS write failed"), None, None]
    robot._run()
    assert robot.release_sent and not robot.active
    assert transport.write.call_count == 3
    assert all(call.args[0].motor_cmd[29].q == 0 for call in transport.write.call_args_list)
    assert "release write failed" in robot.fault
    robot.fault = None


def test_feedback_ingestion_continues_during_blocked_publish(backend):
    robot, transport, _, state = backend
    writing, allow_write, received = threading.Event(), threading.Event(), threading.Event()
    failures = []

    def write(_):
        writing.set()
        assert allow_write.wait(2)

    def publish():
        try:
            robot._step()
        except BaseException as exc:
            failures.append(exc)

    def receive():
        state.tick += 1
        state.motor_state[15].q = 0.3
        robot._receive(state)
        received.set()

    transport.write.side_effect = write
    publisher = threading.Thread(target=publish)
    callback = threading.Thread(target=receive)
    publisher.start()
    try:
        assert writing.wait(1)
        callback.start()
        assert received.wait(0.5), "Publishing must not lock out the feedback callback"
        assert robot._snapshot().motor_state[15].q == 0.3
        assert robot.fault is None
    finally:
        allow_write.set()
        publisher.join(2)
        if callback.ident is not None:
            callback.join(2)
    assert not failures


def test_receive_gap_remains_fatal_and_reports_callback_timing(backend):
    robot, _, now, state = backend
    now[0] += 0.15
    state.tick += 1
    robot._receive(state)
    assert "Feedback receive gap exceeded" in robot.fault
    assert "callback gap=150.0 ms" in robot.fault
    assert "lock wait=0.0 ms" in robot.fault
    robot._run()
    assert robot.release_sent and not robot.active
    robot.fault = None
