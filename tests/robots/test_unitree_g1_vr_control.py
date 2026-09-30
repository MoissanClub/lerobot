# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Offline mapping, RPC watchdog and CLI gates. Never contact hardware."""

import importlib.util
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.robots.unitree_g1.g1_base_motion import G1BaseMotion
from lerobot.robots.unitree_g1.g1_vr_control import map_input, stop_buttons, synthetic_input


def sample():
    wrists = [np.eye(4), np.eye(4)]
    wrists[0][:3, 3], wrists[1][:3, 3] = [0.3, 0.2, 0.2], [0.3, -0.2, 0.2]
    return synthetic_input(wrists), wrists


def test_pose_roundtrip_and_openxr_sticks():
    action, wrists = sample()
    action.update({"left.stick_y": 1.0, "left.stick_x": -1.0, "right.stick_x": 1.0})
    intent = map_input(action)
    np.testing.assert_allclose(intent["wrists"], wrists, atol=1e-12)
    np.testing.assert_allclose(intent["velocity"], [0.3, 0.3, -0.3])
    assert not any(intent["hands"][0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("left.grip_quat", [0] * 4),
        ("head.tracked", False),
        ("left.stick_x", float("nan")),
        ("right.trigger", 1.1),
        ("captured_at", -1),
    ],
)
def test_invalid_input_rejected(field, value):
    action, _ = sample()
    action[field] = value
    with pytest.raises(ValueError):
        map_input(action)


def test_stop_priority_and_pose_loss():
    action, _ = sample()
    action.update({"left.stick_click": True, "right.stick_click": True, "left.stick_y": 1.0})
    assert not map_input(action)["velocity"].any()
    action["head.tracked"] = False
    assert stop_buttons(action) == (False, True)
    action["captured_at"] = -1
    assert stop_buttons(action) == (False, False)


def test_reference_finger_intent_not_clutch():
    action, _ = sample()
    action.update({"left.trigger": 1.0, "left.squeeze": 0.8})
    assert map_input(action)["hands"][0] == [0.98, 0.7, 0.8, 0.98, 0.98, 0.98]


def client():
    return Mock(SetVelocity=Mock(return_value=0), SetFsmId=Mock(return_value=0))


def test_base_no_side_effect_until_activation_and_stop_priority():
    rpc = client()
    base = G1BaseMotion(lambda: {}, client=rpc, domain_id=99931)
    rpc.SetVelocity.assert_not_called()
    try:
        base.activate()
        base.send([1.0, -0.5, 0.3])
        time.sleep(0.06)
        base.close(damp=True)
    finally:
        base.close()
    calls = rpc.SetVelocity.call_args_list
    assert any(c.args[:3] == (0.1, -0.1, 0.1) for c in calls)
    assert calls[-1].args[:3] == (0.0, 0.0, 0.0)
    assert rpc.mock_calls[-1][0] == "SetFsmId"


def test_base_timeout_zeros_and_latches():
    rpc = client()
    base = G1BaseMotion(lambda: {}, timeout=0.05, client=rpc, domain_id=99932)
    base.activate()
    base.thread.join(1)
    with pytest.raises(RuntimeError, match="timeout"):
        base.send([0, 0, 0])
    with pytest.raises(RuntimeError, match="timeout"):
        base.close()
    assert rpc.SetVelocity.call_args.args[:3] == (0.0, 0.0, 0.0)


def test_base_rpc_rejection_is_not_success():
    rpc = client()
    rpc.SetVelocity.return_value = 42
    base = G1BaseMotion(lambda: {}, client=rpc, domain_id=99933)
    base.activate()
    base.thread.join(1)
    with pytest.raises(RuntimeError, match="42"):
        base.close()


@pytest.mark.parametrize(
    "extra",
    [
        ["--mode", "arms"],
        ["--mode", "walk", "--enable-motion"],
        ["--enable-motion"],
        ["--mode", "shadow"],
        ["--input", "synthetic", "--video"],
    ],
)
def test_cli_rejects_unsafe_combinations_before_io(extra):
    path = Path(__file__).parents[2] / "examples/unitree_g1/run_vr_teleop.py"
    spec = importlib.util.spec_from_file_location("vr_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.parse_args(["--assets", "/unused", "--output", "/unused"] + extra)
