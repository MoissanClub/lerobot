# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Runner boundaries and real supported-arm diagnostic subprocesses; no hardware."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.unitree_g1 import run_vr_teleop as runner, validate_arm_sdk as diagnostic
from lerobot.robots.unitree_g1.g1_vr_control import ARM_KEYS, synthetic_input


def test_shadow_cannot_activate_or_create_base_client(tmp_path, monkeypatch):
    from lerobot.robots.unitree_g1 import g1_arm_sdk, g1_base_motion
    from lerobot.teleoperators import xr_controllers

    monkeypatch.setitem(sys.modules, "validate_arm_sdk", diagnostic)
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    cfg = g1_arm_sdk.G1ArmSDKConfig(network_interface="mock-robot")
    monkeypatch.setattr(diagnostic, "validate_contract", lambda *args: cfg)
    monkeypatch.setattr(diagnostic, "digest", lambda *args: "fake")
    wrists = [np.eye(4), np.eye(4)]

    class IK:
        def __init__(self, *args):
            pass

        def fk(self, q):
            return wrists

        def solve(self, *args):
            return np.full(14, 0.01), np.zeros(14)

        def gravity(self, q):
            return np.zeros(14)

    class Robot:
        active = False

        def __init__(self, config):
            assert config.read_only

        def connect(self):
            pass

        def close(self):
            pass

        def observation(self):
            return dict.fromkeys(ARM_KEYS, 0.0)

        def activate(self):
            raise AssertionError("shadow activation")

        def send(self, action):
            raise AssertionError("shadow command")

    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            pass

        def disconnect(self):
            pass

        def get_action(self):
            return synthetic_input(wrists)

    monkeypatch.setattr(runner, "G1VRKinematics", IK)
    monkeypatch.setattr(g1_arm_sdk, "G1ArmSDK", Robot)
    monkeypatch.setattr(xr_controllers, "XRControllers", Reader)
    monkeypatch.setattr(g1_base_motion, "G1BaseMotion", lambda *a, **kw: pytest.fail("base RPC in shadow"))
    monkeypatch.setattr(runner.select, "select", lambda *args: ([runner.sys.stdin], [], []))
    monkeypatch.setattr(runner.sys, "stdin", io.StringIO("r\n"))
    args = runner.parse_args(
        [
            "--mode",
            "shadow",
            "--assets",
            str(tmp_path),
            "--contract",
            str(contract),
            "--output",
            str(tmp_path / "out"),
            "--duration-s",
            ".08",
        ]
    )
    output = io.StringIO()
    runner.run(args, output)
    frames = [r for r in map(json.loads, output.getvalue().splitlines()) if r["event"] == "frame"]
    assert frames and any(r["started"] for r in frames)
    assert all(not r["motor_publication"] and not r["base_rpc"] for r in frames)


@pytest.mark.parametrize(
    "phase,joint",
    [("read-only", None), ("hold", None), ("joint", "kLeftElbow.q"), ("joint", "kRightElbow.q")],
)
def test_real_supported_arm_diagnostic(tmp_path, phase, joint):
    assets = os.environ.get("G1_VR_ASSETS")
    if not assets:
        pytest.skip("Set G1_VR_ASSETS for headless physics tests")
    urdf = Path(assets) / "assets/g1_body29_hand14.urdf"
    contract = tmp_path / "contract.json"
    config = diagnostic.contract_template(urdf)
    # This tests physics/trajectory semantics, not real-time PC2 scheduling. The
    # shared CI host occasionally stalls >100ms; watchdog bounds have unit tests.
    # Never change the hardware template or infer hardware timing acceptance here.
    config["arm_sdk"]["max_state_age_s"] = 0.5
    config["arm_sdk"]["command_timeout_s"] = 0.5
    contract.write_text(json.dumps(config))
    report = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "examples/unitree_g1/validate_arm_sdk.py",
        "--backend",
        "simulation",
        "--phase",
        phase,
        "--contract",
        str(contract),
        "--urdf",
        str(urdf),
        "--output",
        str(report),
    ]
    if joint:
        cmd += ["--joint", joint, "--delta-rad", ".02"]
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert data["status"] == "software_checks_passed"
    assert data["hardware_acceptance"] == "pending_operator_review"
    assert data["motion_enabled"] == (phase != "read-only")
