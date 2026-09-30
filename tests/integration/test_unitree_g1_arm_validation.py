# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Offline CLI gating and opt-in MuJoCo tests; no physical network access."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.unitree_g1.validate_arm_sdk import contract_template, parse_args, trajectory, validate_contract


def test_cli_defaults_and_hardware_optin():
    args = parse_args(["--contract", "review.json", "--output", "result.json"])
    assert args.backend == "dry-run" and args.phase == "read-only"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--backend",
                "hardware",
                "--phase",
                "hold",
                "--contract",
                "review.json",
                "--output",
                "result.json",
            ]
        )


@pytest.mark.parametrize("duration", ["nan", "inf", "0", "31"])
def test_reject_bad_duration(duration):
    with pytest.raises(SystemExit):
        parse_args(["--contract", "review.json", "--output", "result.json", "--duration-s", duration])


def test_trajectory_returns_to_start_and_changes_only_one_joint():
    import numpy as np

    start = np.zeros(14)
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        q = trajectory(start, 3, 0.02, fraction)
        assert np.count_nonzero(np.delete(q, 3)) == 0
        assert 0 <= q[3] <= 0.02
    assert trajectory(start, 3, 0.02, 0.5)[3] == pytest.approx(0.02)
    assert trajectory(start, 3, 0.02, 1)[3] == 0


@pytest.fixture
def assets():
    if not os.environ.get("G1_KINEMATICS_ASSETS"):
        pytest.skip("Set G1_KINEMATICS_ASSETS for real-model validation")
    return Path(os.environ["G1_KINEMATICS_ASSETS"]) / "g1_29.urdf"


def test_unreviewed_template_cannot_enable_hardware(assets):
    contract = contract_template(assets)
    with pytest.raises(ValueError):
        validate_contract(contract, assets, hardware_motion=True)
    contract["arm_sdk"]["expected_mode_machine"] = 5
    with pytest.raises(ValueError, match="not approved"):
        validate_contract(contract, assets, hardware_motion=True)
    contract["urdf_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        validate_contract(contract, assets, hardware_motion=False)


@pytest.mark.parametrize(
    "phase,joint",
    [("read-only", None), ("hold", None), ("joint", "kLeftElbow.q"), ("joint", "kRightElbow.q")],
)
def test_real_physics_and_clean_exit(assets, tmp_path, phase, joint):
    if not os.environ.get("G1_INTEGRATION_TESTS"):
        pytest.skip("Set G1_INTEGRATION_TESTS=1 for physics-in-loop tests")
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(contract_template(assets)))
    output = tmp_path / "report.json"
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
        str(assets),
        "--output",
        str(output),
    ]
    if joint:
        cmd.extend(["--joint", joint, "--delta-rad", "0.02"])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(output.read_text())
    assert data["status"] == "software_checks_passed"
    assert data["physical_stop_verified"] is False
    assert data["hardware_acceptance"] == "pending_operator_review"
    assert data["release_packet_sent"] is (phase != "read-only")
    assert len(data["samples"]) > 20
    # Reports must never silently overwrite earlier evidence.
    repeated = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert repeated.returncode != 0


def test_hardware_unreviewed_fails_before_transport(assets, tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(contract_template(assets)))
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/unitree_g1/validate_arm_sdk.py",
            "--backend",
            "hardware",
            "--phase",
            "hold",
            "--enable-motion",
            "--contract",
            str(contract),
            "--urdf",
            str(assets),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    data = json.loads(output.read_text())
    assert data["motion_enabled"] is False and not data["samples"]
