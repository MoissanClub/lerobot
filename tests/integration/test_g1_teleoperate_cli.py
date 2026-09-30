# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Offline acceptance of the standard CLI, real IK and real MuJoCo/GR00T."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get("G1_VR_ASSETS"), reason="Requires pinned cached assets and policies")
def test_standard_cli_replay(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), HF_HUB_OFFLINE="1", MUJOCO_GL="egl")
    prepare = subprocess.run(
        [
            sys.executable,
            "examples/unitree_g1/prepare_vr_assets.py",
            "--config-dir",
            str(tmp_path),
            "--replay-frames",
            "100",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert prepare.returncode == 0, prepare.stdout + prepare.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_teleoperate",
            f"--config_path={tmp_path}/teleoperate.json",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in (tmp_path / "report.jsonl").read_text().splitlines()]
    actions = [row for row in rows if row["event"] == "action"]
    assert len(actions) == 100
    assert all(row["enabled"] for row in actions)
    assert all(not row["motor_publication"] and not row["base_rpc"] for row in actions)
    assert np.max(np.ptp([row["measured"] for row in actions], axis=0)) > 0.001
    assert rows[-1]["event"] == "closing"
