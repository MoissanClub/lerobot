# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Run on the combined simulation + XR stack, not the independent XR branch."""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("G1_INTEGRATION_TESTS"), reason="Requires combined simulation/XR stack"
)


@pytest.mark.parametrize("embodiment", ["g1_29", "g1_23"])
def test_dual_arm_replay(embodiment, tmp_path):
    from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
    from lerobot.robots.unitree_g1.g1_cartesian_control import G1ArmKinematics, G1CartesianConfig
    from lerobot.robots.unitree_g1.g1_xr_control import G1XRControl

    root = Path(os.environ["G1_KINEMATICS_ASSETS"])
    robot = UnitreeG1(
        UnitreeG1Config(
            embodiment=embodiment,
            simulation_urdf=str(root / f"{embodiment}.urdf"),
            gravity_compensation=True,
            calibration_dir=tmp_path,
        )
    )
    try:
        robot.connect()
        sim = robot._native
        ik = G1ArmKinematics(G1CartesianConfig(embodiment, root / f"{embodiment}.urdf"))
        q = np.zeros(ik.size)
        q[[0, ik.size // 2]] = -0.4
        q[[3, ik.size // 2 + 3]] = 0.7
        robot.send_action(ik.arm_action(q))
        for _ in range(500):
            robot.step_simulation()
        control = G1XRControl(ik)
        action = {"captured_at": 1.0}
        for side in ("left", "right"):
            action.update(
                {
                    f"{side}.tracked": True,
                    f"{side}.grip_pos": np.zeros(3),
                    f"{side}.grip_quat": [0, 0, 0, 1],
                    f"{side}.squeeze": 1.0,
                    f"{side}.trigger": 0.0,
                }
            )
        initial = ik.fk(sim.data.qpos[sim.qadr])
        for frame in range(100):
            for side in ("left", "right"):
                action[f"{side}.grip_pos"] = np.array([0, 0, min(frame / 40, 1) * 0.025])
            command = control.action(action, sim.data.qpos[sim.qadr].copy(), now=1)
            robot.send_action(command)
            for _ in range(4):
                robot.step_simulation()
        final = ik.fk(sim.data.qpos[sim.qadr])
        for before, after in zip(initial, final, strict=True):
            assert after[2, 3] - before[2, 3] > 0.01
        q = sim.data.qpos[sim.qadr].copy()
        held = control.action(action, q, now=2)
        np.testing.assert_allclose(list(held.values()), q)
        assert not any(control.engaged)
    finally:
        robot.disconnect()
