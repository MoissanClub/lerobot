# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("G1_INTEGRATION_TESTS"), reason="Requires combined simulation/video stack"
)


@pytest.mark.parametrize("embodiment", ["g1_29", "g1_23"])
def test_native_camera_channel_motion(embodiment, tmp_path):
    from lerobot.cameras.frame_channel import FrameWriter, read_frame
    from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config

    assets = Path(os.environ["G1_KINEMATICS_ASSETS"])
    robot = UnitreeG1(
        UnitreeG1Config(
            embodiment=embodiment,
            simulation_urdf=str(assets / f"{embodiment}.urdf"),
            simulation_mesh_dir=str(assets / "meshes"),
            gravity_compensation=True,
            calibration_dir=tmp_path,
        )
    )
    writer = FrameWriter(tmp_path / "camera.rgb", 320, 240)
    try:
        robot.connect()
        before = robot.render_simulation(320, 240)
        sim = robot._native
        q = np.zeros(sim.size)
        q[[0, sim.size // 2]] = -0.5
        q[[3, sim.size // 2 + 3]] = 0.8
        robot.send_action(sim.arm_action(q))
        for _ in range(500):
            robot.step_simulation()
        after = robot.render_simulation(320, 240)
        assert before.std() > 2 and after.std() > 2
        assert np.mean(np.abs(after.astype(float) - before)) > 0.1
        for pixels in (before, after):
            writer.publish(pixels, {"captured_monotonic_ns": time.monotonic_ns(), "embodiment": embodiment})
            info, received = read_frame(writer.path)
            assert info["embodiment"] == embodiment
            np.testing.assert_array_equal(received, pixels)
    finally:
        writer.close()
        robot.disconnect()
