# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Opt-in pinned-model acceptance: G1_VR_ASSETS enables real headless physics.

No DDS or hardware. Assets must already be downloaded; no network during tests.
"""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.robots.unitree_g1.g1_vr_control import G1VRKinematics, map_input, synthetic_input


@pytest.fixture(scope="module")
def assets():
    path = os.environ.get("G1_VR_ASSETS")
    if not path:
        pytest.skip("Set G1_VR_ASSETS to the prepared Hub snapshot")
    assert (Path(path) / "assets/g1_body29_hand14.urdf").is_file()
    return Path(path)


@pytest.fixture(scope="module")
def ik(assets):
    return G1VRKinematics(assets)


def test_upstream_ik_order_gravity_and_target_bounds(ik):
    q = np.zeros(14)
    q[[0, 3, 7, 10]] = [-0.15, 0.4, -0.12, 0.35]
    wrists = ik.fk(q)
    intent = map_input(synthetic_input(wrists))
    solved, tau = ik.solve(intent["wrists"], q)
    assert max(abs(solved - q)) <= 0.04 + 1e-9
    assert np.isfinite(tau).all() and max(abs(tau)) > 0.1
    np.testing.assert_allclose(ik.gravity(q), ik.solver.solve_tau(q), atol=1e-9)
    for wanted, actual in zip(wrists, ik.fk(solved), strict=True):
        assert np.linalg.norm(wanted[:3, 3] - actual[:3, 3]) < 0.03
    assert [ik.solver._arm_joint_names_g1[i] for i in ik.to_pin] == ik.solver._arm_joint_names_pin
    np.testing.assert_array_equal(np.asarray(ik.to_pin)[ik.to_motor], np.arange(14))


@pytest.mark.parametrize(
    "velocity",
    [(0, 0, 0), (0.25, 0, 0), (-0.25, 0, 0), (0, 0.25, 0), (0, -0.25, 0), (0, 0, 0.25), (0, 0, -0.25)],
)
def test_groot_direction_without_dds(assets, velocity):
    from lerobot.robots.unitree_g1.g1_vr_simulation import G1VRSimulation

    with patch(
        "unitree_sdk2py.core.channel.ChannelFactoryInitialize", side_effect=AssertionError("DDS forbidden")
    ):
        sim = G1VRSimulation(assets)
        try:
            for _ in range(100):
                sim.step()
            start = sim.data.qpos[:7].copy()
            for _ in range(400):
                sim.step(velocity=velocity)
            end = sim.data.qpos[:7].copy()
            assert 0.6 < end[2] < 0.9
            displacement = end[:2] - start[:2]
            if velocity[0]:
                assert displacement[0] * np.sign(velocity[0]) > 0.5
            elif velocity[1]:
                assert displacement[1] * np.sign(velocity[1]) > 0.5
            elif velocity[2]:
                yaw = [Rotation.from_quat(q[[4, 5, 6, 3]]).as_euler("xyz")[2] for q in (start, end)]
                assert (
                    np.arctan2(np.sin(yaw[1] - yaw[0]), np.cos(yaw[1] - yaw[0])) * np.sign(velocity[2]) > 0.5
                )
            else:
                assert np.linalg.norm(displacement) < 0.1
        finally:
            sim.close()


def test_camera_and_combined_arm_motion(assets, ik):
    from lerobot.robots.unitree_g1.g1_vr_simulation import G1VRSimulation

    with patch(
        "unitree_sdk2py.core.channel.ChannelFactoryInitialize", side_effect=AssertionError("DDS forbidden")
    ):
        sim = G1VRSimulation(assets, video=True)
        try:
            q = np.zeros(14)
            for _ in range(100):
                sim.step(q, ik.gravity(q))
            initial = sim.state().motor_state[18].q
            for step in range(150):
                q[3] = 0.2 * min(step / 100, 1)
                sim.step(q, ik.gravity(q), (0.15, 0, 0))
            assert sim.state().motor_state[18].q > initial + 0.1
            image = sim.render()
            assert image.shape == (480, 640, 3) and image.std() > 5
        finally:
            sim.close()


def test_simulation_inertias_match_upstream_static_gravity(assets, ik):
    from lerobot.robots.unitree_g1.g1_vr_simulation import G1VRSimulation

    sim = G1VRSimulation(assets, stationary_fixture=True)
    try:
        np.testing.assert_allclose(
            ik.gravity(np.zeros(14)), sim.data.qfrc_bias[sim.env.body_dof_index[15:]], atol=1e-4
        )
        assert sim.model.neq == 30  # 14 locked fingers + pelvis + 15 lower-body joints.
    finally:
        sim.close()
