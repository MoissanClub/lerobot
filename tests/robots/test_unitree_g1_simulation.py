# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Native backend configuration and lifecycle boundaries without optional physics."""

import pytest

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config


def test_native_config_allows_g1_23_gravity_without_enabling_hardware():
    config = UnitreeG1Config(embodiment="g1_23", simulation_urdf="local.urdf", gravity_compensation=True)
    assert config.gravity_compensation
    with pytest.raises(ValueError):
        UnitreeG1Config(embodiment="g1_23", gravity_compensation=True)


@pytest.mark.parametrize(
    "options",
    [
        {"is_simulation": False},
        {"controller": "SonicWholeBodyController"},
        {"control_dt": 0},
        {"control_dt": float("nan")},
    ],
)
def test_native_rejects_unsafe_config(options):
    with pytest.raises(ValueError):
        UnitreeG1Config(simulation_urdf="local.urdf", **options)


def test_native_does_not_require_sdk_or_download_legacy_ik(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Native initialization touched SDK or legacy download path")

    monkeypatch.setattr("lerobot.robots.unitree_g1.unitree_g1.require_package", unexpected)
    monkeypatch.setattr("lerobot.robots.unitree_g1.unitree_g1.G1_29_ArmIK", unexpected)
    robot = UnitreeG1(UnitreeG1Config(simulation_urdf="missing.urdf", gravity_compensation=True))
    assert not robot.is_connected
    assert robot.get_observation() == {}
    with pytest.raises(RuntimeError, match="Not connected"):
        robot.send_action({})
    robot.disconnect()
