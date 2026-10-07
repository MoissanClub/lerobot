# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
import json
import time
from dataclasses import replace
from unittest.mock import Mock

import draccus
import numpy as np
import pytest

from lerobot.processor.factory import make_default_processors
from lerobot.robots.unitree_g1 import g1_motion, g1_vr_assets, g1_vr_processor
from lerobot.robots.unitree_g1.g1_arm_sdk import G1ArmSDKConfig
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.teleoperators.xr_controllers import XRControllersConfig


def test_cli_parses_nested_hardware_without_files():
    cfg = draccus.parse(
        TeleoperateConfig,
        args=[
            "--robot.type=unitree_g1_motion",
            "--robot.mode=shadow",
            "--robot.arm_sdk.network_interface=robot_eth",
            "--teleop.type=xr_controllers",
            "--teleop.full_input=true",
        ],
    )
    assert cfg.robot.contract is None
    assert cfg.robot.assets is None
    assert cfg.robot.arm_sdk.network_interface == "robot_eth"
    assert np.allclose(cfg.teleop.base_T_anchor, np.eye(4))


def test_model_resolution_pins_revision_and_respects_local_override(monkeypatch):
    download = Mock(return_value="/model")
    monkeypatch.setattr(g1_vr_assets, "snapshot_download", download)
    assert g1_vr_assets.resolve_g1_vr_assets("/local") == "/local"
    download.assert_not_called()
    assert g1_vr_assets.resolve_g1_vr_assets(None) == "/model"
    download.assert_called_once_with("lerobot/unitree-g1-mujoco", revision=g1_vr_assets.MODEL_REVISION)


def test_pair_selects_processor_without_saved_pipeline(monkeypatch):
    monkeypatch.setattr(g1_vr_processor, "resolve_g1_vr_assets", lambda _: "/model")
    ik = Mock()
    monkeypatch.setattr(g1_vr_processor, "G1VRKinematics", ik)
    processors = make_default_processors(
        robot_config=g1_motion.UnitreeG1MotionConfig(),
        teleop_config=XRControllersConfig(full_input=True),
    )
    assert isinstance(processors[0].steps[0], g1_vr_processor.G1VRActionProcessor)
    ik.assert_called_once_with("/model")
    with pytest.raises(ValueError, match="full_input"):
        make_default_processors(
            robot_config=g1_motion.UnitreeG1MotionConfig(), teleop_config=XRControllersConfig()
        )
    with pytest.raises(ValueError, match="identity"):
        make_default_processors(
            robot_config=g1_motion.UnitreeG1MotionConfig(),
            teleop_config=XRControllersConfig(
                full_input=True, base_T_anchor=XRControllersConfig().base_T_anchor
            ),
        )


def test_shadow_forces_read_only_without_command_transport(tmp_path, monkeypatch):
    arm = Mock()
    sdk = Mock(return_value=arm)
    monkeypatch.setattr(g1_motion, "G1ArmSDK", sdk)
    monkeypatch.setattr(g1_motion, "G1VRKinematics", Mock())
    monkeypatch.setattr(g1_motion, "digest", lambda _: "model-hash")
    monkeypatch.setattr(g1_motion, "G1BaseMotion", Mock(side_effect=AssertionError("base transport")))
    robot = g1_motion.UnitreeG1Motion(
        g1_motion.UnitreeG1MotionConfig(
            mode="shadow",
            assets="/local",
            arm_sdk=G1ArmSDKConfig(network_interface="robot_eth"),
            report_path=str(tmp_path / "report.jsonl"),
        )
    )
    robot.connect()
    try:
        assert sdk.call_args.args[0].read_only
        arm.active = False
        arm.observation.return_value = dict.fromkeys(robot.observation_features, 0.0)
        action = dict.fromkeys(robot.action_features, 0.0)
        action.update({"control.created_at": time.monotonic(), "control.enabled": 1.0, "control.live": 1.0})
        robot.send_action(action)
        arm.activate.assert_not_called()
        arm.send.assert_not_called()
    finally:
        robot.disconnect()
    arm.close.assert_called_once()
    rows = [json.loads(line) for line in (tmp_path / "report.jsonl").read_text().splitlines()]
    sample = next(row for row in rows if row["event"] == "action")
    assert sample["enabled"] and not sample["motor_publication"] and not sample["base_rpc"]
    assert len(sample["observation"]) == 29 * 3
    assert len(sample["measured"]) == len(sample["target"]) == 14
    assert sample["recorded_at"] >= sample["created_at"]


def test_direct_arm_limits_and_mode_are_required_and_model_bounded(tmp_path, monkeypatch):
    kwargs = {"mode": "arms", "enable_motion": True}
    sdk = G1ArmSDKConfig(network_interface="robot_eth")
    with pytest.raises(ValueError, match="expected_mode_machine"):
        g1_motion.UnitreeG1MotionConfig(**kwargs, arm_sdk=sdk)
    sdk = replace(sdk, expected_mode_machine=1)
    with pytest.raises(ValueError, match="kp"):
        g1_motion.UnitreeG1MotionConfig(**kwargs, arm_sdk=sdk)
    sdk = replace(sdk, kp=[10.0] * 14, kd=[1.0] * 14, torque_limits=[2.0] * 14)
    cfg = g1_motion.UnitreeG1MotionConfig(**kwargs, arm_sdk=sdk)
    limits = {"lower": [-1.0] * 14, "upper": [1.0] * 14, "effort": [5.0] * 14}
    monkeypatch.setattr(g1_motion, "model_limits", lambda _: limits)
    from lerobot.robots.unitree_g1 import g1_safety_contract

    monkeypatch.setattr(g1_safety_contract, "model_limits", lambda _: limits)
    robot = g1_motion.UnitreeG1Motion(cfg)
    actual = robot._direct_arm_config(tmp_path / "robot.urdf")
    assert not actual.read_only
    assert actual.lower == limits["lower"]
    assert actual.max_displacement == sdk.max_displacement
    robot.config.arm_sdk = replace(sdk, torque_limits=[6.0] * 14)
    with pytest.raises(ValueError, match="torque bounds"):
        robot._direct_arm_config(tmp_path / "robot.urdf")
    robot.config.arm_sdk = replace(sdk, lower=[-2.0] * 14, upper=[1.0] * 14)
    with pytest.raises(ValueError, match="joint bounds"):
        robot._direct_arm_config(tmp_path / "robot.urdf")
    with pytest.raises(ValueError, match="not both"):
        g1_motion.UnitreeG1MotionConfig(**kwargs, arm_sdk=sdk, contract="old.json")


def test_arm_test_cli_profile_cannot_widen_motion_bounds(tmp_path, monkeypatch):
    cfg = draccus.parse(
        TeleoperateConfig,
        args=[
            "--robot.type=unitree_g1_motion",
            "--robot.mode=arms",
            "--robot.arm_test=true",
            "--robot.arm_test_workspace=small",
            "--robot.enable_motion=true",
            "--robot.arm_sdk.network_interface=robot_eth",
            "--robot.arm_sdk.max_velocity=1",
            "--robot.arm_sdk.max_displacement=1",
            "--teleop.type=xr_controllers",
            "--teleop.full_input=true",
        ],
    )
    monkeypatch.setattr(
        g1_motion,
        "model_limits",
        lambda _: {
            "lower": [-2.0] * 14,
            "upper": [2.0] * 14,
            "effort": [20.0] * 14,
        },
    )
    monkeypatch.setattr(g1_motion, "validate_arm_model", Mock())
    actual = g1_motion.UnitreeG1Motion(cfg.robot)._direct_arm_config(tmp_path / "robot.urdf")
    assert actual.max_velocity == 0.02
    assert actual.max_acceleration == 0.05
    assert actual.max_displacement == 0.03
    assert actual.max_wrist_displacement == 0.025
    assert actual.start_position == [0.0] * 14
    assert actual.start_tolerance == 0.05
    assert actual.blend_s == 5
    assert actual.kp == [60.0] * 14 and actual.kd == [1.5] * 14
    assert not actual.read_only
    cfg.robot.arm_test_start_pose = "current"
    current = g1_motion.UnitreeG1Motion(cfg.robot)._direct_arm_config(tmp_path / "robot.urdf")
    assert current.start_position == []
    assert current.max_displacement == actual.max_displacement
    assert current.max_wrist_displacement == actual.max_wrist_displacement
    assert current.max_velocity == actual.max_velocity
    with pytest.raises(ValueError, match="requires arm_test"):
        g1_motion.UnitreeG1MotionConfig(arm_test_start_pose="current")
    with pytest.raises(ValueError, match="arm_test"):
        g1_motion.UnitreeG1MotionConfig(mode="simulation", arm_test=True)


def test_front_box_profile_retains_slow_limits_and_selects_countdown(tmp_path, monkeypatch):
    limits = {"lower": [-2.0] * 14, "upper": [2.0] * 14, "effort": [20.0] * 14}
    monkeypatch.setattr(g1_motion, "model_limits", lambda _: limits)
    monkeypatch.setattr(g1_motion, "validate_arm_model", lambda *_: None)
    monkeypatch.setattr(g1_vr_processor, "resolve_g1_vr_assets", lambda _: "/model")
    monkeypatch.setattr(g1_vr_processor, "G1VRKinematics", Mock())
    cfg = g1_motion.UnitreeG1MotionConfig(
        mode="arms",
        arm_test=True,
        arm_test_start_pose="current",
        enable_motion=True,
        arm_sdk=G1ArmSDKConfig(network_interface="robot-test"),
    )
    robot = g1_motion.UnitreeG1Motion(cfg)
    sdk = robot._direct_arm_config(tmp_path / "robot.urdf")
    assert sdk.max_velocity == 30.0
    assert sdk.max_acceleration == 0.05
    assert sdk.max_wrist_displacement is None
    assert sdk.workspace_lower == [-0.02, -0.5, -0.5]
    assert not sdk.ramp_authority
    assert sdk.measured_target_clipping
    assert sdk.period_s == 1 / 250
    assert sdk.kp == [80.0] * 4 + [40.0] * 3 + [80.0] * 4 + [40.0] * 3
    assert sdk.kd == [3.0] * 4 + [1.5] * 3 + [3.0] * 4 + [1.5] * 3
    assert sdk.workspace_upper == [1, 0.5, 0.5]
    assert sdk.max_displacement == 4.0
    processors = make_default_processors(robot_config=cfg, teleop_config=XRControllersConfig(full_input=True))
    assert processors[0].steps[0].workspace == "front_box"


def test_cli_sync_distance_reaches_processor_and_serialized_config(monkeypatch):
    monkeypatch.setattr(g1_vr_processor, "resolve_g1_vr_assets", lambda _: "/model")
    monkeypatch.setattr(g1_vr_processor, "G1VRKinematics", Mock())
    cfg = draccus.parse(
        TeleoperateConfig,
        args=[
            "--robot.type=unitree_g1_motion",
            "--robot.mode=arms",
            "--robot.arm_test=true",
            "--robot.enable_motion=true",
            "--robot.arm_sdk.network_interface=robot_eth",
            "--robot.arm_test_sync_distance_m=0.15",
            "--teleop.type=xr_controllers",
            "--teleop.full_input=true",
        ],
    )
    processor = make_default_processors(robot_config=cfg.robot, teleop_config=cfg.teleop)[0].steps[0]
    assert processor.front_box.sync_distance_m == 0.15
    assert processor.front_box.unsync_distance_m == pytest.approx(0.18)
    assert processor.get_config()["sync_distance_m"] == 0.15
