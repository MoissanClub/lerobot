#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for G1 with BrainCo hands; hardware and simulator I/O is mocked."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
from tests.robots.unitree_g1_test_utils import (
    LocomotionOnlyController,
    TokenController,
    arm_for_publish,
    make_robot as _make_robot_fixture,  # noqa: F401 - registers the shared pytest fixture
)


class TestBrainCoSimulation:
    @pytest.mark.parametrize("controller", [None, TokenController(), LocomotionOnlyController()])
    def test_hand_features_extend_controller_features(self, make_robot, controller):
        factory, _ = make_robot
        robot = factory(end_effector="brainco", controller=controller)
        expected = {f"hands.{side}.motor_{i}.pos" for side in ("left", "right") for i in range(6)}
        assert robot._hands_ft.keys() == expected
        assert expected <= robot.action_features.keys()
        assert expected <= robot.observation_features.keys()

    @pytest.mark.parametrize("controller", [None, TokenController(), LocomotionOnlyController()])
    def test_hand_only_action_does_not_publish_body(self, make_robot, controller):
        factory, mocks = make_robot
        robot = arm_for_publish(factory(end_effector="brainco", controller=controller), mocks)
        robot.sim_env = MagicMock()
        action = {"hands.right.motor_2.pos": 0.7}
        assert robot.send_action(action) == action
        robot.sim_env.send_hand_action.assert_called_once_with(action)
        mocks["publisher_mock"].Write.assert_not_called()

    def test_mixed_action_and_measured_feedback(self, make_robot):
        factory, mocks = make_robot
        robot = arm_for_publish(factory(end_effector="brainco"), mocks)
        robot.sim_env = MagicMock()
        robot._lowstate = mocks["lowstate_msg"]
        robot.sim_env.get_hand_observation.return_value = {"hands.left.motor_0.pos": 0.24}
        robot.send_action({"hands.left.motor_0.pos": 0.5, "kLeftElbow.q": 0.3})
        assert robot.msg.motor_cmd[G1_29_JointIndex.kLeftElbow].q == 0.3
        assert robot.get_observation()["hands.left.motor_0.pos"] == 0.24
        assert robot.get_observation()["kLeftElbow.q"] == mocks["lowstate_msg"].motor_state[18].q

    @pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
    def test_invalid_hand_action_is_atomic(self, make_robot, value):
        factory, mocks = make_robot
        robot = arm_for_publish(factory(end_effector="brainco"), mocks)
        robot.sim_env = MagicMock()
        with pytest.raises(ValueError):
            robot.send_action({"kLeftElbow.q": 0.5, "hands.left.motor_0.pos": value})
        robot.sim_env.send_hand_action.assert_not_called()
        mocks["publisher_mock"].Write.assert_not_called()

    def test_no_hands_and_physical_guard(self, make_robot):
        factory, _ = make_robot
        robot = factory(end_effector="dummy")
        assert not robot._hands_ft
        with pytest.raises(ValueError, match="not supported"):
            robot.send_action({"hands.left.motor_0.pos": 0.5})
        with pytest.raises(ValueError, match="simulation-only"):
            UnitreeG1Config(end_effector="brainco", is_simulation=False)

    def test_old_hub_interface_is_rejected_and_closed(self, make_robot):
        factory, _ = make_robot
        robot = factory(end_effector="brainco")
        env = MagicMock(brainco_interface_version=None)
        with (
            patch("lerobot.envs.make_env", return_value={"hub_env": {0: SimpleNamespace(envs=[env])}}),
            pytest.raises(RuntimeError, match="interface v1"),
        ):
            robot.connect()
        env.close.assert_called_once()
        assert robot.sim_env is None
        assert robot.subscribe_thread is None

    def test_config_roundtrip_and_hub_revision(self):
        import draccus

        config = draccus.parse(
            UnitreeG1Config,
            args=[
                "--end_effector=brainco",
                "--sim_publish_images=false",
                "--sim_onscreen=false",
                "--sim_hub_path=owner/simulator@revision",
            ],
        )
        assert config.sim_env.end_effector.value == "brainco"
        assert config.sim_env.hub_path == "owner/simulator@revision"
        assert not config.sim_env.onscreen
