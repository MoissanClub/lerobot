"""Intel backport integration checks without motor or camera connections."""

from unittest.mock import MagicMock

import pytest

from lerobot.scripts.lerobot_record import record_loop
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.teleoperators import Teleoperator


@pytest.mark.parametrize("recording", [False, True])
@pytest.mark.parametrize("monitor_enabled", [False, True])
def test_loop_observes_existing_state_and_final_target_without_changing_action(recording, monitor_enabled):
    robot = MagicMock()
    robot.action_features = {"shoulder_pan.pos": float}
    observation = {"shoulder_pan.pos": 1.0}
    robot.get_observation.return_value = observation
    teleop = MagicMock(spec=Teleoperator)
    teleop.get_action.return_value = {"shoulder_pan.pos": 3.0}
    monitor = MagicMock() if monitor_enabled else None

    def process_action(transition):
        action, _ = transition
        return {key: value + 1 for key, value in action.items()}

    kwargs = {
        "robot": robot,
        "teleop": teleop,
        "fps": 100000,
        "teleop_action_processor": lambda pair: pair[0],
        "robot_action_processor": process_action,
        "robot_observation_processor": lambda obs: obs,
        "catch_up_monitor": monitor,
    }
    if recording:
        record_loop(**kwargs, events={"exit_early": False}, control_time_s=0.000001)
    else:
        teleop_loop(**kwargs, duration=0)
    robot.get_observation.assert_called_once_with()
    teleop.get_action.assert_called_once_with()
    robot.send_action.assert_called_once_with({"shoulder_pan.pos": 4.0})
    if monitor is not None:
        monitor.update.assert_called_once_with(observation, {"shoulder_pan.pos": 4.0})
