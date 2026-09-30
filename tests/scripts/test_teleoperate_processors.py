# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
import json
from unittest.mock import Mock

import pytest

from lerobot.scripts import lerobot_teleoperate as cli


@pytest.mark.parametrize("saved", [False, True])
def test_processor_loading_and_robot_first_cleanup(tmp_path, monkeypatch, saved):
    events = []
    teleop, robot = Mock(), Mock()
    teleop.disconnect.side_effect = lambda: events.append("teleop")
    robot.disconnect.side_effect = lambda: events.append("robot")
    monkeypatch.setattr(cli, "make_teleoperator_from_config", lambda _: teleop)
    monkeypatch.setattr(cli, "make_robot_from_config", lambda _: robot)
    path = tmp_path / "processor.json"
    path.write_text(
        json.dumps({"steps": [{"class": "lerobot.processor.pipeline.IdentityProcessorStep", "config": {}}]})
    )
    load = Mock(wraps=cli.RobotProcessorPipeline.from_pretrained)
    monkeypatch.setattr(cli.RobotProcessorPipeline, "from_pretrained", load)

    def loop(**kwargs):
        assert kwargs["teleop_action_processor"](({"joint": 0.2}, {"joint": 0.1})) == {"joint": 0.2}

    monkeypatch.setattr(cli, "teleop_loop", loop)
    cfg = cli.TeleoperateConfig(
        teleop=None, robot=None, teleop_action_processor_path=str(path) if saved else None
    )
    cli.teleoperate.__wrapped__(cfg)
    assert load.call_count == int(saved)
    assert events == ["robot", "teleop"]


def test_robot_startup_failure_closes_teleoperator(monkeypatch):
    teleop, robot = Mock(), Mock()
    robot.connect.side_effect = RuntimeError("startup failed")
    monkeypatch.setattr(cli, "make_teleoperator_from_config", lambda _: teleop)
    monkeypatch.setattr(cli, "make_robot_from_config", lambda _: robot)
    with pytest.raises(RuntimeError, match="startup failed"):
        cli.teleoperate.__wrapped__(cli.TeleoperateConfig(teleop=None, robot=None))
    teleop.disconnect.assert_called_once()
