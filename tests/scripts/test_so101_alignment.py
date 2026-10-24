"""Alignment startup checks use mocks only, with no serial access."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.scripts import lerobot_align_so101 as alignment


def pose(value):
    return dict.fromkeys(alignment.JOINTS, value)


def test_goal_ramp_limits_commands_and_stops_at_fixed_snapshot():
    target = pose(10)
    target["gripper"] = 50
    assert alignment.ramp_goal(pose(0), target, 1, 5, 10) == {**pose(5), "gripper": 10}
    assert alignment.ramp_goal(pose(0), target, 100, 5, 10) == target
    assert alignment.ramp_goal(pose(0), pose(-10), 1, 5, 10)["elbow_flex"] == -5


def test_alignment_requires_every_joint_and_current_leader_to_match():
    target = pose(0)
    actual = pose(0)
    actual["elbow_flex"] = 3.6
    assert alignment.alignment_errors(target, target, actual, 1, 3)[1:] == (False, True)
    moved_leader = pose(0)
    moved_leader["shoulder_pan"] = 2
    assert alignment.alignment_errors(target, moved_leader, target, 1, 3)[1:] == (True, False)


def test_seed_current_goal_before_enabling_follower_torque():
    calls = []
    bus = MagicMock()
    bus.connect.side_effect = lambda: calls.append("connect")
    bus.disable_torque.side_effect = lambda: calls.append("disable")
    bus.sync_read.side_effect = lambda name: (calls.append("read") or pose(2))
    bus.sync_write.side_effect = lambda name, value: calls.append((name, value))
    robot = SimpleNamespace(bus=bus, is_calibrated=True, configure=lambda: calls.append("configure_enable"))
    assert alignment.connect_follower_holding_present_pose(robot) == pose(2)
    assert calls == ["connect", "disable", "read", ("Goal_Position", pose(2)), "configure_enable"]


def test_calibration_mismatch_never_enables_follower():
    robot = MagicMock()
    robot.is_calibrated = False
    with pytest.raises(RuntimeError, match="calibration differs"):
        alignment.connect_follower_holding_present_pose(robot)
    robot.configure.assert_not_called()
    robot.bus.sync_write.assert_not_called()


def test_nonfinite_feedback_is_rejected():
    device = SimpleNamespace(bus=SimpleNamespace(sync_read=lambda name: pose(float("nan"))))
    with pytest.raises(RuntimeError, match="non-finite"):
        alignment.read_positions(device)


@pytest.mark.parametrize("mode", ["tracking", "residual", "leader_moved"])
def test_full_alignment_session_freezes_leader_goal_logs_and_disconnects(tmp_path, monkeypatch, mode):
    for arm in ("leader", "follower"):
        (tmp_path / (arm + ".json")).write_text("{}")
    cfg = {
        "robot": {
            "type": "so101_follower",
            "port": "fake",
            "id": "follower",
            "calibration_dir": str(tmp_path),
        },
        "teleop": {"type": "so101_leader", "port": "fake", "id": "leader", "calibration_dir": str(tmp_path)},
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    created = []

    class Bus:
        def __init__(self, value):
            self.position = pose(value)
            self.goal = pose(value)
            self.is_connected = False

        def connect(self):
            self.is_connected = True

        def disconnect(self, **kwargs):
            self.is_connected = False

        def disable_torque(self):
            pass

        def sync_read(self, name, **kwargs):
            if name == "Present_Position":
                return self.position.copy()
            if name == "Goal_Position":
                return self.goal.copy()
            return dict.fromkeys(alignment.JOINTS, 1)

        def sync_write(self, name, value):
            self.goal = value.copy()

    class Device:
        def __init__(self, config):
            self.bus = Bus(0 if config.id == "follower" else 1)
            self.is_calibrated = True
            self.calibration = {j: SimpleNamespace(range_min=0, range_max=4095) for j in alignment.JOINTS}
            created.append(self)

        def configure(self):
            pass

        def send_action(self, action):
            self.bus.position = {j.removesuffix(".pos"): value for j, value in action.items()}
            self.bus.goal = self.bus.position.copy()
            if mode == "residual":
                self.bus.position["elbow_flex"] += 3.6
            if mode == "leader_moved":
                created[1].bus.position["elbow_flex"] = 3

    clock = [0.0]

    def monotonic():
        clock[0] += 0.025
        return clock[0]

    monkeypatch.setattr(alignment, "SO101Follower", Device)
    monkeypatch.setattr(alignment, "SO101Leader", Device)
    monkeypatch.setattr(alignment.time, "monotonic", monotonic)
    monkeypatch.setattr(alignment.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(alignment.select, "select", lambda *args: ([], [], []))
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr(
        "sys.argv",
        [
            "align",
            "--config",
            str(config),
            "--timeout-s",
            "0.5",
            "--settle-s",
            "0.05",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )
    if mode == "tracking":
        alignment.main()
    else:
        with pytest.raises(SystemExit) as exc:
            alignment.main()
        assert exc.value.code == 2
    logs = list((tmp_path / "logs").glob("*.jsonl"))
    rows = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert rows[-1] == {
        "type": "summary",
        "outcome": "timeout_aligned" if mode == "tracking" else "timeout_not_aligned",
        "aligned_seen": mode == "tracking",
    }
    assert all(not device.bus.is_connected for device in created)
    assert next(r for r in rows if r["type"] == "initial_snapshot")["target"] == pose(1)
    assert next(r for r in rows if r["type"] == "initial_snapshot")["motor_goal_readback"] == pose(0)
    snapshot = next(r for r in rows if r["type"] == "motor_configuration_snapshot")
    assert snapshot["follower"]["Torque_Enable"]["values"]["elbow_flex"] == 1
    assert snapshot["leader"]["Torque_Enable"]["error"] is None
    assert all(r["commanded_goal"]["elbow_flex"] <= 1 for r in rows if r["type"] == "sample")


def test_holding_gripper_does_not_require_leader_gripper_alignment():
    target = pose(0)
    live = pose(0)
    live["gripper"] = 100
    assert alignment.alignment_errors(target, live, target, 1, 3, hold_gripper=True)[1:] == (True, True)
    assert alignment.alignment_errors(target, live, target, 1, 3, hold_gripper=False)[1:] == (True, False)
