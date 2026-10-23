"""Recording integrity and kinematic projection tests, without hardware."""

import json
import threading
from unittest.mock import MagicMock

import pytest

from lerobot.utils.catch_up_analysis import project_catch_up
from lerobot.utils.catch_up_monitor import JointTrackingErrorMonitor
from lerobot.utils.joint_telemetry import JointTelemetryWriter


def sample(index, t, positions, targets):
    return {
        "type": "sample",
        "sample_index": index,
        "monotonic_s": t,
        "follower_position": positions,
        "leader_target": targets,
    }


@pytest.mark.parametrize("direction", [1, -1])
def test_constant_speed_projection_uses_follower_velocity_and_signed_error(direction):
    history = [sample(1, 0, {"j": 0}, {"j": 99}), sample(2, 0.1, {"j": direction * 2}, {"j": direction * 12})]
    result = project_catch_up(history)
    assert result["joints"]["j"]["follower_velocity"] == direction * 20
    assert result["arm_catch_up_time_s"] == pytest.approx(9.5 / 20)


@pytest.mark.parametrize(("previous", "current", "status"), [(2, 2, "stationary"), (1, 2, "moving_away")])
def test_nonclosing_motion_has_no_finite_eta(previous, current, status):
    result = project_catch_up(
        [sample(1, 0, {"j": previous}, {"j": 0}), sample(2, 0.1, {"j": current}, {"j": 0})]
    )
    assert result["joints"]["j"]["status"] == status
    assert result["arm_catch_up_time_s"] is None


def test_slowest_joint_can_differ_from_largest_position_error():
    result = project_catch_up(
        [
            sample(1, 0, {"a": 0, "b": 0}, {"a": 20, "b": 6}),
            sample(2, 0.1, {"a": 10, "b": 1}, {"a": 20, "b": 6}),
        ]
    )
    assert result["limiting_joint"] == "b"
    assert result["arm_catch_up_time_s"] == pytest.approx(0.45)


def test_warmup_and_stale_feedback_are_not_zero_eta():
    first = sample(1, 0, {"j": 0}, {"j": 10})
    assert project_catch_up([first])["arm_catch_up_time_s"] is None
    later = sample(2, 0.5, {"j": 2}, {"j": 10})
    assert project_catch_up([first, later], velocity_window_s=1)["arm_catch_up_time_s"] is None


def test_missing_joint_prevents_arm_eta_and_tolerance_is_applied():
    row = sample(1, 0, {"a": 1, "b": None}, {"a": 1.4, "b": 5})
    result = project_catch_up([row])
    assert result["joints"]["a"]["catch_up_time_s"] == 0
    assert result["arm_catch_up_time_s"] is None


def test_raw_telemetry_preserves_sample_ids_targets_invalid_values_and_summary(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = JointTelemetryWriter(str(path), {"joint_units": "degrees"})
    notifier = MagicMock()
    monitor = JointTrackingErrorMonitor({"j": 2.0}, notifier, telemetry=writer)
    monitor.update({"j": float("nan")}, {"j": 3}, timestamp=0)
    status = monitor.update({"j": 1.0}, {"j": 4.0}, timestamp=0.1)
    assert status.sample_index == 2
    notifier.notify_latest.assert_called_once_with(status)
    monitor.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[1]["follower_position"]["j"] is None
    assert rows[2]["sample_index"] == 2
    assert rows[2]["leader_target"] == {"j": 4.0}
    assert rows[2]["follower_position"] == {"j": 1.0}
    assert rows[2]["warning_due"] is True
    assert rows[-1] == {"type": "summary", "submitted": 2, "written": 2, "dropped": 0}
    with pytest.raises(FileExistsError):
        JointTelemetryWriter(str(path), {})


def test_slow_disk_drops_explicitly_instead_of_blocking_control_loop(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original = JointTelemetryWriter._write

    def slow_write(self, row):
        if row.get("sample_index") == 1:
            entered.set()
            assert release.wait(timeout=2)
        original(self, row)

    monkeypatch.setattr(JointTelemetryWriter, "_write", slow_write)
    path = tmp_path / "slow.jsonl"
    writer = JointTelemetryWriter(str(path), {}, capacity=1)
    try:
        writer.submit({"type": "sample", "sample_index": 1})
        assert entered.wait(timeout=2)
        writer.submit({"type": "sample", "sample_index": 2})
        writer.submit({"type": "sample", "sample_index": 3})
        assert writer.dropped == 1
    finally:
        release.set()
        writer.close()
    summary = json.loads(path.read_text().splitlines()[-1])
    assert summary == {"type": "summary", "submitted": 3, "written": 2, "dropped": 1}


def test_disk_write_failure_is_reported_without_stopping_control(tmp_path, monkeypatch):
    def fail(self, row):
        raise OSError("synthetic disk error")

    monkeypatch.setattr(JointTelemetryWriter, "_write", fail)
    writer = JointTelemetryWriter(str(tmp_path / "failed.jsonl"), {})
    writer.close()
    assert "synthetic disk error" in writer.error
    writer.submit({"type": "sample"})
    assert writer.dropped == 1


def test_spoken_notification_logs_the_corresponding_sample_id(caplog):
    from lerobot.utils.catch_up_monitor import JointTrackingStatus, SlowDownNotifier

    finished = threading.Event()
    notifier = SlowDownNotifier(play_sounds=True, speaker=lambda *args, **kwargs: finished.set())
    status = JointTrackingStatus(1.25, 2.0, "j", 4.0, 2.0, True, True, sample_index=17)
    try:
        notifier.notify_latest(status)
        assert finished.wait(timeout=2)
    finally:
        notifier.close()
    assert "slow down event=17 sample_monotonic=1.250000000" in caplog.text
