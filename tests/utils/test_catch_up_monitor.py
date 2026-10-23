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

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.utils.catch_up_monitor import (
    JointTrackingErrorMonitor,
    JointTrackingStatus,
    JointTrackingWarningConfig,
    SlowDownNotifier,
    make_so_arm_tracking_monitor,
)


class _NotifierSpy:
    def __init__(self) -> None:
        self.statuses: list[JointTrackingStatus] = []
        self.closed = False

    def notify_latest(self, status: JointTrackingStatus) -> None:
        self.statuses.append(status)

    def close(self) -> None:
        self.closed = True


def test_joint_tracking_monitor_threshold_rate_limit_and_recovery() -> None:
    notifier = _NotifierSpy()
    monitor = JointTrackingErrorMonitor(
        {"shoulder.pos": 10.0, "elbow.pos": 20.0},
        notifier,
        recovery_ratio=0.8,
        warning_interval_s=5.0,
    )

    # The threshold is strict: exactly one threshold unit is not slow.
    status = monitor.update(
        {"shoulder.pos": 0.0, "elbow.pos": 0.0},
        {"shoulder.pos": 10.0, "elbow.pos": 10.0},
        timestamp=0.0,
    )
    assert status is not None and not status.is_slow and not status.warning_due

    status = monitor.update(
        {"shoulder.pos": 0.0, "elbow.pos": 0.0},
        {"shoulder.pos": 11.0, "elbow.pos": 30.0},
        timestamp=1.0,
    )
    assert status is not None
    assert status.is_slow and status.warning_due
    assert status.worst_joint == "elbow.pos"
    assert status.max_error_ratio == pytest.approx(1.5)

    # Remaining above the threshold does not warn again before five seconds.
    status = monitor.update({"shoulder.pos": 0.0}, {"shoulder.pos": 12.0}, timestamp=5.9)
    assert status is not None and status.is_slow and not status.warning_due
    status = monitor.update({"shoulder.pos": 0.0}, {"shoulder.pos": 12.0}, timestamp=6.0)
    assert status is not None and status.warning_due

    # Hysteresis keeps the slow state at 0.8 and clears only below it.
    status = monitor.update({"shoulder.pos": 0.0}, {"shoulder.pos": 8.0}, timestamp=7.0)
    assert status is not None and status.is_slow
    status = monitor.update({"shoulder.pos": 0.0}, {"shoulder.pos": 7.9}, timestamp=8.0)
    assert status is not None and not status.is_slow

    # Recovery rearms an immediate warning on the next threshold crossing.
    status = monitor.update({"shoulder.pos": 0.0}, {"shoulder.pos": 11.0}, timestamp=8.1)
    assert status is not None and status.warning_due
    assert len(notifier.statuses) == 3


def test_joint_tracking_monitor_ignores_missing_and_non_finite_values() -> None:
    notifier = _NotifierSpy()
    monitor = JointTrackingErrorMonitor({"shoulder.pos": 10.0}, notifier)

    assert monitor.update({}, {"shoulder.pos": 20.0}, timestamp=0.0) is None
    assert monitor.update({"shoulder.pos": float("nan")}, {"shoulder.pos": 20.0}, timestamp=0.0) is None
    assert notifier.statuses == []


def test_so_factory_excludes_gripper_and_non_so_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    notifier = MagicMock()
    monkeypatch.setattr(
        "lerobot.utils.catch_up_monitor.SlowDownNotifier",
        MagicMock(return_value=notifier),
    )
    robot = SimpleNamespace(
        name="so_follower",
        action_features={"shoulder_pan.pos": float, "elbow_flex.pos": float, "gripper.pos": float},
    )
    teleop = SimpleNamespace(name="so_leader")
    config = JointTrackingWarningConfig(
        joint_error_threshold=10.0,
        joint_error_thresholds={"elbow_flex.pos": 15.0},
    )

    monitor = make_so_arm_tracking_monitor(robot, teleop, config, play_sounds=False)
    assert monitor is not None
    assert monitor.thresholds == {"shoulder_pan.pos": 10.0, "elbow_flex.pos": 15.0}

    assert (
        make_so_arm_tracking_monitor(
            SimpleNamespace(name="unitree_g1", action_features={}),
            teleop,
            config,
            play_sounds=False,
        )
        is None
    )


def test_slow_down_notifier_keeps_only_latest_pending_warning() -> None:
    first_speaking = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    speaker_calls: list[tuple[str, bool]] = []

    def speaker(text: str, blocking: bool = False) -> None:
        speaker_calls.append((text, blocking))
        if len(speaker_calls) == 1:
            first_speaking.set()
            assert release_first.wait(timeout=2.0)
        else:
            second_finished.set()

    notifier = SlowDownNotifier(play_sounds=True, speaker=speaker)
    status = JointTrackingStatus(0.0, 2.0, "joint.pos", 20.0, 10.0, True, True)
    try:
        notifier.notify_latest(status)
        assert first_speaking.wait(timeout=2.0)

        # Both arrive while the first prompt is active; the second is superseded.
        notifier.notify_latest(status)
        notifier.notify_latest(status)
        release_first.set()
        assert second_finished.wait(timeout=2.0)
    finally:
        release_first.set()
        notifier.close()

    assert speaker_calls == [
        ("Slow down", True),
        ("Slow down", True),
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"joint_error_threshold": 0.0}, "joint_error_threshold"),
        ({"recovery_ratio": 1.0}, "recovery_ratio"),
        ({"warning_interval_s": 0.0}, "warning_interval_s"),
        ({"joint_error_thresholds": {"joint.pos": -1.0}}, "joint_error_thresholds"),
    ],
)
def test_joint_tracking_warning_config_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        JointTrackingWarningConfig(**kwargs)
