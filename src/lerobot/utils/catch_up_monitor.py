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

"""Joint-tracking warnings for leader/follower control loops.

The monitor observes an already-read robot observation and the final processed
robot target.  It never reads hardware or changes the command.  Text and speech
I/O live on a background worker so a warning cannot consume control-loop time.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.joint_telemetry import JointTelemetryWriter, finite_joint_values
from lerobot.utils.utils import say

logger = logging.getLogger(__name__)


@dataclass
class JointTrackingWarningConfig:
    """Configuration for an operator warning based on normalized joint error.

    ``joint_error_threshold`` is in the robot action space.  SO arms normally
    use degrees (or calibrated range units when ``use_degrees=false``).
    ``joint_error_thresholds`` can override it for individual feature keys.
    """

    enabled: bool = True
    joint_error_threshold: float = 10.0
    joint_error_thresholds: dict[str, float] = field(default_factory=dict)
    recovery_ratio: float = 0.8
    warning_interval_s: float = 5.0
    # Optional raw joint samples for offline velocity/catch-up analysis.
    telemetry_path: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.joint_error_threshold) or self.joint_error_threshold <= 0:
            raise ValueError("joint_error_threshold must be finite and > 0")
        if not 0 <= self.recovery_ratio < 1:
            raise ValueError("recovery_ratio must be in [0, 1)")
        if not math.isfinite(self.warning_interval_s) or self.warning_interval_s <= 0:
            raise ValueError("warning_interval_s must be finite and > 0")
        invalid = {
            key: value
            for key, value in self.joint_error_thresholds.items()
            if not math.isfinite(value) or value <= 0
        }
        if invalid:
            raise ValueError(f"joint_error_thresholds must contain finite positive values: {invalid}")


@dataclass(frozen=True)
class JointTrackingStatus:
    timestamp: float
    max_error_ratio: float
    worst_joint: str
    worst_error: float
    threshold: float
    is_slow: bool
    warning_due: bool
    sample_index: int | None = None


class ControlLoopMonitor(Protocol):
    """Observational hook called with one control tick's state and target."""

    def update(
        self,
        observation: RobotObservation,
        target: RobotAction,
        timestamp: float | None = None,
    ) -> JointTrackingStatus | None: ...

    def close(self) -> None: ...


class _SlowDownNotificationSink(Protocol):
    def notify_latest(self, status: JointTrackingStatus) -> None: ...

    def close(self) -> None: ...


class SlowDownNotifier:
    """Serialize operator prompts on a latest-only background worker.

    One notification may be active and one may be pending.  While speech is in
    progress, a new status replaces the pending one instead of growing a FIFO.
    """

    def __init__(self, play_sounds: bool = True, speaker: Any | None = None) -> None:
        self.play_sounds = play_sounds
        self._speaker = say if speaker is None else speaker
        self._condition = threading.Condition()
        self._latest: JointTrackingStatus | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="lerobot-slow-down-notifier",
            daemon=True,
        )
        self._thread.start()

    def notify_latest(self, status: JointTrackingStatus) -> None:
        """Publish *status* without waiting; replace any pending notification."""
        with self._condition:
            if self._closed:
                return
            self._latest = status
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._latest is not None)
                if self._closed:
                    return
                status = self._latest
                self._latest = None

            logger.warning(
                "slow down event=%s sample_monotonic=%.9f joint=%s error=%.3f threshold=%.3f",
                status.sample_index,
                status.timestamp,
                status.worst_joint,
                status.worst_error,
                status.threshold,
            )
            if self.play_sounds:
                try:
                    # Blocking here prevents the platform speech service from accumulating
                    # prompts.  This worker, never the control loop, does the waiting.
                    self._speaker("Slow down", blocking=True)
                except Exception:
                    logger.exception("Slow-down voice prompt failed")

    def close(self) -> None:
        """Discard pending work and stop the notifier worker."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._latest = None
            self._condition.notify()
        self._thread.join(timeout=6.0)
        if self._thread.is_alive():
            logger.warning("Slow-down notifier did not stop within 6 seconds")


class JointTrackingErrorMonitor:
    """Warn when any monitored joint exceeds its allowed tracking error."""

    def __init__(
        self,
        thresholds: dict[str, float],
        notifier: _SlowDownNotificationSink,
        *,
        recovery_ratio: float = 0.8,
        warning_interval_s: float = 5.0,
        telemetry: JointTelemetryWriter | None = None,
    ) -> None:
        if not thresholds:
            raise ValueError("At least one joint tracking threshold is required")
        if any(not math.isfinite(value) or value <= 0 for value in thresholds.values()):
            raise ValueError("Joint tracking thresholds must be finite and > 0")
        if not 0 <= recovery_ratio < 1:
            raise ValueError("recovery_ratio must be in [0, 1)")
        if not math.isfinite(warning_interval_s) or warning_interval_s <= 0:
            raise ValueError("warning_interval_s must be finite and > 0")

        self.thresholds = dict(thresholds)
        self.notifier = notifier
        self.recovery_ratio = recovery_ratio
        self.warning_interval_s = warning_interval_s
        self._is_slow = False
        self._last_warning_at: float | None = None
        self._sample_index = 0
        self.telemetry = telemetry

    def update(
        self,
        observation: RobotObservation,
        target: RobotAction,
        timestamp: float | None = None,
    ) -> JointTrackingStatus | None:
        now = time.monotonic() if timestamp is None else timestamp
        self._sample_index += 1
        errors: list[tuple[float, str, float, float]] = []
        for joint, threshold in self.thresholds.items():
            if joint not in observation or joint not in target:
                continue
            observed = observation[joint]
            requested = target[joint]
            try:
                error = abs(float(requested) - float(observed))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(error):
                continue
            errors.append((error / threshold, joint, error, threshold))

        if not errors:
            self._log_sample(observation, target, now, None)
            return None

        max_ratio, worst_joint, worst_error, threshold = max(errors)
        if not self._is_slow and max_ratio > 1.0:
            self._is_slow = True
            self._last_warning_at = None
        elif self._is_slow and max_ratio < self.recovery_ratio:
            self._is_slow = False
            self._last_warning_at = None

        warning_due = self._is_slow and (
            self._last_warning_at is None or now - self._last_warning_at >= self.warning_interval_s
        )
        status = JointTrackingStatus(
            timestamp=now,
            max_error_ratio=max_ratio,
            worst_joint=worst_joint,
            worst_error=worst_error,
            threshold=threshold,
            is_slow=self._is_slow,
            warning_due=warning_due,
            sample_index=self._sample_index,
        )
        self._log_sample(observation, target, now, status)
        if warning_due:
            self._last_warning_at = now
            self.notifier.notify_latest(status)
        return status

    def _log_sample(self, observation, target, now, status):
        if self.telemetry is None:
            return
        self.telemetry.submit(
            {
                "type": "sample",
                "sample_index": self._sample_index,
                "monotonic_s": now,
                "wall_time_ns": time.time_ns(),
                "follower_position": finite_joint_values(observation, self.thresholds),
                "leader_target": finite_joint_values(target, self.thresholds),
                "warning_due": status.warning_due if status is not None else False,
                "is_slow": status.is_slow if status is not None else None,
                "worst_joint": status.worst_joint if status is not None else None,
                "worst_error": status.worst_error if status is not None else None,
            }
        )

    def close(self) -> None:
        self.notifier.close()
        if self.telemetry is not None:
            self.telemetry.close()


def make_so_arm_tracking_monitor(
    robot: Any,
    teleop: Any | None,
    config: JointTrackingWarningConfig,
    *,
    play_sounds: bool,
) -> JointTrackingErrorMonitor | None:
    """Build the monitor for the shared SO-100/SO-101 runtime family."""
    if not config.enabled or robot.name != "so_follower" or getattr(teleop, "name", None) != "so_leader":
        return None

    position_keys = [key for key in robot.action_features if key.endswith(".pos") and key != "gripper.pos"]
    thresholds = {
        key: config.joint_error_thresholds.get(key, config.joint_error_threshold) for key in position_keys
    }
    if not thresholds:
        return None
    telemetry = None
    if config.telemetry_path:
        units = (
            "degrees" if getattr(getattr(robot, "config", None), "use_degrees", True) else "normalized_range"
        )
        telemetry = JointTelemetryWriter(
            config.telemetry_path,
            {
                "joint_units": units,
                "thresholds": thresholds,
                "gripper_excluded": True,
                "max_relative_target": getattr(getattr(robot, "config", None), "max_relative_target", None),
                "target_semantics": "processed leader target before robot-side clipping",
                "sample_time_semantics": "software timestamp after observation/action reads, before motor send; not simultaneous hardware sampling",
            },
        )
    notifier = SlowDownNotifier(play_sounds=play_sounds)
    return JointTrackingErrorMonitor(
        thresholds,
        notifier,
        recovery_ratio=config.recovery_ratio,
        warning_interval_s=config.warning_interval_s,
        telemetry=telemetry,
    )
