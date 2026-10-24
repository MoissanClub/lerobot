"""Bounded, timestamped pre-transition register history; serial owner thread only."""

from collections import deque

from lerobot.utils.so101_motor_diagnostics import MotorDiagnostics


class MotorStateHistory:
    def __init__(self, bus, window_s=3.0, interval_s=0.05, clock=None):
        self.window_s = window_s
        self.diagnostics = MotorDiagnostics(bus, interval_s=interval_s, clock=clock)
        self.records = deque(maxlen=512)
        self.started = self.diagnostics.clock()

    def poll(self, now):
        record = self.diagnostics.poll(now)
        if record is not None:
            self.records.append(record)
        self._prune(now)
        return record

    def _prune(self, now):
        while self.records and self.records[0]["read_started_monotonic_s"] < now - self.window_s:
            self.records.popleft()

    def snapshot(self, transition_time):
        self._prune(transition_time)
        records = [r for r in self.records if r["read_finished_monotonic_s"] <= transition_time]
        return {
            "window_s": self.window_s,
            "window_start_monotonic_s": transition_time - self.window_s,
            "transition_monotonic_s": transition_time,
            "monitor_started_monotonic_s": self.started,
            "startup_truncated": transition_time - self.started < self.window_s,
            "sampling": "one_register_per_eligible_cycle_not_simultaneous",
            "units": "raw_register_units_after_driver_sign_decode_not_measured_torque",
            "records": records,
        }
