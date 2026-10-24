"""Deterministic SO101 supervision and bounded command generation; no hardware I/O."""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    IDLE = "idle"
    LAGGING = "lagging"
    TRACKING = "tracking"
    ALIGN = "awaiting_alignment"
    RETURN = "returning_to_rest"
    HALTED = "halted"
    FAULT = "fault"
    SETUP = "setup_hold"


@dataclass
class Config:
    alignment: dict
    lag: dict
    speed: dict
    acceleration: dict
    dwell_s: float = 0.7
    lag_dwell_s: float = 0.3
    progress_window_s: float = 1.0
    nonprogress_s: float = 3.0
    min_progress: float = 0.5
    boundary_tolerance: float = 0.5
    direction_tolerance: float = 0.1
    max_feedback_gap_s: float = 0.25
    return_timeout_s: float = 30.0

    def validate(self, joints):
        for name in ("alignment", "lag", "speed", "acceleration"):
            values = getattr(self, name)
            if set(values) != set(joints) or any(not math.isfinite(v) or v <= 0 for v in values.values()):
                raise ValueError(f"Invalid per-joint {name}")
        if any(self.lag[j] <= self.alignment[j] for j in joints):
            raise ValueError("Lag thresholds must exceed alignment tolerances")
        for name in (
            "dwell_s",
            "lag_dwell_s",
            "progress_window_s",
            "nonprogress_s",
            "min_progress",
            "boundary_tolerance",
            "direction_tolerance",
            "max_feedback_gap_s",
            "return_timeout_s",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid {name}")


@dataclass
class Decision:
    state: str
    previous: str
    reason: str
    command: dict | None
    errors: dict = field(default_factory=dict)
    boundary: str | None = None


class Controller:
    """One instance per session. Times are monotonic; fault/halt latch until restart."""

    def __init__(self, rest, bounds, config):
        self.rest, self.bounds, self.cfg = dict(rest), bounds, config
        config.validate(rest)
        if set(bounds) != set(rest):
            raise ValueError("Bounds/rest joints differ")
        for j, value in rest.items():
            low, high = bounds[j]
            if (
                not all(math.isfinite(v) for v in (low, high, value))
                or not low <= value <= high
                or low >= high
            ):
                raise ValueError(f"Invalid rest/bounds for {j}")
        self.state = State.SETUP
        self.last = None
        self.entered = 0.0
        self.timers = {}
        self.history = deque()
        self.command = None
        self.velocity = dict.fromkeys(rest, 0.0)
        self.hold = None

    def _sustained(self, key, condition, now, duration):
        if not condition:
            self.timers.pop(key, None)
            return False
        self.timers.setdefault(key, now)
        return now - self.timers[key] >= duration

    def step(self, now, leader, follower, events=(), fault=None):
        previous = self.state
        reason, boundary = "", None
        valid = set(follower) == set(self.rest) == set(leader) and all(
            math.isfinite(v) for p in (leader, follower) for v in p.values()
        )
        gap = 0 if self.last is None else now - self.last
        stale = self.last is not None and (gap <= 0 or gap > self.cfg.max_feedback_gap_s)
        self.last = now
        errors = {j: leader[j] - follower[j] for j in self.rest} if valid else {}
        aligned = valid and all(abs(errors[j]) <= self.cfg.alignment[j] for j in self.rest)
        rest_ok = valid and all(abs(self.rest[j] - follower[j]) <= self.cfg.alignment[j] for j in self.rest)
        if self.state in (State.FAULT, State.HALTED):
            return Decision(self.state.value, previous.value, "latched", None, errors)
        if fault or not valid or stale or not math.isfinite(now):
            self.state, reason = State.FAULT, fault or "invalid_or_stale_feedback"
        elif "halt" in events:
            self.state, reason = State.HALTED, "keyboard_halt"
        else:
            if self.command is None:
                self.command = dict(follower)
                self.hold = dict(follower)
            for j, (low, high) in self.bounds.items():
                q, t = follower[j], leader[j]
                if q >= high - self.cfg.boundary_tolerance and t > q + self.cfg.direction_tolerance:
                    boundary = f"upper:{j}"
                    break
                if q <= low + self.cfg.boundary_tolerance and t < q - self.cfg.direction_tolerance:
                    boundary = f"lower:{j}"
                    break
            self.history.append((now, dict(follower)))
            while len(self.history) > 1 and self.history[1][0] <= now - self.cfg.progress_window_s:
                self.history.popleft()
            active = self.state in (State.TRACKING, State.LAGGING)
            if "stop" in events and self.state != State.RETURN:
                self.state, reason = State.RETURN, "keyboard_stop"
            elif active and (boundary or "disengage" in events):
                self.state, reason = State.ALIGN, boundary or "keyboard_disengage"
            elif self.state == State.SETUP:
                if rest_ok:
                    self.state, reason = State.IDLE, "already_at_rest"
            elif self.state == State.IDLE and "start" in events:
                self.state, reason = State.LAGGING, "keyboard_start"
            elif self.state == State.TRACKING:
                excessive = any(abs(errors[j]) > self.cfg.lag[j] for j in self.rest)
                if self._sustained("lag", excessive, now, self.cfg.lag_dwell_s):
                    self.state, reason = State.LAGGING, "sustained_error"
            elif self.state == State.LAGGING:
                if self._sustained("aligned", aligned, now, self.cfg.dwell_s):
                    self.state, reason = State.TRACKING, "alignment_dwell"
                else:
                    old_time, old = self.history[0]
                    ready = now - old_time >= self.cfg.progress_window_s * 0.95
                    stalled = []
                    for j in self.rest:
                        signed_progress = (follower[j] - old[j]) * (1 if errors[j] > 0 else -1)
                        if (
                            abs(errors[j]) > self.cfg.lag[j]
                            and ready
                            and signed_progress < self.cfg.min_progress
                        ):
                            stalled.append(j)
                    if self._sustained("stalled", bool(stalled), now, self.cfg.nonprogress_s):
                        self.state, reason = State.ALIGN, "nonprogress:" + ",".join(stalled)
            elif self.state == State.ALIGN:
                if self._sustained("aligned", aligned and boundary is None, now, self.cfg.dwell_s):
                    self.state, reason = State.TRACKING, "pickup_aligned"
            elif self.state == State.RETURN:
                if self._sustained("rest", rest_ok, now, self.cfg.dwell_s):
                    self.state, reason = State.IDLE, "rest_reached"
                elif now - self.entered > self.cfg.return_timeout_s:
                    self.state, reason = State.FAULT, "return_timeout"
                else:
                    old_time, old = self.history[0]
                    stuck = now - old_time >= self.cfg.progress_window_s * 0.95 and any(
                        abs(self.rest[j] - follower[j]) > self.cfg.alignment[j]
                        and (follower[j] - old[j]) * (1 if self.rest[j] > follower[j] else -1)
                        < self.cfg.min_progress
                        for j in self.rest
                    )
                    if self._sustained("return_stall", stuck, now, self.cfg.nonprogress_s):
                        self.state, reason = State.FAULT, "return_nonprogress"
        if self.state != previous:
            self.entered = now
            self.timers.clear()
            self.history.clear()
            if valid and self.state in (State.ALIGN, State.HALTED, State.FAULT):
                self.hold = dict(follower)
                self.command = dict(follower)
                self.velocity = dict.fromkeys(self.rest, 0.0)
            if valid and previous == State.ALIGN and self.state == State.TRACKING:
                self.command = dict(follower)
                self.velocity = dict.fromkeys(self.rest, 0.0)
        if self.state in (State.HALTED, State.FAULT):
            # One measured-pose hold on entry only, if feedback is fresh; no blind recovery writes.
            command = dict(follower) if valid and not stale and self.state != previous else None
        elif self.state in (State.SETUP, State.ALIGN):
            command = dict(self.hold)
        else:
            target = self.rest if self.state in (State.IDLE, State.RETURN) else leader
            dt = 0 if previous == State.ALIGN else max(0, min(gap, self.cfg.max_feedback_gap_s))
            for j in self.rest:
                low, high = self.bounds[j]
                desired = min(high, max(low, target[j]))
                delta = desired - self.command[j]
                speed = min(self.cfg.speed[j], math.sqrt(2 * self.cfg.acceleration[j] * abs(delta)))
                wanted = math.copysign(speed, delta) if delta else 0.0
                change = self.cfg.acceleration[j] * dt
                v = max(self.velocity[j] - change, min(self.velocity[j] + change, wanted))
                move = v * dt
                if move * delta > 0 and abs(move) >= abs(delta) and abs(v) <= change:
                    move, v = delta, 0.0
                self.command[j] = min(high, max(low, self.command[j] + move))
                self.velocity[j] = v
            command = dict(self.command)
        return Decision(self.state.value, previous.value, reason, command, errors, boundary)
