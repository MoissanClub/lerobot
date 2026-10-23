"""Offline constant-speed projections from recorded joint positions."""

from __future__ import annotations

import math
from typing import Any


def project_catch_up(
    samples: list[dict[str, Any]],
    *,
    tolerance: float = 0.5,
    velocity_window_s: float = 0.1,
    max_gap_s: float = 0.25,
    speed_epsilon: float = 0.1,
) -> dict[str, Any]:
    """Freeze the latest target; project using signed follower speed, not leader speed.

    This is a counterfactual constant-speed estimate, not a motion/controller model.
    Invalid, stationary or moving-away joints outside tolerance make arm ETA unknown.
    """
    if not samples:
        raise ValueError("At least one sample is required")
    for value in (tolerance, velocity_window_s, max_gap_s, speed_epsilon):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Projection parameters must be finite and positive")
    current = samples[-1]
    now = current["monotonic_s"]
    result: dict[str, Any] = {"sample_index": current["sample_index"], "joints": {}}
    for joint, target in current["leader_target"].items():
        position = current["follower_position"].get(joint)
        row: dict[str, Any] = {
            "target": target,
            "position": position,
            "error": None,
            "follower_velocity": None,
            "closing_speed": None,
            "velocity_baseline_s": None,
            "catch_up_time_s": None,
            "status": "missing_sample",
        }
        result["joints"][joint] = row
        if target is None or position is None or not all(math.isfinite(x) for x in (target, position)):
            continue
        error = target - position
        row["error"] = error
        # Only use the contiguous tail with valid feedback and increasing timestamps.
        history = [current]
        for earlier in reversed(samples[:-1]):
            dt = history[0]["monotonic_s"] - earlier["monotonic_s"]
            value = earlier["follower_position"].get(joint)
            if dt <= 0 or dt > max_gap_s or value is None or not math.isfinite(value):
                break
            if now - earlier["monotonic_s"] > velocity_window_s + 1e-9:
                break
            history.insert(0, earlier)
        if len(history) >= 2:
            baseline = now - history[0]["monotonic_s"]
            velocity = (position - history[0]["follower_position"][joint]) / baseline
            closing_speed = velocity * (1 if error > 0 else -1)
            row.update(follower_velocity=velocity, closing_speed=closing_speed, velocity_baseline_s=baseline)
        if abs(error) <= tolerance:
            row.update(status="within_tolerance", catch_up_time_s=0.0)
        elif row["closing_speed"] is None:
            row["status"] = "insufficient_or_stale_velocity"
        elif abs(row["follower_velocity"]) <= speed_epsilon:
            row["status"] = "stationary"
        elif row["closing_speed"] <= 0:
            row["status"] = "moving_away"
        else:
            row.update(status="closing", catch_up_time_s=(abs(error) - tolerance) / row["closing_speed"])
    times = {joint: row["catch_up_time_s"] for joint, row in result["joints"].items()}
    valid = bool(times) and all(value is not None for value in times.values())
    result["arm_catch_up_time_s"] = max(times.values()) if valid else None
    result["limiting_joint"] = max(times, key=times.get) if valid else None
    result["status"] = "constant_speed_projection" if valid else "unavailable_nonconverging_or_missing_joint"
    return result
