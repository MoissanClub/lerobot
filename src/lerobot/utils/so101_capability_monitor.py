"""Advisory gravity and effort/response monitoring. Never changes motor commands."""

import math
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def vector(text):
    return np.array([float(v) for v in text.split()])


class GravityModel:
    """URDF gravity torque via COM lever arms, no dynamics engine dependency."""

    def __init__(self, urdf, calibration, offsets_deg=None):
        root = ET.parse(urdf).getroot()
        self.links = {}
        self.edges = {}
        children = set()
        for link in root.findall("link"):
            inertial = link.find("inertial")
            self.links[link.attrib["name"]] = (
                (
                    float(inertial.find("mass").attrib["value"]),
                    vector(inertial.find("origin").get("xyz", "0 0 0")),
                )
                if inertial is not None
                else (0, np.zeros(3))
            )
        for joint in root.findall("joint"):
            parent, child = joint.find("parent").attrib["link"], joint.find("child").attrib["link"]
            origin = joint.find("origin")
            r, p, y = vector(origin.get("rpy", "0 0 0"))
            fixed_rotation = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
            axis = (
                vector(joint.find("axis").get("xyz", "0 0 1"))
                if joint.find("axis") is not None
                else np.array([0, 0, 1])
            )
            self.edges.setdefault(parent, []).append(
                (
                    child,
                    joint.attrib["name"],
                    joint.attrib["type"],
                    vector(origin.get("xyz", "0 0 0")),
                    fixed_rotation,
                    axis,
                )
            )
            children.add(child)
        self.root = (set(self.links) - children).pop()
        self.scale = dict.fromkeys(JOINTS, math.pi / 180)
        grip = calibration["gripper"]
        self.scale["gripper"] = (grip["range_max"] - grip["range_min"]) / 100 * 2 * math.pi / 4095
        self.offsets = {j: math.radians((offsets_deg or {}).get(j, 0)) for j in JOINTS}

    def torque(self, positions):
        if set(positions) != set(JOINTS) or not all(math.isfinite(v) for v in positions.values()):
            raise ValueError("Invalid gravity input")
        torque = dict.fromkeys(JOINTS, 0.0)

        def visit(link, orient_world, p, ancestors):
            mass, com = self.links[link]
            world_com = p + orient_world @ com
            force = np.array([0, 0, -9.81 * mass])
            for name, origin, axis in ancestors:
                torque[name] += float(axis @ np.cross(world_com - origin, force))
            for child, name, kind, xyz, orient, axis in self.edges.get(link, []):
                joint_r = orient_world @ orient
                joint_p = p + orient_world @ xyz
                active = ancestors
                if kind != "fixed":
                    active = [*ancestors, (name, joint_p, joint_r @ axis)]
                    joint_r = joint_r @ rotation(
                        axis, positions[name] * self.scale[name] + self.offsets[name]
                    )
                visit(child, joint_r, joint_p, active)

        visit(self.root, np.eye(3), np.zeros(3), [])
        return torque


@dataclass
class MonitorConfig:
    interval_s: float = 0.1
    effort_window_s: float = 2.0
    effort_dwell_s: float = 0.7
    max_effort_age_s: float = 0.8
    minimum_points: int = 3
    minimum_error_deg: float = 3.0
    error_growth_deg: float = 1.0
    load_flat_range: float = 40.0
    current_flat_range: float = 8.0
    minimum_load: float = 100.0
    minimum_current: float = 5.0
    progress_limit_deg: float = 0.5
    near_boundary_deg: float = 2.0
    calibrated_boundary_tolerance: float = 0.5
    utilization_warning: float = 0.8
    accuracy_limit_deg: float = 5.0
    minimum_command_motion_deg: float = 0.5

    def validate(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid monitor parameter {name}")
        if not isinstance(self.minimum_points, int) or self.minimum_points < 3:
            raise ValueError("minimum_points must be integer >=3")


def default_servo_models():
    # Rated continuous loads and exploratory stiffness, not identified joint limits.
    k_per_nm = (16 / 4) * (4095 / (2 * math.pi)) / 1000
    return {
        "12v": {
            "voltage_v": 12.0,
            "continuous_torque_nm": 10 * 0.0980665,
            "stiffness_nm_rad": 30 * 0.0980665 * k_per_nm,
        },
        "7p4v": {
            "voltage_v": 7.4,
            "continuous_torque_nm": 5 * 0.0980665,
            "stiffness_nm_rad": 19.5 * 0.0980665 * k_per_nm,
        },
    }


class CapabilityMonitor:
    def __init__(self, gravity, bounds, config=None, servo_models=None):
        self.gravity, self.bounds = gravity, bounds
        self.config = config or MonitorConfig()
        self.config.validate()
        self.models = servo_models or default_servo_models()
        for model in self.models.values():
            for key in ("voltage_v", "continuous_torque_nm", "stiffness_nm_rad"):
                if not math.isfinite(model[key]) or model[key] <= 0:
                    raise ValueError(f"Invalid servo model {key}")
        self.samples = deque(maxlen=512)
        self.effort = {
            j: {"Present_Load": deque(maxlen=128), "Present_Current": deque(maxlen=128)} for j in JOINTS
        }
        self.last_effort = {}
        self.unavailable = set()
        self.suspect_since = {}
        self.last_time = None

    def update(self, now, measured, requested, sent, latest, active=True):
        cfg = self.config
        if self.last_time is not None and (now <= self.last_time or now - self.last_time > 0.5):
            self.samples.clear()
            self.suspect_since.clear()
            for channels in self.effort.values():
                for records in channels.values():
                    records.clear()
        self.last_time = now
        if not active:
            self.samples.clear()
            self.suspect_since.clear()
            for channels in self.effort.values():
                for records in channels.values():
                    records.clear()
        actual_g = self.gravity.torque(measured)
        target_g = self.gravity.torque(requested)
        self.samples.append((now, dict(measured), dict(sent)))
        while self.samples and self.samples[0][0] < now - cfg.effort_window_s:
            self.samples.popleft()
        for reg in ("Present_Load", "Present_Current"):
            record = latest.get(reg)
            if record is None:
                continue
            stamp = record["read_finished_monotonic_s"]
            if self.last_effort.get(reg) == stamp:
                continue
            self.last_effort[reg] = stamp
            if record["error"] or record["values"] is None:
                self.unavailable.add(reg)
                continue
            self.unavailable.discard(reg)
            for j, value in record["values"].items():
                if j in self.effort:
                    self.effort[j][reg].append((stamp, abs(float(value))))
        joints = {}
        flagged = []
        for j in JOINTS:
            low, high = self.bounds[j]
            q = measured[j]
            target = requested[j]
            margin = min(q - low, high - q)
            geometry = margin <= cfg.near_boundary_deg
            outward_trend = active and (
                (q >= high - cfg.near_boundary_deg and target > q + 0.1)
                or (q <= low + cfg.near_boundary_deg and target < q - 0.1)
            )
            outward = (q >= high - cfg.calibrated_boundary_tolerance and target > q + 0.1) or (
                q <= low + cfg.calibrated_boundary_tolerance and target < q - 0.1
            )
            predictions = {}
            for name, model in self.models.items():
                required = max(abs(actual_g[j]), abs(target_g[j]))
                utilization = required / model["continuous_torque_nm"]
                error_deg = math.degrees(required / model["stiffness_nm_rad"])
                predictions[name] = {
                    "holding_utilization": utilization,
                    "predicted_static_error_deg": error_deg,
                    "torque_warning": utilization >= cfg.utilization_warning,
                    "accuracy_warning": error_deg >= cfg.accuracy_limit_deg,
                }
            pattern = self._pattern(j, now, measured, sent, active)
            if pattern["candidate"]:
                self.suspect_since.setdefault(j, now)
            else:
                self.suspect_since.pop(j, None)
            pattern["sustained"] = (
                j in self.suspect_since and now - self.suspect_since[j] >= cfg.effort_dwell_s
            )
            model_warning = any(p["torque_warning"] or p["accuracy_warning"] for p in predictions.values())
            reason = (
                "calibrated_position_boundary"
                if outward and active
                else "load_and_response_warning"
                if model_warning and pattern["sustained"]
                else "suspected_response_boundary"
                if pattern["sustained"]
                else "possible_load_boundary"
                if model_warning
                else "near_position_boundary"
                if outward_trend
                else "no_warning"
            )
            voltage = latest.get("Present_Voltage")
            voltage_v = None
            if (
                voltage
                and not voltage["error"]
                and voltage["values"]
                and 0 <= now - voltage["read_finished_monotonic_s"] <= cfg.max_effort_age_s
            ):
                raw = voltage["values"].get(j)
                voltage_v = raw * 0.1 if raw is not None else None
            joints[j] = {
                "gravity_torque_measured_nm": actual_g[j],
                "gravity_torque_requested_nm": target_g[j],
                "joint_margin": margin,
                "near_position_boundary": geometry,
                "requested_outward": outward_trend,
                "margin_units": "opening_units" if j == "gripper" else "degrees",
                "model_predictions": predictions,
                "model_disagreement": len(
                    {p["torque_warning"] or p["accuracy_warning"] for p in predictions.values()}
                )
                > 1,
                "effort_response": pattern,
                "prediction": reason,
                "voltage_v": voltage_v,
                "voltage_outside_nominal_models": voltage_v is not None
                and all(abs(voltage_v / m["voltage_v"] - 1) > 0.15 for m in self.models.values()),
            }
            if reason != "no_warning":
                flagged.append(j)
        return {
            "type": "capability_monitor",
            "monotonic_s": now,
            "advisory_only": True,
            "active_tracking": active,
            "prediction": "possible_operational_boundary" if flagged else "no_warning_not_proof_of_safety",
            "flagged_joints": flagged,
            "joints": joints,
            "assumptions": "nominal rated torque and exploratory stiffness; no dynamic torque, voltage derating, contact or compensation",
        }

    def _pattern(self, j, now, measured, sent, active):
        cfg = self.config
        series = self.samples
        direction = 1 if sent[j] > measured[j] else -1
        errors = [(c[j] - q[j]) * direction for _, q, c in series]
        ages = {}
        channels = {}
        for reg, records in self.effort[j].items():
            while records and records[0][0] < now - cfg.effort_window_s:
                records.popleft()
            ages[reg] = now - records[-1][0] if records else None
            fresh = bool(records) and reg not in self.unavailable and 0 <= ages[reg] <= cfg.max_effort_age_s
            span = records[-1][0] - records[0][0] if records else 0
            values = [v for _, v in records]
            threshold = cfg.load_flat_range if reg == "Present_Load" else cfg.current_flat_range
            floor = cfg.minimum_load if reg == "Present_Load" else cfg.minimum_current
            channels[reg] = {
                "fresh": fresh,
                "points": len(records),
                "span_s": span,
                "age_s": ages[reg],
                "range": max(values) - min(values) if values else None,
                "flat_nonzero": fresh
                and len(records) >= cfg.minimum_points
                and span >= cfg.effort_window_s * 0.5
                and max(values) - min(values) <= threshold
                and min(values) >= floor,
            }
        enough = len(series) >= 3 and now - series[0][0] >= cfg.effort_window_s * 0.8
        growth = errors[-1] - errors[0] if errors else 0
        progress = (measured[j] - series[0][1][j]) * direction
        demand_motion = (sent[j] - series[0][2][j]) * direction
        same_direction = all(e > 0 for e in errors)
        candidate = (
            active
            and enough
            and same_direction
            and abs(sent[j] - measured[j]) >= cfg.minimum_error_deg
            and growth >= cfg.error_growth_deg
            and progress < cfg.progress_limit_deg
            and demand_motion >= cfg.minimum_command_motion_deg
            and any(v["flat_nonzero"] for v in channels.values())
        )
        # Gripper error/progress thresholds are opening units, not degrees; gravity error is radians->degrees.
        return {
            "candidate": candidate,
            "error_growth": growth,
            "signed_progress": progress,
            "corrective_command_motion": demand_motion,
            "command_error": sent[j] - measured[j],
            "channels": channels,
            "basis": "sent_command_not_leader; plateau_is_not_proof_of_saturation",
        }
