from pathlib import Path

import numpy as np

from lerobot.utils.so101_capability_monitor import JOINTS, CapabilityMonitor, GravityModel


class Gravity:
    def torque(self, pose):
        return dict.fromkeys(JOINTS, 0.0)


def monitor():
    return CapabilityMonitor(Gravity(), dict.fromkeys(JOINTS, (-100, 100)))


def readings(t, load=250, current=25):
    return {
        reg: {"read_finished_monotonic_s": t, "error": None, "values": dict.fromkeys(JOINTS, v)}
        for reg, v in [("Present_Load", load), ("Present_Current", current), ("Present_Voltage", 49)]
    }


def test_growing_demand_flat_effort_stalled_response_flags_after_dwell():
    m = monitor()
    q = dict.fromkeys(JOINTS, 0)
    for k in range(51):
        t = k * 0.1
        cmd = dict.fromkeys(JOINTS, 5 + t)
        result = m.update(t, q, cmd, cmd, readings(t))
    assert result["joints"]["shoulder_lift"]["effort_response"]["sustained"]
    assert result["joints"]["shoulder_lift"]["prediction"] == "suspected_response_boundary"
    assert result["advisory_only"]
    assert result["joints"]["shoulder_lift"]["voltage_outside_nominal_models"]


def test_constant_error_flat_effort_is_not_saturation():
    m = monitor()
    q = dict.fromkeys(JOINTS, 0)
    cmd = dict.fromkeys(JOINTS, 8)
    for k in range(51):
        result = m.update(k * 0.1, q, cmd, cmd, readings(k * 0.1))
    assert not result["joints"]["shoulder_lift"]["effort_response"]["candidate"]


def test_positive_progress_and_stale_effort_clear_prediction():
    m = monitor()
    for k in range(51):
        t = k * 0.1
        q = dict.fromkeys(JOINTS, t)
        cmd = dict.fromkeys(JOINTS, 6 + 2 * t)
        r = m.update(t, q, cmd, cmd, readings(t))
    assert not r["joints"]["shoulder_lift"]["effort_response"]["candidate"]
    m = monitor()
    q = dict.fromkeys(JOINTS, 0)
    for k in range(51):
        t = k * 0.1
        cmd = dict.fromkeys(JOINTS, 5 + t)
        r = m.update(t, q, cmd, cmd, readings(0))
    assert not r["joints"]["shoulder_lift"]["effort_response"]["candidate"]


def test_geometry_and_model_only_predictions_have_explicit_reasons():
    class LoadedGravity:
        def torque(self, pose):
            return {j: 0.6 + 0.02 * pose[j] for j in JOINTS}

    m = CapabilityMonitor(LoadedGravity(), dict.fromkeys(JOINTS, (-100, 100)))
    q = dict.fromkeys(JOINTS, 0)
    target = dict.fromkeys(JOINTS, 1)
    r = m.update(0, q, target, q, {})
    assert r["joints"]["shoulder_lift"]["prediction"] == "possible_load_boundary"
    assert r["joints"]["shoulder_lift"]["model_disagreement"]
    q = dict.fromkeys(JOINTS, 99.8)
    target = dict.fromkeys(JOINTS, 101)
    r = m.update(0.1, q, target, q, {})
    assert r["joints"]["shoulder_lift"]["prediction"] == "calibrated_position_boundary"


def test_gravity_is_gradient_of_potential_and_pan_is_zero():
    path = Path(__file__).resolve().parents[2] / "simulation/so101_gravity/vendor/so101_new_calib.urdf"
    g = GravityModel(path, {"gripper": {"range_min": 2000, "range_max": 3500}})
    pose = dict(zip(JOINTS, [0, 10, -60, 40, -20, 1], strict=True))
    result = g.torque(pose)
    assert abs(result["shoulder_pan"]) < 1e-4
    assert result["shoulder_lift"] > 0.3
    assert all(np.isfinite(v) for v in result.values())


def test_broad_proximity_is_not_controller_boundary_and_inactive_is_not_outward_alarm():
    m = monitor()
    q = dict.fromkeys(JOINTS, 98.5)
    target = dict.fromkeys(JOINTS, 100)
    r = m.update(0, q, target, q, {})
    assert r["joints"]["gripper"]["prediction"] == "near_position_boundary"
    q = dict.fromkeys(JOINTS, 99.8)
    r = m.update(0.1, q, target, q, {}, active=False)
    assert not r["active_tracking"]
    assert r["joints"]["gripper"]["prediction"] == "no_warning"


def test_position_proximity_warns_only_for_outward_request():
    for q, inward, outward in [(99, 98, 100), (-99, -98, -100)]:
        for target, expected in [
            (q, "no_warning"),
            (inward, "no_warning"),
            (outward, "near_position_boundary"),
        ]:
            m = monitor()
            measured = dict.fromkeys(JOINTS, q)
            request = dict.fromkeys(JOINTS, target)
            result = m.update(0, measured, request, measured, {})
            joint = result["joints"]["shoulder_lift"]
            assert joint["prediction"] == expected
            assert joint["near_position_boundary"]
            assert joint["requested_outward"] == (expected != "no_warning")


def test_load_boundary_advisory_requires_increasing_absolute_gravity_demand():
    class LoadedGravity:
        def __init__(self, sign):
            self.sign = sign

        def torque(self, pose):
            return {j: self.sign * (0.6 + 0.02 * pose[j]) for j in JOINTS}

    for sign in (-1, 1):
        for request, expected in [
            (0, "no_warning"),
            (-1, "no_warning"),
            (0.1, "no_warning"),
            (1, "possible_load_boundary"),
        ]:
            m = CapabilityMonitor(LoadedGravity(sign), dict.fromkeys(JOINTS, (-100, 100)))
            measured = dict.fromkeys(JOINTS, 0)
            target = dict.fromkeys(JOINTS, request)
            result = m.update(0, measured, target, measured, {})
            joint = result["joints"]["shoulder_lift"]
            assert joint["prediction"] == expected
            assert joint["requested_increasing_load"] == (expected != "no_warning")
            assert joint["model_predictions"]["7p4v"]["torque_warning"]
        m = CapabilityMonitor(LoadedGravity(sign), dict.fromkeys(JOINTS, (-100, 100)))
        result = m.update(
            0, dict.fromkeys(JOINTS, 0), dict.fromkeys(JOINTS, 1), dict.fromkeys(JOINTS, 0), {}, active=False
        )
        assert result["joints"]["shoulder_lift"]["prediction"] == "no_warning"
