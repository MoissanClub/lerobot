from pathlib import Path

import mujoco
import numpy as np
import pytest
from simulate import JOINTS, SERVOS, build_model, gravity_along_measurements, pwm, simulate

URDF = Path(__file__).parent / "vendor/so101_new_calib.urdf"


def test_import_preserves_all_joints_and_nominal_moving_mass(tmp_path):
    model = build_model(URDF, tmp_path / "model.xml", SERVOS[0], True)
    assert model.nv == 6
    assert [model.joint(i).name for i in range(model.njnt)] == list(JOINTS)
    assert sum(model.body_mass) == pytest.approx(
        0.147 + 0.100006 + 0.103 + 0.104 + 0.079 + 0.087 + 0.012, abs=1e-7
    )
    # Saved XML round-trip retains simulation dynamics, not only source URDF.
    saved = mujoco.MjModel.from_xml_path(str(tmp_path / "model.xml"))
    np.testing.assert_allclose(saved.dof_damping, model.dof_damping, rtol=1e-5)
    np.testing.assert_equal(saved.opt.gravity, model.opt.gravity)


def test_gravity_torque_matches_potential_energy_gradient(tmp_path):
    model = build_model(URDF, tmp_path / "model.xml", SERVOS[0], True)
    q = np.array([-0.1, 0.2, -0.8, 0.6, -0.4, 0.02])
    g = gravity_along_measurements(model, q[None, :], np.ones(6), np.zeros(6))[0]
    data = mujoco.MjData(model)
    potential = []
    for delta in [-1e-6, 1e-6]:
        data.qpos[:] = q
        data.qpos[1] += delta
        mujoco.mj_forward(model, data)
        potential.append(-np.sum(model.body_mass * (data.xipos @ model.opt.gravity)))
    assert g[1] == pytest.approx(-(potential[1] - potential[0]) / 2e-6, rel=1e-5)


def test_no_gravity_no_error_no_drift_and_gravity_changes_response(tmp_path):
    t = np.linspace(0, 0.5, 26)
    q = np.tile(np.array([0, 0.1, -0.7, 0.5, 0, 0]), (len(t), 1))
    predictions = []
    for gravity in (False, True):
        model = build_model(URDF, tmp_path / f"{gravity}.xml", SERVOS[0], gravity)
        pred, _, _ = simulate(model, SERVOS[0], t, q, q, np.ones(6), np.zeros(6))
        predictions.append(pred)
    np.testing.assert_allclose(predictions[0], q, atol=1e-12)
    assert np.max(np.abs(predictions[1] - q)) > 0.001


def test_pd_polarity_saturation_and_rated_models():
    assert pwm(np.array([0.1]), np.array([0]))[0] > 0
    assert pwm(np.array([-0.1]), np.array([0]))[0] < 0
    assert pwm(np.array([100]), np.array([0]))[0] == 1
    assert SERVOS[0].stall_nm > SERVOS[1].stall_nm
    assert SERVOS[0].speed_rad_s < SERVOS[1].speed_rad_s


def test_realtime_gravity_matches_mujoco_at_multiple_poses(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from lerobot.utils.so101_capability_monitor import GravityModel

    calibration = {"gripper": {"range_min": 2016, "range_max": 3510}}
    gravity = GravityModel(URDF, calibration)
    model = build_model(URDF, tmp_path / "model.xml", SERVOS[0], True)
    data = mujoco.MjData(model)
    for pose in ([0, 0, 0, 0, 0, 0], [-10, 12, -53, 41, -47, 1.4], [20, -80, 60, -30, 10, 30]):
        values = dict(zip(JOINTS, pose, strict=True))
        data.qpos[:] = [values[j] * gravity.scale[j] for j in JOINTS]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(list(gravity.torque(values).values()), -data.qfrc_bias, atol=2e-6)
