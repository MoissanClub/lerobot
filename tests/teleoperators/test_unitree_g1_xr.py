# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.unitree_g1.g1_xr_control import G1XRControl
from lerobot.teleoperators.xr_controllers import XRControllers, XRControllersConfig
from lerobot.teleoperators.xr_controllers.xr_controllers import controller_pose


def sample(stamp=10.0, left=1.0, right=1.0):
    result = {"captured_at": stamp}
    for side, squeeze in (("left", left), ("right", right)):
        result.update(
            {
                f"{side}.grip_pos": np.zeros(3),
                f"{side}.grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                f"{side}.squeeze": squeeze,
                f"{side}.trigger": 0.0,
                f"{side}.tracked": True,
            }
        )
    return result


class Kinematics:
    size = 6

    def fk(self, q):
        result = [np.eye(4), np.eye(4)]
        for i in range(2):
            result[i][:3, 3] = q[3 * i : 3 * i + 3]
        return result

    def solve(self, left, right, seed):
        return SimpleNamespace(q=np.r_[left[:3, 3], right[:3, 3]])

    def arm_action(self, q):
        return {str(i): v for i, v in enumerate(q)}


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("grip_quat", [0, 0, 0, 0]),
        ("grip_quat", [0, 0, 0, np.nan]),
        ("grip_pos", [0, np.inf, 0]),
        ("grip_pos", [0]),
        ("squeeze", 2),
        ("trigger", -1),
        ("tracked", False),
    ],
)
def test_invalid_pose_releases(side, field, value):
    control = G1XRControl(Kinematics())
    action = sample()
    control.action(action, np.zeros(6), now=10)
    action[f"{side}.{field}"] = value
    assert controller_pose(action, side) is None
    control.action(action, np.zeros(6), now=10)
    assert not control.engaged[("left", "right").index(side)]


def test_independent_motion_release_reengage_and_staleness():
    control = G1XRControl(Kinematics())
    q = np.zeros(6)
    action = sample(right=0)
    control.action(action, q, now=10)
    action["left.grip_pos"] = np.array([0.1, 0.2, 0.3])
    result = control.action(action, q, now=10)
    np.testing.assert_allclose(list(result.values()), [0.1, 0.2, 0.3, 0, 0, 0])
    control.action(action, q, now=11)
    assert control.engaged == (False, False)
    q[:3] = [0.01, 0.02, 0.03]
    action["captured_at"] = 11
    np.testing.assert_allclose(list(control.action(action, q, now=11).values()), q)


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), 20.0, 1.0, None])
def test_bad_timestamp_holds(stamp):
    control = G1XRControl(Kinematics())
    assert list(control.action(sample(stamp), np.zeros(6), now=10).values()) == [0] * 6
    assert not any(control.engaged)


def test_rotation_and_timestamp_regression():
    control = G1XRControl(Kinematics())
    action = sample()
    control.action(action, np.zeros(6), now=10)
    action["left.grip_quat"] = [0, 0, np.sin(0.2), np.cos(0.2)]
    control.action(action, np.zeros(6), now=10)
    assert not np.allclose(control.targets[0][:3, :3], np.eye(3))
    np.testing.assert_allclose(control.targets[1][:3, :3], np.eye(3))
    action["captured_at"] = 9.99
    control.action(action, np.zeros(6), now=10)
    assert not any(control.engaged)


def test_device_lifecycle_and_factory(tmp_path):
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    config = XRControllersConfig(calibration_dir=tmp_path)
    assert isinstance(make_teleoperator_from_config(config), XRControllers)

    class Backend:
        closed = False

        def __init__(self, config):
            pass

        def connect(self):
            pass

        def read(self):
            return sample()

        def close(self):
            self.closed = True

    device = XRControllers(config, session_factory=Backend)
    with pytest.raises(RuntimeError):
        device.get_action()
    device.connect()
    backend = device._backend
    assert set(device.get_action()) == set(device.action_features)
    with pytest.raises(RuntimeError):
        device.connect()
    device.disconnect()
    assert backend.closed and not device.is_connected
    device.connect()
    device.disconnect()


def test_connect_failure_cleanup(tmp_path):
    class Backend:
        closed = False

        def connect(self):
            raise ValueError("connect failed")

        def close(self):
            self.closed = True

    backend = Backend()
    device = XRControllers(XRControllersConfig(calibration_dir=tmp_path), session_factory=lambda _: backend)
    with pytest.raises(ValueError):
        device.connect()
    assert backend.closed and not device.is_connected


def test_invalid_transform():
    with pytest.raises(ValueError):
        XRControllersConfig(base_T_anchor=np.zeros((4, 4)).tolist())
