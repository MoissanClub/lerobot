# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
import asyncio
import os
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.unitree_g1.brainco_hand import MOTORS, BrainCoHand, BrainCoHandConfig
from lerobot.robots.unitree_g1.hand_system import make_hand


class Client:
    def __init__(self, side, model):
        self.side, self.model = side, model
        self.mode = "normalized"
        self.closed = False
        self.commands = []
        self.status = SimpleNamespace(positions=[0] * 6, speeds=[0] * 6, currents=[0] * 6, states=[0] * 6)
        self.touch = [
            SimpleNamespace(status=0, normal_force1=1.0, tangential_force1=2.0, self_proximity1=3.0)
            for _ in range(5)
        ]
        self.stall = False

    async def get_device_info(self, dev):
        return SimpleNamespace(
            hand_type=self.side,
            get_hardware_type=lambda: self.model,
            uses_revo2_touch_api=lambda: self.model == "touch",
        )

    async def set_finger_unit_mode(self, dev, mode):
        pass

    async def get_finger_unit_mode(self, dev):
        return self.mode

    async def get_motor_status(self, dev):
        return self.status

    async def touch_sensor_setup(self, dev, bits):
        pass

    async def get_touch_sensor_enabled(self, dev):
        return 0x1F

    async def get_touch_sensor_status(self, dev):
        return self.touch

    async def set_finger_positions_and_speeds(self, dev, positions, speeds):
        if self.stall:
            await asyncio.sleep(1)
        self.commands.append((dev, positions, speeds))


class SDK:
    Baudrate = SimpleNamespace(Baud460800=460800)
    HandType = SimpleNamespace(Left="left", Right="right")
    StarkHardwareType = SimpleNamespace(Revo2Basic="basic", Revo2Touch="touch")
    FingerUnitMode = SimpleNamespace(Normalized="normalized")

    def __init__(self, client):
        self.client = client

    async def modbus_open(self, port, baud):
        self.open_args = port, baud
        return self.client

    def modbus_close(self, client):
        client.closed = True


@pytest.fixture(params=["left", "right"])
def hand(request):
    client = Client(request.param, "basic")
    sdk = SDK(client)
    clock = [10.0]
    config = BrainCoHandConfig(side=request.param, port="/dev/test", allow_hardware=True)
    result = BrainCoHand(config, sdk_factory=lambda: sdk, clock=lambda: clock[0])
    result.connect()
    yield result, client, clock
    result.disconnect()


def test_default_denies_hardware_and_factory():
    config = BrainCoHandConfig(side="left", port="/dev/test")
    hand = make_hand(config)
    assert isinstance(hand, BrainCoHand)
    with pytest.raises(PermissionError):
        hand.connect()
    assert hand.runner is None


def test_units_order_speeds_and_rate(hand):
    device, client, clock = hand
    clock[0] += 0.1
    requested = {f"{name}.pos": (i + 1) * 0.01 for i, name in enumerate(MOTORS)}
    sent = device.send_action(requested)
    dev, positions, speeds = client.commands[-1]
    assert dev == (0x7E if device.config.side == "left" else 0x7F)
    assert positions[:4] == [10, 20, 30, 40]
    assert max(positions) <= 50 and speeds == [200] * 6
    np.testing.assert_allclose(list(sent.values()), np.array(positions) / 1000)
    old = sent.copy()
    assert device.send_action(device.closure_action(1)) == old
    clock[0] += 100
    sent = device.send_action(device.closure_action(1))
    assert max(np.array(list(sent.values())) - np.array(list(old.values()))) <= 0.050001


@pytest.mark.parametrize(
    "bad", [{"unknown": 0}, {"motor_0.pos": 0.6}, {"motor_2.pos": np.nan}, {"motor_1.pos": -1}]
)
def test_invalid_commands_no_write(hand, bad):
    device, client, _ = hand
    with pytest.raises(ValueError):
        device.send_action(bad)
    assert client.commands == []


def test_observation_roundtrip_and_fault(hand):
    device, client, _ = hand
    client.status.positions = [100, 200, 300, 400, 500, 600]
    result = device.get_observation()
    assert set(result) == set(device.observation_features)
    assert result["motor_5.pos"] == 0.6
    client.status.positions = [np.nan] * 6
    with pytest.raises(ValueError):
        device.get_observation()
    assert client.closed and not device.is_connected


def test_timeout_closes_transport(hand):
    device, client, clock = hand
    device.config.timeout_s = 0.01
    client.stall = True
    clock[0] += 0.1
    with pytest.raises(TimeoutError):
        device.send_action(device.closure_action(0.5))
    assert client.closed and device.runner is None


@pytest.mark.parametrize("problem", ["side", "model", "units"])
def test_connect_identity_and_units_fail_closed(problem):
    client = Client("left", "basic")
    if problem == "side":
        client.side = "right"
    if problem == "model":
        client.model = "other"
    if problem == "units":
        client.mode = "physical"
    device = BrainCoHand(
        BrainCoHandConfig(side="left", port="/dev/test", allow_hardware=True), sdk_factory=lambda: SDK(client)
    )
    with pytest.raises(ValueError):
        device.connect()
    assert client.closed and device.runner is None


def test_tactile_features_and_fault():
    client = Client("right", "touch")
    device = BrainCoHand(
        BrainCoHandConfig(
            side="right", port="/dev/test", allow_hardware=True, hardware_type="Revo2Touch", tactile=True
        ),
        sdk_factory=lambda: SDK(client),
    )
    device.connect()
    result = device.get_observation()
    assert set(result) == set(device.observation_features)
    assert result["pinky.normal_force_raw"] == 1
    client.touch[2].status = 2
    with pytest.raises(ValueError):
        device.get_observation()
    assert client.closed


def test_configuration_serialization():
    import draccus

    from lerobot.robots.unitree_g1.g1_with_hands import G1WithHandsConfig

    config = G1WithHandsConfig(hands={"left": BrainCoHandConfig(side="left", port="/dev/test")})
    restored = draccus.decode(G1WithHandsConfig, draccus.encode(config))
    assert isinstance(restored.hands["left"], BrainCoHandConfig)
    assert restored.hands["left"].device_id == 0x7E


@pytest.mark.parametrize(
    "kwargs",
    [
        {"side": "middle"},
        {"port": ""},
        {"device_id": 0},
        {"speed": 0},
        {"max_rate": np.inf},
        {"upper": [1] * 5},
        {"tactile": True},
        {"upper": [0.3333] * 6},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        BrainCoHandConfig(**{"side": "left", "port": "/dev/test", **kwargs})


@pytest.mark.skipif(
    not os.environ.get("G1_BRAINCO_SDK_TESTS"), reason="Opt-in installed SDK API audit; never opens a port"
)
def test_installed_sdk_contract():
    from lerobot.robots.unitree_g1.brainco_hand import load_sdk

    sdk = load_sdk()
    for name in ("modbus_open", "modbus_close"):
        assert callable(getattr(sdk, name))
    for name in (
        "get_device_info",
        "get_motor_status",
        "set_finger_unit_mode",
        "get_finger_unit_mode",
        "set_finger_positions_and_speeds",
        "get_touch_sensor_status",
        "touch_sensor_setup",
        "get_touch_sensor_enabled",
    ):
        assert callable(getattr(sdk.DeviceContext, name))
    for name in ("positions", "speeds", "currents", "states"):
        assert hasattr(sdk.MotorStatusData, name)
    assert hasattr(sdk.DeviceInfo, "hand_type") and hasattr(sdk.DeviceInfo, "get_hardware_type")
def test_brainco_uses_robot_end_effector_configuration():
    from lerobot.envs.configs import UnitreeG1MujocoEnv
    from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

    hands = {"left": BrainCoHandConfig(side="left", port="/dev/unused")}
    config = UnitreeG1Config(is_simulation=False, end_effector="brainco", hands=hands)
    assert config.hands == hands and config.sim_env is None
    with pytest.raises(ValueError, match="BrainCo"):
        UnitreeG1Config(is_simulation=False, hands=hands)
    with pytest.raises(ValueError, match="BrainCo"):
        UnitreeG1Config(is_simulation=False, end_effector="brainco")
    with pytest.raises(ValueError, match="simulation"):
        UnitreeG1MujocoEnv(end_effector="brainco")
