# BrainCo Revo2 hands

This optional adapter targets **bc-stark-sdk==2.0.2**, Modbus at 460800 baud,
Revo2Basic or capacitive Revo2Touch. It does not support other tactile variants.
Install the SDK explicitly (`pip install bc-stark-sdk==2.0.2`). Importing the
configuration or running mock tests does not require it.

The API was checked against the published wheel and vendor examples pinned to
[`bb8f8209b69a85c95ca8b0219fd4c22e62aa4a52`](https://github.com/BrainCoTech/brainco-hand-sdk/tree/bb8f8209b69a85c95ca8b0219fd4c22e62aa4a52/python/revo2).
The wheel uses `sdk.modbus_close(client)`, unlike the current dual-hand example's
client method. SDK version is checked before opening any port.

```python
from lerobot.robots.unitree_g1.brainco_hand import BrainCoHandConfig
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

config = UnitreeG1Config(is_simulation=False, end_effector="brainco", hands={
    "left": BrainCoHandConfig(side="left", port="/dev/serial/by-id/LEFT"),
    "right": BrainCoHandConfig(side="right", port="/dev/serial/by-id/RIGHT"),
})
```

The robot remains `unitree_g1`; no separate hand-equipped robot identity is
needed. Hand drivers must agree with the robot's `end_effector` selection.
BrainCo simulation is explicitly rejected: a mocked SDK is not an articulated
MuJoCo hand model. The physical G1-23 connection remains gated independently.

Hardware is **disabled by default**. Physical preflight and explicit
`allow_hardware=True` are required. No device scanning, automatic homing, grasp,
or startup motion occurs. Connection checks reported model, left/right identity,
and normalized unit mode. Default IDs are 0x7e left / 0x7f right; override when
needed. Commands are accepted only after a valid measured starting pose.

Commands are `hands.left.motor_0.pos` through `motor_5.pos` (likewise right),
normalized 0..1 and converted to SDK 0..1000 integers in exactly that order.
Motor slots are explicit because vendor examples disagree on anatomical labels.
The first two slots have provisional software upper limits 0.5; others 1.0.
These limits and the six-slot mapping require physical review before use.
`closure_action(0..1)` scales each slot to its configured upper limit.

Default requested speed is 0.2, command slew is 0.5 normalized units/s, and elapsed
time is capped at 0.1s to prevent a large jump after a stall. These are software
bounds, not certified safety limits. Driver calls are synchronous and have an
asyncio timeout; a blocking native SDK call cannot be forcibly interrupted by
that timeout. Use a dedicated non-async control thread.

Position feedback is normalized; speed/current/state remain explicitly named
raw SDK values. Optional capacitive tactile feedback exposes five fingers' status,
normal/tangential force and proximity in **raw units**, not calibrated SI values.
Malformed feedback, wrong units/model/side, faults, and communication failures
raise and close the adapter. Closing serial is **not** torque disable or an E-stop.

```bash
PYTHONPATH=src python -m pytest tests/robots/test_unitree_g1_brainco_hands.py -q
# Also inspect installed SDK symbols; never opens a serial port:
G1_BRAINCO_SDK_TESTS=1 PYTHONPATH=src python -m pytest \
  tests/robots/test_unitree_g1_brainco_hands.py -q
```

Tests use fake SDK responses for both sides, mapping, rate limits, unit/identity
rejection, timeouts, tactile failures and config serialization. Physical motion,
grasping, tactile calibration, and safety acceptance are not completed by this branch.
