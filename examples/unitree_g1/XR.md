# Dual-arm XR control

`XRControllers` is a LeRobot teleoperator registered as `xr_controllers`. It reads
both controllers in one Isaac Teleop session and has no robot-model dependency.
Start your separately installed CloudXR runtime first. No runtime installation,
EULA acceptance, or external process launch happens implicitly.

```python
from lerobot.teleoperators.xr_controllers import XRControllers, XRControllersConfig
from lerobot.robots.unitree_g1.g1_xr_control import G1XRControl

reader = XRControllers(XRControllersConfig())
control = G1XRControl(kinematics)  # G1ArmKinematics for the selected embodiment
reader.connect()
try:
    sample = reader.get_action()
    command = control.action(sample, measured_arm_positions)
finally:
    reader.disconnect()
```

Controller poses are rebased from OpenXR into robot-base coordinates. Quaternions
are xyzw; positions are meters. Squeeze engages each arm independently. Grip
translation/rotation changes are relative to the engagement pose. Release,
malformed data, worker deadline misses, and stale/out-of-order samples disengage;
re-engagement rebases from measured FK. Call `control.reset()` when reconnecting
the input source. This is not a physical emergency stop or collision avoidance.

The installed Isaac Teleop session's public exit method skips teardown when
initialization is incomplete. The adapter drains its private entry ExitStack on
startup failure, so contexts already opened before DeviceIO/plugin failure close.
This SDK-specific workaround has a regression test and must be revisited when
updating the SDK; native failures inside a context's own entry remain vendor-owned.

The SDK interface follows the existing NVIDIA Isaac Teleop example in this
repository; installation instructions remain in `examples/isaac_teleop_to_so101`.
The unit suite needs no SDK/headset. Simulation is already inherited through
the Cartesian parent; its integration test additionally needs pinned model assets:

```bash
PYTHONPATH=src python -m pytest tests/teleoperators/test_unitree_g1_xr.py -q
G1_INTEGRATION_TESTS=1 G1_KINEMATICS_ASSETS=/path/to/assets PYTHONPATH=src \
  python -m pytest tests/integration/test_unitree_g1_xr_mujoco.py -q
```

Automated replay verifies mapping and dynamics, not headset tracking quality.
Manual acceptance: both embodiments, independent arm motion, rotation, release,
tracking loss, and reconnect. The physical G1-23 connection remains disabled.
