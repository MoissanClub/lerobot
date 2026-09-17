# Native Simulation Branch

Parent: `g1/cartesian-control`. `UnitreeG1Config(simulation_urdf=...)` explicitly
selects a local, supported-arm MuJoCo backend for either embodiment. Without this
option, the existing G1-29 Hub/DDS and physical paths are unchanged. G1-23 physical
connections and unsupported whole-body controllers remain rejected.

The new backend uses the same pinned URDF as FK/IK, locks non-arm joints at zero,
and uses configured LeRobot gains (G1-29) or source-derived gains (G1-23). Physics
has gravity enabled. `gravity_compensation=True` adds the shared Pinocchio static
feedforward at the measured arm pose, clipped with PD torque to URDF effort limits.
It adds explicit simulation damping 0.05 and armature 0.01, not calibrated motor
properties. Contacts are removed for this supported-arm acceptance backend.

This is deterministic in-process simulation: `send_action` advances one control
period, `step_simulation` advances held targets, and observations do not step time.
There is no new DDS transport, robot discovery, or background simulation process.
Use the preserved integration repository for the previously verified multi-process
DDS workflow. Porting that workflow is not implied by passing these direct-backend tests.

Prepare assets/environment following `README.md`. In addition to its cumulative
tests, run:

```bash
export PYTHONPATH="$PWD/src"
export G1_KINEMATICS_ASSETS=/tmp/g1-cartesian-assets
G1_RENDER_TESTS=1 MUJOCO_GL=egl python -m pytest -q \
  tests/robots/test_unitree_g1_simulation.py \
  tests/integration/test_unitree_g1_mujoco_runtime.py
python examples/unitree_g1/run_simulation.py --assets "$G1_KINEMATICS_ASSETS" --embodiment g1_23 --headless
python examples/unitree_g1/run_simulation.py --assets "$G1_KINEMATICS_ASSETS" --embodiment g1_29 --headless
```

Omit `--headless` for a Tk spectator view and keyboard Cartesian targets. The example
enables gravity compensation unless `--no-gravity-compensation` is supplied. Test
acceptance requires both runtime variants, invalid-command rejection, reset/reconnect,
gravity matching and reduced static sag, and changing nonblank robot-camera frames.
Run with `G1_RENDER_TESTS=1`; a skipped renderer test is not full branch acceptance.
