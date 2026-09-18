# Native Simulation Branch

Parent: `g1/embodiments`. `UnitreeG1Config(simulation_urdf=...)` explicitly
selects a local, supported-arm MuJoCo backend for either embodiment. Without this
option, the existing G1-29 Hub/DDS and physical paths are unchanged. G1-23 physical
connections and unsupported whole-body controllers remain rejected.

The backend uses a pinned URDF, locks non-arm joints at zero,
and uses configured LeRobot gains (G1-29) or source-derived gains (G1-23). Physics
has gravity enabled. `gravity_compensation=True` adds MuJoCo zero-velocity gravity
feedforward at the measured arm pose, clipped with PD torque to URDF effort limits.
It adds explicit simulation damping 0.05 and armature 0.01, not calibrated motor
properties. Contacts are removed for this supported-arm acceptance backend.

This is deterministic in-process simulation: `send_action` advances one control
period, `step_simulation` advances held targets, and observations do not step time.
There is no new DDS transport, robot discovery, or background simulation process.
Use the preserved integration repository for the previously verified multi-process
DDS workflow. Porting that workflow is not implied by passing these direct-backend tests.

No Cartesian, Pinocchio or CasADi dependency is needed. Prepare the pinned models
with `python examples/unitree_g1/prepare_cartesian_assets.py --output /tmp/g1-cartesian-assets --meshes`.
The downloader keeps its historical filename for compatibility; it performs no IK.
In addition to cumulative tests, run:

```bash
export PYTHONPATH="$PWD/src"
export G1_KINEMATICS_ASSETS=/tmp/g1-cartesian-assets
G1_RENDER_TESTS=1 MUJOCO_GL=egl python -m pytest -q \
  tests/robots/test_unitree_g1_simulation.py \
  tests/integration/test_unitree_g1_mujoco_runtime.py
python examples/unitree_g1/run_simulation.py --assets "$G1_KINEMATICS_ASSETS" --embodiment g1_23 --headless
python examples/unitree_g1/run_simulation.py --assets "$G1_KINEMATICS_ASSETS" --embodiment g1_29 --headless
```

Omit `--headless` for a Tk spectator view and keyboard joint targets. The example
enables gravity compensation unless `--no-gravity-compensation` is supplied. Test
acceptance requires both runtime variants, invalid-command rejection, reset/reconnect,
gravity matching and reduced static sag, and changing nonblank robot-camera frames.
Run with `G1_RENDER_TESTS=1`; a skipped renderer test is not full branch acceptance.

## Standard CLI Viewer

The existing CLI can open the native MuJoCo viewer, without a new launcher or IK:

```bash
MUJOCO_GL=glfw lerobot-teleoperate \
  --robot.type=unitree_g1 --robot.embodiment=g1_23 --robot.is_simulation=true \
  --robot.simulation_urdf="$G1_KINEMATICS_ASSETS/g1_23.urdf" \
  --robot.simulation_mesh_dir="$G1_KINEMATICS_ASSETS/meshes" \
  --robot.sim_publish_images=false --robot.sim_onscreen=true \
  --robot.gravity_compensation=true --robot.cameras='{}' \
  --teleop.type=unitree_g1_keyboard --teleop.id=simulation_review --display_data=true
```

Repeat with `g1_29` and its URDF. Press Enter to enable, L/R to select an arm,
1-7 to select a joint (1-5 on G1-23), +/- to jog, Space to hold/disarm, Escape to exit.
No controller or exoskeleton ports are configured: this is joint-space arm control,
not locomotion or Cartesian control. The older Tk example remains a diagnostic.
Rerun receives joint telemetry, not camera images in native-viewer mode.
For headless CLI verification set `sim_onscreen=false`, `display_data=false`, and
`teleop_time_s=1`. Actual viewer/X review remains manual.
