# G1 Cartesian Control Verification

This example verifies the shared G1-29/G1-23 kinematics layer in the checked-out
LeRobot fork. It does not connect DDS, a physical robot, or XR. The viewer loads
each embodiment's native URDF and sets joint positions directly: it is kinematic
playback, not motor dynamics. Existing G1-29 Hub runtime behavior is unchanged.
Parent: `g1/simulation`. Both embodiments already have joint-level native simulation;
this branch adds Cartesian targets without making the solver depend on MuJoCo.
Physical G1-23 connections remain disabled.

## Environment

Use a dedicated environment. Do not install PyPI `pin` or `pinocchio` over the
conda-forge packages: the symbolic Pinocchio/CasADi ABI must match.

```bash
conda create -n lerobot-g1-cartesian -c conda-forge --strict-channel-priority \
  python=3.12 pinocchio=3.9.0 casadi=3.7.2 pip tk
conda activate lerobot-g1-cartesian
python -m pip install -e '.[test,unitree_g1]' 'mujoco==3.12.0' pillow
python -c 'import pinocchio; from pinocchio import casadi; print(pinocchio.__version__)'
```

The full LeRobot dependencies are needed by its common pytest fixtures. The new
numerical layer needs only NumPy, Pinocchio, and CasADi. MuJoCo/Pillow/Tk are for
the verification example, not dependencies of the control API. Verification here
used a separate venv inheriting an existing working conda-forge environment; the
fresh installation recipe above has not been independently exercised.

## Prepare Pinned Models

From the fork repository root:

```bash
export PYTHONPATH="$PWD/src"
export G1_KINEMATICS_ASSETS=/tmp/g1-cartesian-assets
python examples/unitree_g1/prepare_cartesian_assets.py \
  --output "$G1_KINEMATICS_ASSETS" --meshes
```

Omit `--meshes` for numerical-only tests. This explicit download step uses:

- G1-29: `lerobot/unitree-g1-mujoco` at `a38dc8617f0fca51b38e9354dc58ee35ad850fb5`.
- G1-23: `unitreerobotics/unitree_lerobot` at `41c2805742de879ddab2d8d6beaeaf215f876395`.

The command verifies URDF SHA-256 hashes and writes `manifest.json`. These are
the same URDF bytes as the integration reference. G1-23 meshes reuse the pinned
Hub snapshot; a runtime with another hardware/model revision must be audited
separately. The solver itself takes an explicit URDF path and never downloads.

## Automated Acceptance

Run the cumulative embodiment, simulation, and Cartesian suites:

```bash
G1_RENDER_TESTS=1 MUJOCO_GL=egl python -m pytest -q \
  tests/robots/test_unitree_g1.py \
  tests/robots/test_unitree_g1_utils.py \
  tests/robots/test_unitree_g1_embodiments.py \
  tests/teleoperators/test_unitree_g1_teleoperator.py \
  tests/robots/test_sonic_whole_body.py \
  tests/robots/test_unitree_g1_simulation.py \
  tests/integration/test_unitree_g1_mujoco_runtime.py \
  tests/robots/test_unitree_g1_action_processor.py \
  tests/robots/test_unitree_g1_cartesian_control.py \
  tests/robots/test_unitree_g1_kinematics.py
```

`G1_KINEMATICS_ASSETS` is required for milestone acceptance. Without it, the
real-model module is intentionally skipped for lightweight CI. With it, missing
dependencies, missing files, or mismatched model hashes fail the suite. An
unrelated optional SONIC-module skip is reported separately.

Checks cover full/reduced FK agreement (1e-12), gravity versus finite-difference
potential-energy gradients (absolute 1e-7 Nm plus relative 1e-6), reachable IK
within 5 mm / 0.05 rad, moving targets, independent arms, sparse transport mapping,
nonidentity model ordering, invalid inputs, and failed/unreachable solves. Every
returned IK command is joint-limited and changes at most 0.1 rad per call.

## Headless and Visible Verification

```bash
MUJOCO_GL=egl python examples/unitree_g1/verify_cartesian_control.py \
  --assets "$G1_KINEMATICS_ASSETS" --headless --samples-per-phase 24 \
  --frames /tmp/g1-cartesian-frames --report /tmp/g1-cartesian-report.json
```

The headless command rejects blank/static panels, optimizer failures, nonfinite
or out-of-bounds commands, and disagreement between Pinocchio FK and rendered
MuJoCo URDF poses. MJCF serialization tolerance is 1e-5 m per coordinate and
1e-5 per rotation-matrix entry. Saved JSON includes the source path/hash, Git
commit, URDF hashes, pose errors, status, and actual motion. It retains at most
2400 frames per model to keep a continuously running viewer's memory bounded.

For an SSH session with working `ssh -Y` forwarding:

```bash
MUJOCO_GL=egl python examples/unitree_g1/verify_cartesian_control.py \
  --assets "$G1_KINEMATICS_ASSETS" --camera-azimuth -135
```

This opens two Tk panels and repeats forward/back, left/right, up/down, and hand
roll motion until the window closes. EGL renders on the host; X forwarding only
transports the Tk images. Use `--cycles 1` for a finite run. `--camera-azimuth`
controls the shared camera; -135 is the robots' left-front.

The common Cartesian sweep is a workspace probe, not a promise that all poses
are reachable. Its exit status verifies bounded, moving, correctly mapped
playback; pose mismatches are explicitly labeled `bounded_best_effort`, not
silently counted as IK successes. Numerical tests separately require reachable
targets to converge. G1-23 has five joints per arm and cannot independently match
arbitrary six-dimensional poses. Inspect both panels' actual motion and error
labels rather than comparing success counts alone.

## Recorded Development Verification

On 2026-09-11, the cumulative suite passed **181 tests**, with one unrelated
optional SONIC-module skip. All 53 new Cartesian/kinematics cases ran, including
the real pinned G1-29 and G1-23 models. Ruff lint/format checks passed.

The 96-frame-per-model headless sweep completed with no optimizer failures and
95 changed frames in each panel. Maximum positional residuals were approximately
32.6 mm for G1-29 and 31.0 mm for G1-23 on the common workspace probe; these are
reported best-effort cases, not successes at the stricter 5 mm IK threshold.
Reachable-target acceptance passed separately. A real Tk/Xvfb session completed
32 frames per model and closed normally. Saved native-model frames were inspected;
interactive review over the user's SSH connection remains a manual acceptance step.

## API Contract

```python
import numpy as np
from lerobot.robots.unitree_g1.g1_cartesian_control import G1ArmKinematics, G1CartesianConfig

ik = G1ArmKinematics(G1CartesianConfig("g1_23", "/tmp/g1-cartesian-assets/g1_23.urdf"))
q = np.zeros(10)
left, right = ik.fk(q)
left[2, 3] += 0.02
result = ik.solve(left, right, seed=q)
action = ik.arm_action(result.q)
tau_body = ik.scatter_arm(result.gravity_tau)
```

- Compact vectors always use active DDS arm order: 14 values for G1-29, 10 for G1-23.
- Transforms are root-frame homogeneous matrices in metres/radians; legs, waist,
  and fingers are locked at URDF neutral. Root-frame means the URDF root, not an
  implicit headset, torso, or world frame.
- Endpoints match the reference: 0.05 m along local X from G1-29 wrist yaw, 0.20 m
  from G1-23 wrist roll. These are task frames, not interchangeable wrist origins.
- `gravity()` returns static Nm feedforward for the URDF's masses, including any
  locked hand masses. No additional payload or velocity/acceleration term is added.
- `arm_action()` does not zero legs/waist. `scatter_arm()` returns a new 29-slot
  vector with non-arm entries zero; it does not send a command.
- Optimizer failure holds the seed and reports `optimizer_failed_hold`; a bounded
  pose mismatch reports `bounded_best_effort`. `converged` applies to the returned
  pose, not simply solver termination. Invalid input raises without changing state.
- Step bounds are per-call, not a real-time velocity guarantee. Use one solver
  instance per control loop. No collision avoidance or physical safety claim is made.
