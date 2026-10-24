# SO101 gravity replay: 12 V and 7.4 V nominal actuators

Offline six-joint rigid-body simulation using MuJoCo 3.2.7. No serial access, hardware commands, LeRobot imports, cameras or GUI are required. The robot-control environment is unchanged; simulation dependencies live in the parent workspace's `.venv-sim`.

Four model/config pairs under `models/` cover 12 V and 7.4 V servo variants at their nominal supply, each with gravity on/off. XML describes the passive mechanical plant, including implicit electrical damping. The companion JSON specifies the external PD/duty controller; loading XML alone does **not** execute that controller. `simulate.py` applies it. All six follower motors use the selected variant.

## Reproduce

From the parent workspace containing `lerobot-src` and the existing `.tools/uv`:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" .tools/uv venv .venv-sim --python .venv/bin/python
UV_CACHE_DIR="$PWD/.cache/uv" .tools/uv pip install --python .venv-sim/bin/python -r lerobot-src/simulation/so101_gravity/requirements.lock
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv-sim/bin/python lerobot-src/simulation/so101_gravity/simulate.py \
  --log outputs/lab/so101-20261023/state-machine-1792809011865027000.jsonl \
  --calibration outputs/lab/so101-20261023/calibration/follower/follower.json \
  --output outputs/development/so101-gravity-new --sweep
.venv-sim/bin/python -m pytest -q lerobot-src/simulation/so101_gravity/test_simulation.py
```

Output directory must not exist. It contains four XML/JSON model pairs, four compressed per-joint NPZ trajectories, measured-configuration gravity torques, comparison CSV, PNG/PDF plots and summary JSON with input/script hashes, versions, metrics and sensitivity results. Input logs/calibration remain private artifacts, outside source Git. `.npz` position arrays use logged degrees for arm joints and opening units for gripper; torque arrays are Nm, duty is normalized [-1,1].

Replay starts at keyboard Start, initializes **all six joints** from measured positions with zero velocities, then drives all six simulated joints dynamically with the actual sent goals. Commands are zero-order held at logged timestamps, not interpolated leader targets. Replay stops before the first Lagging → Awaiting alignment holding command. It does not rerun the state machine, infer new operator behavior or introduce the observed holding command prematurely.

## Mechanics and coordinates

Pinned public SO101 URDF, upstream license and revision/hash are in `vendor/`. The operator states the physical assembly matches the public version. MuJoCo imports the original joint transforms and inertial parameters. Visual/collision meshes are removed, so no mesh downloads or contact forces are involved. Joint constraints are disabled for the free-space comparison; this is not a hard-stop or collision simulator.

Arm degrees map directly to the public `so101_new_calib` URDF's joint radians, following the LeRobot kinematics examples. **Physical alignment of local calibration midpoints to CAD zeros has not been independently verified.** Gripper opening maps to radians using the recorded calibration encoder span, with closed opening assumed at URDF zero. The base uses the URDF's upright orientation. `--offsets-deg` exposes six joint-zero offsets; the default is all zero.

## Actuator approximation

Nominal data from [Waveshare](https://docs.waveshare.com/ST3215_Servo):

| Variant | Stall torque | No-load time per 60° | Stall current | Listed torque constant |
|---|---:|---:|---:|---:|
| 12 V | 30 kg·cm | .222 s | 2.7 A | 11 kg·cm/A |
| 7.4 V | 19.5 kg·cm | .192 s | 2.5 A | 7.8 kg·cm/A |

Using 1 kg·cm = .0980665 Nm, the approximate DC torque/speed law is:

```
u = clip(gain_scale * ((P/4)*error_ticks - (D/8)*velocity_ticks_s*assumed_sample_s) / 1000, -1, 1)
tau_motor = tau_stall*u - (tau_stall/omega_no_load)*joint_velocity
```

P=16, I=0, D=32 match the controller configuration. [FEETECH](https://ftservo.github.io/ftservo-wiki/en/reference/parameter/memory-sts/) documents P/4 and D/8 scaling; interpreting them as this duty law and interpreting D as sampled measured-velocity damping are **modeling assumptions**, not an identified firmware equation. Assumed derivative period is 1ms. Back-EMF is integrated as implicit joint damping; reflected rotor inertia .0001 kg·m² per joint is assumed. Default mechanical friction is zero. Duty saturation is modeled; a separate current limiter, protection logic, deadband, backlash and internal trajectory planner are not. Current specifications are metadata, not a fitted sensor model. Motor torque is not computed by multiplying measured load by stall torque.

CLI parameters: `--gain-scale`, `--derivative-sample-s`, `--friction-nm`, `--offsets-deg`. The default `--sweep` runs 36 short simulations (two variants × gravity on/off × gain .5/1/2 × shoulder zero offset -5/0/+5°), each initialized from measured state four seconds before recovery. These short runs are sensitivity checks, distinct from the four full command replays.

Both models are at **rated voltage**, not the approximately 4.7–4.9 V reported in the physical log. Matching one nominal model cannot identify the installed variant, actual effort or real supply. Remaining mass, friction, actuator and coordinate uncertainties prevent causal confirmation from a curve match.

## Validation and interpretation

Tests check imported joint order/total mass, model XML round-trip, gravity torque against an independently calculated potential-energy gradient, zero-error no-gravity equilibrium, gravity response and PD polarity/saturation. A separate numerical comparison of 1ms and .5ms integration over the final four seconds checks timestep sensitivity.

Use final-four-second RMSE in addition to whole-run RMSE: much of the run is a simple ramp that all models can track. Report predicted gravity torque along the *measured* configuration separately from simulated configurations. Positive gravity torque means gravity pushes the positive joint direction; `qfrc_bias` has the opposite sign at rest.

Gravity can support a residual-error explanation without reproducing the exact onset/plateau. Rising sent goals can produce increasing simulated angles even with gravity disabled, so increasing angle alone is not evidence of gravitational sag. No automatic controller gains or physical-run changes follow from this simulation.
