# Supervised SO101 state-machine trial

This is a separate opt-in runner; existing teleoperate/record commands are unchanged. It uses the shared deterministic `lerobot.utils.so101_state_machine.Controller`. No cameras or dataset recording are enabled. Telemetry is always captured. Software validation does not establish physical holding, stall detection or recovery safety.

## First physical test

From `/Users/dwei/lerobot`, in the operator's Terminal (the agent sandbox previously could not access USB/audio):

```sh
.venv/bin/python supervised_so101.py capture
.venv/bin/python supervised_so101.py check
.venv/bin/python supervised_so101.py run
```

Before capture, put/support the follower at the intended clear rest pose. Capture only reads positions; it does not change torque/configuration. Inspect the printed values and confirm saving. Existing rest files are never overwritten. The saved pose is bound to the exact follower calibration hash. A new calibration requires recapture. `check` validates files/settings without connecting hardware; it cannot verify actual motor-side calibration until connection.

Live connection briefly disables follower torque before enabling measured-pose holding (same connector as the prior alignment utility). Support the arm during connection. No return trajectory is issued automatically on launch. If not already near rest, it stays in **setup_hold**: inspect the path and press S to return to rest. Space is ignored until Idle. This makes startup initialization explicit.

Keys (terminal must have focus, no Enter needed):

- **Space:** Idle → Lagging; “Slow down” including startup.
- **P:** Tracking/Lagging → Awaiting alignment; capture current measured follower pose as hold target; “Pick up.”
- **S:** controlled return to saved rest, then Idle. Ignored during a latched halt/fault. Repeated S does not restart the return timer. Space is ignored during return.
- **Esc:** latch Halted, send one fresh measured-pose holding target, no return home. Restart session to recover. This is a software halt, not a hardware emergency stop.
- **Q:** exit. The runner asks you to support the follower before Enter disconnects and releases torque. Ctrl+C also goes through cleanup. Loss of terminal/process/power cannot guarantee holding.

Suggested sequence: inspect setup/Idle; Space with arms already close; move leader gently; P and move leader into alignment; observe automatic resume after dwell; S with clear return path; Esc and Q. Stop on unexpected motion. Do not deliberately jam the arm to test nonprogress in the first trial. Calibrated boundaries are not collision-free workspace guarantees; return-to-rest is joint-space interpolation, not collision planning.

## Initial settings and known limits

Defaults are trial parameters, not physically validated thresholds: arm alignment 1.5°, lag 5°, gripper alignment 3 / lag 6 opening units; alignment dwell .7s; lag dwell .3s; 1s signed-progress window; nonprogress under .5 units over that window for 3s while error exceeds lag threshold. Boundary tolerance .5 units, outward direction noise tolerance .1 units. Command speed 5°/s and gripper 10 units/s; acceleration parameter 10°/s² and gripper 20 units/s². Setpoint shaping does not guarantee physical speed/acceleration. State-entry holds intentionally bypass trajectory smoothing to replace the previous target.

The previous elbow residual was 3.43°, so 1.5° alignment may not be achievable at that pose. Do not hide this by silently widening gates: inspect the displayed errors and choose a different clear pose or explicitly review a per-joint override. Lagging may persist if error is above alignment tolerance but below the nonprogress error threshold.

Pass `--settings /path/settings.json` to `run` or `check` for explicit overrides. Per-joint dictionaries merge with defaults, e.g. `{"alignment":{"elbow_flex":4.0}}`; this example changes what “aligned” means and is not a recommended default. All active joints, including gripper, participate in alignment.

Known boundary: follower near calibrated upper/lower position AND leader-derived target points outward, checked before clipping. Either Tracking or Lagging enters Awaiting alignment directly. Otherwise unexplained error goes Tracking → Lagging → Awaiting alignment only after sustained nonprogress. This heuristic can misclassify gravity/friction or moving-target behavior; it does not prove obstruction. Logs preserve the reason and raw positions.

Awaiting alignment holds the measured pose captured on entry, not the unreachable leader target. Holding can still sustain force or settle under gravity; it is not a verified compliant/contact recovery policy. Resume automatically after alignment dwell, seeded from the current follower position and then shaped. No manual resume key.

Feedback gaps over .25s, nonzero sampled motor Status, unexpected torque-disable, return nonprogress/30s timeout latch Fault. Serial exceptions end normal following: a fresh measured-pose hold is attempted only if available; no blind recovery or return. With unavailable feedback/communications, the previous servo goal may persist until operator intervention. Status/torque are sampled every .5s by the same serial owner. Fault/Halted cannot be cleared by Space/S/P.

## Evidence and replay

Per-sample JSONL includes leader/follower positions, state, previous state, reason, errors, detected boundary, keyboard events, shaped command and returned sent action. Metadata includes source revision, hardware config, bounds, settings, saved rest and both calibration hashes. Logs include motor status/torque polls, speech requests/process exits (not proof of audibility), writer drop counts and exceptions. Default output is next to hardware.json with a unique name.

Replay old angular-warning telemetry without USB:

```sh
.venv/bin/python -m lerobot.scripts.lerobot_so101_supervised \
  --config outputs/lab/so101-20261023/hardware.json \
  --rest outputs/lab/so101-20261023/rest-pose.json \
  --replay outputs/lab/so101-20261023/teleop-voice_20261023_140617_77951.telemetry.jsonl \
  --output outputs/development/replay-new.jsonl
```

Replay injects Start at the first observed sample and uses recorded measurements regardless of hypothetical commands. Old telemetry excluded gripper, so replay excludes gripper too. It does not simulate resulting motion, pickup, or new operator behavior. Output is created exclusively, never overwritten.

The first historical replay processed 1,347 samples: Lagging at sample 1, Awaiting alignment at sample 1073 for shoulder/elbow nonprogress. It used a synthetic zero rest only for offline initialization, not as a physical rest recommendation. First recorded elbow position lay beyond the calibrated endpoint, while the leader requested inward motion; the outward-boundary guard correctly did not trigger merely for being outside/near the endpoint.

Recording-gate integration remains future work. Reuse the Controller rather than reimplementing transitions inside record.py; define interrupted-frame semantics before enabling dataset recording.

Entering Tracking announces “Tracking” once, including pickup resume. It bypasses the repeating-warning cooldown, waits for an active short speech process to finish, and cancels if Tracking is left before delivery. Restart the runner to load code updates; a running process keeps its loaded version.

Entering Idle announces “Stopping” once (including initial entry at rest and arrival after return-to-rest). It uses the same nonblocking state-entry delivery policy as “Tracking.”

## Pre-alignment motor history

The runner continuously rotates through Torque_Enable, Present_Current, Present_Load, Present_Temperature, Present_Voltage, Present_Velocity, Status and Moving for every follower motor. It performs at most one additional register read per eligible control cycle (minimum spacing .05s, so a full eight-register pass usually takes roughly .5s at 30Hz). Reads run on the existing serial-owner thread with no retries; errors remain explicit unavailable readings. Existing .5s fault polling is unchanged.

Every Lagging → Awaiting alignment transition writes a `pre_alignment_motor_history` event into the same JSONL, containing the preceding 3s of register reads plus transition reason, positions, errors and previous commanded target. It covers nonprogress, calibrated-boundary and keyboard-disengage transitions from Lagging. Capturing the buffer requires no additional serial reads at the transition. `motor_register` records also preserve individual reads throughout the run. Values are timestamped separately, not simultaneous samples or calibrated torque measurements. If the transition occurs within the first 3s, `startup_truncated` marks incomplete history. Restart the runner to enable this logging; missing load/current cannot be reconstructed from earlier runs.

## Advisory operational capability monitor

Enabled by default; restart `supervised_so101.py run`. Every ~.1s it calculates gravity torque at measured and requested poses using the pinned public URDF, NumPy and COM lever arms. No MuJoCo or extra serial thread is needed in the robot environment. Console `CAPABILITY advisory` lists affected joints. JSONL `capability_monitor` records contain per-joint torque, position margin, nominal 12V/7.4V continuous-torque utilization, predicted PD static error, model disagreement, sampled voltage and effort/response evidence. Pre-alignment snapshots include the most recent prediction. This does not change motion commands, state transitions or speech.

Predictions: `near_position_boundary`, `calibrated_position_boundary` (outward request within advisory 2-unit margin), `possible_load_boundary` (either nominal model predicts >=80% continuous holding capability or >=5° static error), `suspected_response_boundary` (sustained empirical pattern), or `load_and_response_warning`. This advisory position margin is broader than the controller's .5-unit trigger and never overrides it. `no_warning` is not proof of safe/reachable space. No collision or general workspace model is provided.

Continuous torque assumptions are supplier rated loads: 12V 10kg.cm, 7.4V 5kg.cm; stiffness uses the same unvalidated PD-to-duty conversion as the simulation. Both are nominal models. Voltage more than 15% from both nominal values is explicitly flagged; no unvalidated voltage derating is applied. Predicted error is a modeled static joint-angle error; gripper geometry margins and empirical errors use opening units. Gravity estimates exclude acceleration torque, external payload changes and contact.

Empirical detector uses *sent command versus measured state*, not raw leader backlog. Over a 2s window: error >=3 units grows by >=1 unit in the same corrective direction, commanded target moves at least .5 unit farther in that direction, follower makes <.5 unit net corrective progress, and load OR current remains flat/nonzero with >=3 distinct readings covering >=1s. Flat thresholds: load range <=40 with magnitude >=100; current range <=8 with magnitude >=5. Readings must be <=.8s old. Candidate must persist .7s. These are initial empirical settings, not identified physical limits. Constant effort with constant error is deliberately not sufficient. Signs of load/current are not mapped to joint torque direction. Diagnostic read failures/stale readings suppress evidence, never become zero effort. Monitoring resets its empirical history outside Tracking/Lagging or after time gaps.

Use `--capability-settings /path/file.json` with top-level `monitor` (MonitorConfig fields), `servo_models` (named dicts with voltage_v, continuous_torque_nm, stiffness_nm_rad), optional `offsets_deg` (joint dict) and `urdf` to make assumptions explicit. `--no-capability-monitor` disables only this advisory. Settings/model hash are recorded in metadata. The model path is available in the editable checkout; configuration fails before hardware connection if missing/invalid.

Pickup display now reports `PICKUP n/6 aligned`, live follower targets and signed leader corrections for each joint outside tolerance, followed by “hold leader still” when alignment dwell can begin. Signs refer to calibrated angle/opening coordinates, not an assumed physical up/down direction. Tolerances and automatic resume behavior are unchanged.

Position-boundary advisories now require an outward leader-derived request during active Tracking/Lagging: upper-side target > measured +.1 unit, or lower-side target < measured -.1 unit. Near-boundary holding or inward requests stay silent. Proximity remains logged separately; the .5-unit state-machine guard is unchanged.

Load-boundary advisories now also require active tracking toward greater absolute modeled gravity torque: `abs(g(requested)) - abs(g(measured)) > .01 Nm` (configurable `load_increase_tolerance_nm`). Holding, lower-load requests and sub-tolerance changes stay silent. Raw utilization/static-error predictions remain logged even when suppressed. This compares endpoint poses, not a collision/path-load guarantee; empirical effort-response warnings remain independent. No motion-state changes.
