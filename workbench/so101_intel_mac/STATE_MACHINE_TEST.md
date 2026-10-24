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
