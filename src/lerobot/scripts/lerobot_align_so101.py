"""Supervised, fixed-pose SO101 startup alignment and live inspection.

Uses existing calibration, no cameras, no speech, no automatic teleoperation.
Goal slew limits do not guarantee physical joint speed or a collision-free path.
"""

import argparse
import hashlib
import json
import math
import select
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.so101_motor_diagnostics import MotorDiagnostics, diagnostic_value

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JOINTS = (*ARM_JOINTS, "gripper")


def ramp_goal(start, target, elapsed, speed, gripper_speed):
    result = {}
    for joint in JOINTS:
        delta = target[joint] - start[joint]
        limit = (gripper_speed if joint == "gripper" else speed) * elapsed
        result[joint] = start[joint] + max(-limit, min(limit, delta))
    return result


def alignment_errors(target, live_leader, follower, tolerance, gripper_tolerance, hold_gripper=False):
    errors = {j: target[j] - follower[j] for j in JOINTS}
    follower_ok = all(abs(errors[j]) <= (gripper_tolerance if j == "gripper" else tolerance) for j in JOINTS)
    leader_ok = all(
        (hold_gripper and j == "gripper")
        or abs(target[j] - live_leader[j]) <= (gripper_tolerance if j == "gripper" else tolerance)
        for j in JOINTS
    )
    return errors, follower_ok, leader_ok


def read_positions(device):
    values = device.bus.sync_read("Present_Position")
    if set(values) != set(JOINTS) or not all(math.isfinite(float(v)) for v in values.values()):
        raise RuntimeError("Incomplete or non-finite joint position feedback")
    return {j: float(values[j]) for j in JOINTS}


def connect_follower_holding_present_pose(robot):
    """Seed the measured position before the standard configuration enables torque."""
    robot.bus.connect()
    if not robot.is_calibrated:
        raise RuntimeError("Follower calibration differs from the motors; recalibrate explicitly first")
    robot.bus.disable_torque()
    position = read_positions(robot)
    robot.bus.sync_write("Goal_Position", position)
    robot.configure()
    return position


def display(state, elapsed, target, live, actual, motor_goal, errors, motor_states=None):
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    print(f"{state} | {elapsed:.1f}s | fixed target; Enter/Ctrl+C exits and releases follower torque")
    print(f"{'joint':<17} {'target':>9} {'leader':>9} {'follower':>9} {'motor goal':>11} {'error':>9}")
    for joint in JOINTS:
        print(
            f"{joint:<17} {target[joint]:9.2f} {live[joint]:9.2f} {actual[joint]:9.2f} "
            f"{motor_goal[joint]:11.2f} {errors[joint]:9.2f}"
        )
    print("Arm values: degrees. Gripper: calibrated 0–100 opening units.")
    if motor_states is not None:
        print("\nFollower motor states (raw register values; fields refreshed separately):")
        print(f"{'joint':<17} {'torque':>7} {'current':>8} {'load':>8} {'temp':>7} {'status':>8}")
        for joint in JOINTS:
            values = []
            for register in (
                "Torque_Enable",
                "Present_Current",
                "Present_Load",
                "Present_Temperature",
                "Status",
            ):
                value = diagnostic_value(motor_states, register, joint)
                values.append(
                    "N/A" if value is None else (hex(value) if register == "Status" else str(value))
                )
            print(f"{joint:<17} {values[0]:>7} {values[1]:>8} {values[2]:>8} {values[3]:>7} {values[4]:>8}")

        def elbow(register):
            value = diagnostic_value(motor_states, register, "elbow_flex")
            return "N/A" if value is None else str(value)

        print(
            "Elbow P/I/D="
            + "/".join(elbow(r) for r in ("P_Coefficient", "I_Coefficient", "D_Coefficient"))
            + "  dead zones CW/CCW="
            + "/".join(elbow(r) for r in ("CW_Dead_Zone", "CCW_Dead_Zone"))
        )
        failed = [name for name, record in motor_states.items() if record["error"]]
        if failed:
            print("Unavailable diagnostic registers: " + ", ".join(failed))
        print("Raw current/load/temperature/status are not converted or interpreted as motor faults.")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate config without connecting hardware")
    parser.add_argument("--speed-deg-s", type=float, default=5.0)
    parser.add_argument("--gripper-speed", type=float, default=10.0)
    parser.add_argument("--tolerance-deg", type=float, default=1.0)
    parser.add_argument("--gripper-tolerance", type=float, default=3.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--max-start-error-deg", type=float, default=30.0)
    parser.add_argument("--hold-gripper", action="store_true", help="Keep gripper at its starting opening")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument(
        "--motor-state-interval-s",
        type=float,
        default=0.15,
        help="Minimum interval between individual diagnostic register reads",
    )
    args = parser.parse_args()
    for name in (
        "speed_deg_s",
        "gripper_speed",
        "tolerance_deg",
        "gripper_tolerance",
        "settle_s",
        "timeout_s",
        "max_start_error_deg",
        "motor_state_interval_s",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    config = json.loads(args.config.read_text())
    options = {}
    for key, expected in (("robot", "so101_follower"), ("teleop", "so101_leader")):
        item = config[key].copy()
        if item.pop("type") != expected or not item.get("use_degrees", True):
            parser.error("Requires so101 follower/leader with degree-based arm joints")
        directory = item.get("calibration_dir")
        if not directory:
            parser.error("Set explicit calibration_dir paths in the hardware config")
        item["calibration_dir"] = Path(directory)
        if not (item["calibration_dir"] / (item["id"] + ".json")).is_file():
            parser.error(f"Missing {key} calibration file")
        options[key] = item
    options["robot"]["cameras"] = {}
    options["robot"]["max_relative_target"] = None
    options["robot"]["disable_torque_on_disconnect"] = True
    robot_config = SO101FollowerConfig(**options["robot"])
    leader_config = SO101LeaderConfig(**options["teleop"])
    if args.check:
        print("PASS alignment configuration and calibration files; no hardware connected")
        return
    print("This connects/configures both arms, releases torque briefly, then enables follower holding.")
    print(
        "Clear the swept space, remove grasped objects, place the leader in the desired pose, and support the follower."
    )
    print(
        f"The gripper {'will hold its initial opening' if args.hold_gripper else 'will also align and may close'}."
    )
    input("Press Enter when ready to connect. Ctrl+C cancels. ")
    robot = SO101Follower(robot_config)
    leader = SO101Leader(leader_config)
    directory = args.log_dir or args.config.resolve().parent / "alignment"
    directory.mkdir(parents=True, exist_ok=True)
    logpath = (
        directory / f"alignment_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{time.time_ns()}.jsonl"
    )
    checkout = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    with logpath.open("x") as initial_log:
        calibration_hashes = {}
        for name, item in options.items():
            calibration_file = item["calibration_dir"] / (item["id"] + ".json")
            calibration_hashes[name] = hashlib.sha256(calibration_file.read_bytes()).hexdigest()
        initial_log.write(
            json.dumps(
                {
                    "type": "session_start",
                    "source_revision": revision.stdout.strip(),
                    "config": config,
                    "calibration_sha256": calibration_hashes,
                }
            )
            + "\n"
        )
    aligned_seen = False
    outcome = "interrupted"
    try:
        leader.bus.connect()
        if not leader.is_calibrated:
            raise RuntimeError("Leader calibration differs from the motors; recalibrate explicitly first")
        leader.configure()
        start_pose = connect_follower_holding_present_pose(robot)
        target = read_positions(leader)
        if args.hold_gripper:
            target["gripper"] = start_pose["gripper"]
        for joint in ARM_JOINTS:
            if abs(target[joint] - start_pose[joint]) > args.max_start_error_deg:
                raise RuntimeError(f"{joint} starts too far from the leader; reposition before rerunning")
            cal = robot.calibration[joint]
            half_range = (cal.range_max - cal.range_min) / 2 * 360 / 4095
            if abs(target[joint]) > half_range:
                raise RuntimeError(f"{joint} target lies outside follower's calibrated range")
        follower_diagnostics = MotorDiagnostics(robot.bus, args.motor_state_interval_s)
        leader_diagnostics = MotorDiagnostics(leader.bus, args.motor_state_interval_s)
        with logpath.open("a") as diagnostic_log:
            diagnostic_log.write(
                json.dumps(
                    {
                        "type": "motor_configuration_snapshot",
                        "units": "raw_register_units_after_driver_sign_decode",
                        "follower": follower_diagnostics.snapshot(),
                        "leader": leader_diagnostics.snapshot(),
                    }
                )
                + "\n"
            )
        live = read_positions(leader)
        initial_actual = read_positions(robot)
        initial_goal = robot.bus.sync_read("Goal_Position")
        errors, _, _ = alignment_errors(
            target, live, initial_actual, args.tolerance_deg, args.gripper_tolerance
        )
        display(
            "INITIAL SNAPSHOT — NOT YET MOVING TO LEADER",
            0,
            target,
            live,
            initial_actual,
            initial_goal,
            errors,
            follower_diagnostics.latest,
        )
        with logpath.open("a") as snapshot_log:
            snapshot_log.write(
                json.dumps(
                    {
                        "type": "initial_snapshot",
                        "target": target,
                        "follower": initial_actual,
                        "motor_goal_readback": initial_goal,
                        "leader_live": live,
                        "errors": errors,
                    }
                )
                + "\n"
            )
        input("Inspect these differences. Press Enter to start gradual alignment, or Ctrl+C to cancel. ")
        # Refresh the follower starting pose; the leader target remains the displayed snapshot.
        start_pose = read_positions(robot)
        if any(abs(target[j] - start_pose[j]) > args.max_start_error_deg for j in ARM_JOINTS):
            raise RuntimeError("Follower moved too far while inspecting; reposition and rerun")
        motor_goal = start_pose.copy()
        begun = time.monotonic()
        last_print = -1.0
        last_goal_read = -1.0
        aligned_since = None
        with logpath.open("a") as log:
            log.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "config_path": str(args.config.resolve()),
                        "target": target,
                        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    }
                )
                + "\n"
            )
            while True:
                tick = time.monotonic()
                elapsed = tick - begun
                actual = read_positions(robot)
                live = read_positions(leader)
                command = ramp_goal(start_pose, target, elapsed, args.speed_deg_s, args.gripper_speed)
                robot.send_action({f"{j}.pos": value for j, value in command.items()})
                if elapsed - last_goal_read >= 0.5:
                    motor_goal = robot.bus.sync_read("Goal_Position")
                    last_goal_read = elapsed
                diagnostic = follower_diagnostics.poll(time.monotonic())
                if diagnostic is not None:
                    log.write(json.dumps({"type": "motor_state", "arm": "follower", **diagnostic}) + "\n")
                errors, follower_ok, leader_ok = alignment_errors(
                    target, live, actual, args.tolerance_deg, args.gripper_tolerance, args.hold_gripper
                )
                ramp_done = all(abs(command[j] - target[j]) < 1e-6 for j in JOINTS)
                ready = ramp_done and follower_ok and leader_ok
                aligned_since = (aligned_since if aligned_since is not None else tick) if ready else None
                aligned = aligned_since is not None and tick - aligned_since >= args.settle_s
                state = (
                    "ALIGNED — INSPECT"
                    if aligned
                    else (
                        "LEADER MOVED FROM SNAPSHOT"
                        if not leader_ok
                        else ("SETTLING" if ramp_done else "RAMPING")
                    )
                )
                aligned_seen |= aligned
                log.write(
                    json.dumps(
                        {
                            "type": "sample",
                            "monotonic_s": tick,
                            "elapsed_s": elapsed,
                            "state": state,
                            "leader_live": live,
                            "follower": actual,
                            "commanded_goal": command,
                            "motor_goal_readback": motor_goal,
                            "goal_read_elapsed_s": last_goal_read,
                            "errors": errors,
                            "aligned": aligned,
                        }
                    )
                    + "\n"
                )
                if elapsed - last_print >= 0.5:
                    display(
                        state, elapsed, target, live, actual, motor_goal, errors, follower_diagnostics.latest
                    )
                    log.flush()
                    last_print = elapsed
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    outcome = "operator_exit_aligned" if aligned else "operator_exit_not_aligned"
                    break
                if elapsed >= args.timeout_s:
                    outcome = "timeout_aligned" if aligned else "timeout_not_aligned"
                    print(f"{outcome}: ending and releasing follower torque; support the arm.")
                    break
                time.sleep(max(0, 0.05 - (time.monotonic() - tick)))
            log.write(
                json.dumps({"type": "summary", "outcome": outcome, "aligned_seen": aligned_seen}) + "\n"
            )
    except KeyboardInterrupt:
        print("Stopped by operator.")
        with logpath.open("a") as log:
            log.write(json.dumps({"type": "interrupted", "aligned_seen": aligned_seen}) + "\n")
    except Exception as exc:
        with logpath.open("a") as log:
            log.write(json.dumps({"type": "error", "message": repr(exc)}) + "\n")
        raise
    finally:
        # Each cleanup is attempted even if another bus fails to disconnect.
        try:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=True)
        finally:
            if leader.bus.is_connected:
                leader.bus.disconnect()
        print(f"Alignment log: {logpath}")
    if outcome.endswith("not_aligned"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
