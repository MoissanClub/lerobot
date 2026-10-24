"""SO101 supervised state-machine trial, offline replay and explicit rest-pose capture."""

import argparse
import hashlib
import json
import os
import select
import subprocess
import sys
import termios
import time
import tty
from dataclasses import asdict
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_align_so101 import JOINTS, connect_follower_holding_present_pose, read_positions
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.joint_telemetry import JointTelemetryWriter
from lerobot.utils.so101_capability_monitor import CapabilityMonitor, GravityModel, MonitorConfig
from lerobot.utils.so101_motor_history import MotorStateHistory
from lerobot.utils.so101_state_machine import Config, Controller, State


def load_setup(path):
    data = json.loads(path.read_text())
    options = {}
    for key, kind in (("robot", "so101_follower"), ("teleop", "so101_leader")):
        item = data[key].copy()
        if item.pop("type") != kind or not item.get("use_degrees", True):
            raise ValueError("Requires degree-based SO101 leader/follower")
        item["calibration_dir"] = Path(item["calibration_dir"])
        calibration = item["calibration_dir"] / (item["id"] + ".json")
        values = json.loads(calibration.read_text())
        if set(values) != set(JOINTS):
            raise ValueError("Incomplete calibration")
        options[key] = item
    options["robot"].update(cameras={}, max_relative_target=None, disable_torque_on_disconnect=True)
    return options


def bounds_from_calibration(calibration):
    return {
        j: (0.0, 100.0)
        if j == "gripper"
        else (-(v["range_max"] - v["range_min"]) * 180 / 4095, (v["range_max"] - v["range_min"]) * 180 / 4095)
        for j, v in calibration.items()
    }


def trial_config():
    return Config(
        alignment={j: 3.0 if j == "gripper" else 1.5 for j in JOINTS},
        lag={j: 6.0 if j == "gripper" else 5.0 for j in JOINTS},
        speed={j: 10.0 if j == "gripper" else 5.0 for j in JOINTS},
        acceleration={j: 20.0 if j == "gripper" else 10.0 for j in JOINTS},
    )


def replay(path, output, rest, bounds, cfg):
    controller = Controller(rest, bounds, cfg)
    controller.state = State.IDLE
    count, transitions = 0, []
    with output.open("x") as out:
        for line in path.open():
            sample = json.loads(line)
            if sample.get("type") != "sample":
                continue
            leader = {j.removesuffix(".pos"): v for j, v in sample["leader_target"].items()}
            follower = {j.removesuffix(".pos"): v for j, v in sample["follower_position"].items()}
            # Old warning telemetry excluded gripper; replay only the observed joints.
            leader, follower = ({j: p[j] for j in rest} for p in (leader, follower))
            decision = controller.step(
                sample["monotonic_s"], leader, follower, ["start"] if count == 0 else []
            )
            record = {"type": "replay", "sample_index": sample.get("sample_index"), **asdict(decision)}
            out.write(json.dumps(record) + "\n")
            if decision.state != decision.previous:
                transitions.append(
                    {"sample": sample.get("sample_index"), "state": decision.state, "reason": decision.reason}
                )
            count += 1
    print(json.dumps({"samples": count, "transitions": transitions, "output": str(output)}, indent=2))


def pickup_guidance(leader, follower, tolerances):
    joints = {
        j: {
            "leader": leader[j],
            "follower_target": follower[j],
            "move_leader_by": follower[j] - leader[j],
            "tolerance": tolerances[j],
            "aligned": abs(follower[j] - leader[j]) <= tolerances[j],
        }
        for j in JOINTS
    }
    count = sum(v["aligned"] for v in joints.values())
    return {
        "aligned_joints": count,
        "total_joints": len(joints),
        "hold_still": count == len(joints),
        "joints": joints,
    }


class TerminalKeys:
    def __enter__(self):
        self.original = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def read(self):
        mapping = {" ": "start", "s": "stop", "p": "disengage", "\x1b": "halt", "q": "quit"}
        events = []
        while select.select([sys.stdin], [], [], 0)[0]:
            for ch in os.read(sys.stdin.fileno(), 64).decode("utf-8", errors="ignore"):
                if ch in mapping:
                    events.append(mapping[ch])
        return events

    def __exit__(self, *exc):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original)


class Speaker:
    """Single nonblocking speech process; no audio queue or serial worker."""

    def __init__(self, enabled, log):
        self.enabled, self.log, self.process = enabled, log, None
        self.last, self.last_message = float("-inf"), None
        self.reported = True
        self.state = None
        self.entry_prompt = None

    def update(self, now, state):
        if state != self.state:
            self.entry_prompt = {"tracking": "Tracking", "idle": "Stopping"}.get(state)
            self.state = state
        if self.process and self.process.poll() is not None and not self.reported:
            self.log.submit(
                {
                    "type": "speech_process_exit",
                    "returncode": self.process.returncode,
                    "text": self.last_message,
                    "audibility": "not_verified",
                }
            )
            self.reported = True
        text = (
            self.entry_prompt
            if self.entry_prompt
            else {"lagging": "Slow down", "awaiting_alignment": "Pick up"}.get(state)
        )
        if (
            not text
            or (not self.entry_prompt and now - self.last < 5)
            or (self.process and self.process.poll() is None)
        ):
            return
        self.entry_prompt = None
        self.last, self.last_message = now, text
        print(f"PROMPT: {text}", flush=True)
        self.log.submit(
            {"type": "prompt_request", "monotonic_s": now, "text": text, "audio_enabled": self.enabled}
        )
        if self.enabled:
            self.process = subprocess.Popen(["say", text])
            self.reported = False

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--rest", type=Path)
    p.add_argument("--capture-rest", type=Path, help="Read-only measurement, no torque/configuration writes")
    p.add_argument("--settings", type=Path, help="JSON overrides for Config fields")
    p.add_argument("--check", action="store_true")
    p.add_argument("--replay", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--silent", action="store_true")
    p.add_argument("--no-capability-monitor", action="store_true")
    p.add_argument(
        "--capability-settings", type=Path, help="Advisory monitor JSON, separate from motion settings"
    )
    args = p.parse_args()
    options = load_setup(args.config)
    calpath = options["robot"]["calibration_dir"] / (options["robot"]["id"] + ".json")
    calibration_hash = hashlib.sha256(calpath.read_bytes()).hexdigest()
    bounds = bounds_from_calibration(json.loads(calpath.read_text()))
    cfg_data = asdict(trial_config())
    if args.settings:
        overrides = json.loads(args.settings.read_text())
        for key, value in overrides.items():
            if isinstance(cfg_data.get(key), dict):
                cfg_data[key].update(value)
            else:
                cfg_data[key] = value
    cfg = Config(**cfg_data)
    cfg.validate(JOINTS)
    if args.capture_rest:
        if args.capture_rest.exists():
            p.error("Rest file already exists; choose a new path")
        robot = SO101Follower(SO101FollowerConfig(**options["robot"]))
        input(
            "Place/support follower in a verified clear rest pose. Enter reads position without changing torque. "
        )
        try:
            robot.bus.connect()
            if not robot.is_calibrated:
                raise RuntimeError("Calibration mismatch")
            rest = read_positions(robot)
            Controller(rest, bounds, cfg)
            print(json.dumps(rest, indent=2))
            input("Inspect pose and values. Enter saves as rest pose; Ctrl+C cancels. ")
            args.capture_rest.parent.mkdir(parents=True, exist_ok=True)
            with args.capture_rest.open("x") as f:
                json.dump({"pose": rest, "follower_calibration_sha256": calibration_hash}, f, indent=2)
        finally:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=False)
        return
    if not args.rest:
        p.error("--rest required; first use --capture-rest")
    saved = json.loads(args.rest.read_text())
    if saved["follower_calibration_sha256"] != calibration_hash:
        p.error("Rest pose belongs to different calibration; capture a new rest pose")
    controller = Controller(saved["pose"], bounds, cfg)
    capability = None
    capability_config = {}
    if not args.no_capability_monitor:
        capability_config = (
            json.loads(args.capability_settings.read_text()) if args.capability_settings else {}
        )
        model_path = Path(
            capability_config.get(
                "urdf",
                Path(__file__).resolve().parents[3] / "simulation/so101_gravity/vendor/so101_new_calib.urdf",
            )
        )
        gravity = GravityModel(
            model_path, json.loads(calpath.read_text()), capability_config.get("offsets_deg")
        )
        capability = CapabilityMonitor(
            gravity,
            bounds,
            MonitorConfig(**capability_config.get("monitor", {})),
            capability_config.get("servo_models"),
        )
        capability_config = {
            "urdf": str(model_path),
            "urdf_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "monitor": asdict(capability.config),
            "servo_models": capability.models,
            "offsets_deg": capability_config.get("offsets_deg", {}),
            "advisory_only": True,
        }

    if args.replay:
        if not args.output:
            p.error("--output required for replay")
        keys = [j for j in JOINTS if j != "gripper"]
        subset = {k: {j: v[j] for j in keys} if isinstance(v, dict) else v for k, v in asdict(cfg).items()}
        replay(
            args.replay,
            args.output,
            {j: saved["pose"][j] for j in keys},
            {j: bounds[j] for j in keys},
            Config(**subset),
        )
        return
    if args.check:
        print("PASS config, calibration, rest-pose hash, controller settings; no hardware connected")
        return
    if not sys.stdin.isatty():
        p.error("Live mode requires an interactive terminal")
    robot = SO101Follower(SO101FollowerConfig(**options["robot"]))
    leader = SO101Leader(SO101LeaderConfig(**options["teleop"]))
    path = args.output or args.config.parent / f"state-machine-{time.time_ns()}.jsonl"
    revision = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    hashes = {
        key: hashlib.sha256((item["calibration_dir"] / (item["id"] + ".json")).read_bytes()).hexdigest()
        for key, item in options.items()
    }
    writer = JointTelemetryWriter(
        str(path),
        {
            "config": json.loads(args.config.read_text()),
            "settings": asdict(cfg),
            "rest": saved,
            "bounds": bounds,
            "source_revision": revision,
            "calibration_sha256": hashes,
            "recording": False,
            "capability_monitor": capability_config if capability else {"enabled": False},
            "motor_history": {
                "window_s": 3.0,
                "register_interval_s": 0.05,
                "trigger": "lagging_to_awaiting_alignment",
            },
        },
    )
    speaker = Speaker(not args.silent, writer)
    print("Space=start; P=disengage; S=return to rest; Esc=latched halt; Q=exit (releases torque).")
    print("Support arm during connection/exit. Rest return is a joint-space path, not collision planning.")
    actual, actual_time = None, None
    try:
        input("Clear workspace and support follower. Enter connects and enables measured-pose holding. ")
        leader.bus.connect()
        if not leader.is_calibrated:
            raise RuntimeError("Leader calibration mismatch")
        leader.configure()
        connect_follower_holding_present_pose(robot)
        motor_history = MotorStateHistory(robot.bus)
        capability_result = None
        last_capability = float("-inf")
        last_display = 0
        last_status = 0
        latched_reported = False
        with TerminalKeys() as keyboard:
            while True:
                tick = time.monotonic()
                events = keyboard.read()
                if "quit" in events:
                    break
                actual = read_positions(robot)
                actual_time = time.monotonic()
                target = read_positions(leader)
                motor_fault = None
                if tick - last_status >= 0.5:
                    status = robot.bus.sync_read("Status", normalize=False, num_retry=0)
                    torque = robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=0)
                    writer.submit(
                        {
                            "type": "motor_state",
                            "monotonic_s": time.monotonic(),
                            "status": status,
                            "torque_enable": torque,
                        }
                    )
                    if any(status.values()) or any(v != 1 for v in torque.values()):
                        motor_fault = "motor_status_or_torque_fault"
                    last_status = tick
                diagnostic = motor_history.poll(time.monotonic())
                if diagnostic is not None:
                    writer.submit({"type": "motor_register", **diagnostic})
                if capability and tick - last_capability >= capability.config.interval_s:
                    capability_result = capability.update(
                        time.monotonic(),
                        actual,
                        target,
                        controller.command or actual,
                        motor_history.diagnostics.latest,
                        active=controller.state in (State.LAGGING, State.TRACKING),
                    )
                    writer.submit(capability_result)
                    last_capability = tick
                transition_time = time.monotonic()
                previous_command = dict(controller.command) if controller.command is not None else None
                decision = controller.step(transition_time, target, actual, events, fault=motor_fault)
                if decision.previous == "lagging" and decision.state == "awaiting_alignment":
                    writer.submit(
                        {
                            "type": "pre_alignment_motor_history",
                            "previous": decision.previous,
                            "state": decision.state,
                            "reason": decision.reason,
                            "capability": capability_result,
                            "leader": target,
                            "follower": actual,
                            "previous_command": previous_command,
                            "errors": decision.errors,
                            **motor_history.snapshot(transition_time),
                        }
                    )
                sent = None
                if decision.command is not None:
                    sent = robot.send_action({f"{j}.pos": v for j, v in decision.command.items()})
                writer.submit(
                    {
                        "type": "sample",
                        "monotonic_s": tick,
                        "wall_time_ns": time.time_ns(),
                        "leader": target,
                        "follower": actual,
                        "events": events,
                        "sent_action": sent,
                        "pickup_guidance": pickup_guidance(target, actual, cfg.alignment)
                        if decision.state == "awaiting_alignment"
                        else None,
                        **asdict(decision),
                    }
                )
                if decision.previous != decision.state:
                    print(f"{decision.previous} -> {decision.state}: {decision.reason}", flush=True)
                if tick - last_display >= 0.5:
                    error_text = " ".join(f"{j}={v:+.1f}" for j, v in decision.errors.items())
                    print(f"{decision.state}: {error_text}", flush=True)
                    if decision.state == "awaiting_alignment":
                        guide = pickup_guidance(target, actual, cfg.alignment)
                        print(
                            f"PICKUP {guide['aligned_joints']}/{guide['total_joints']} aligned. Move LEADER toward live FOLLOWER:",
                            flush=True,
                        )
                        for j, values in guide["joints"].items():
                            if not values["aligned"]:
                                print(
                                    f"  {j}: leader {values['leader']:.1f} -> follower {values['follower_target']:.1f}; move leader {values['move_leader_by']:+.1f}",
                                    flush=True,
                                )
                        if guide["hold_still"]:
                            print("Aligned: hold leader still for resume dwell.", flush=True)
                    if capability_result:
                        warnings = [
                            f"{j}:{v['prediction']}"
                            for j, v in capability_result["joints"].items()
                            if v["prediction"] != "no_warning"
                        ]
                        print(
                            "CAPABILITY advisory: "
                            + (" ".join(warnings) if warnings else "no warning; not a safety guarantee"),
                            flush=True,
                        )
                    last_display = tick
                speaker.update(tick, decision.state)
                if controller.state in (State.HALTED, State.FAULT) and not latched_reported:
                    latched_reported = True
                    print(
                        "Latched halt/fault. Support arm, then Q to exit/release torque. Restart required.",
                        flush=True,
                    )
                if writer.error or writer.dropped:
                    raise RuntimeError("Telemetry failed or dropped samples; ending trial")
                time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))
    except (Exception, KeyboardInterrupt) as exc:
        # Hold a recent measured pose if available; never invent a pose after loss of feedback.
        hold_result = "not_attempted_stale_or_missing"
        if (
            actual is not None
            and actual_time is not None
            and time.monotonic() - actual_time <= cfg.max_feedback_gap_s
        ):
            try:
                robot.send_action({f"{j}.pos": v for j, v in actual.items()})
                hold_result = "recent_measured_pose_sent"
            except Exception as hold_exc:
                hold_result = repr(hold_exc)
        writer.submit({"type": "exception_hold", "result": hold_result})
        writer.submit(
            {"type": "fault", "message": repr(exc), "action": "no_return_no_blind_write_then_disconnect"}
        )
        print(f"Session stopped: {exc}. No return trajectory commanded.", flush=True)
        raise
    finally:
        speaker.close()
        # Keep holding until operator supports the arm, even on serial read/write failure.
        try:
            input("Support follower. Enter disconnects and releases follower torque. ")
        finally:
            try:
                if robot.bus.is_connected:
                    robot.bus.disconnect(disable_torque=True)
            finally:
                try:
                    if leader.bus.is_connected:
                        leader.bus.disconnect()
                finally:
                    writer.close()
        print(f"Telemetry: {path}")


if __name__ == "__main__":
    main()
