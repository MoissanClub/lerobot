#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Bounded G1-29 arm validation. Defaults to offline dry-run; never enables hands.

Hardware requires a reviewed contract. Simulation exercises the same arm SDK
backend against MuJoCo with fake transport: it does not certify firmware handover.
"""

import argparse
import hashlib
import json
import math
import platform
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from lerobot.robots.unitree_g1.g1_arm_sdk import ARM_KEYS, ARM_SLOTS, G1ArmSDK, G1ArmSDKConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_limits(urdf):
    joints = (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
    root = ET.parse(urdf).getroot()
    names = [f"{side}_{joint}_joint" for side in ("left", "right") for joint in joints]
    limits = {j.get("name"): j.find("limit") for j in root.findall("joint")}
    return {
        field: [float(limits[name].get(field)) for name in names] for field in ("lower", "upper", "effort")
    }


def contract_template(urdf):
    limits = model_limits(urdf)
    cfg = G1ArmSDKConfig(
        network_interface="REPLACE_WITH_ROBOT_INTERFACE",
        read_only=False,
        expected_mode_machine=None,
        kp=[60.0] * 14,
        kd=[1.5] * 14,
        lower=limits["lower"],
        upper=limits["upper"],
        torque_limits=[min(15.0, effort) for effort in limits["effort"]],
        gravity_urdf=str(urdf.absolute()),
        max_feedforward=15.0,
    )
    return {
        "schema": 1,
        "reviewed": False,
        "vr_pose_mapping_reviewed": False,
        "locomotion_reviewed": False,
        "damping_reviewed": False,
        "combined_ownership_reviewed": False,
        "base_limits": {"max_speed": 0.1, "max_yaw": 0.1, "timeout": 0.2},
        "reviewer": "",
        "robot_serial": "",
        "firmware": "",
        "mode_and_manufacturer_procedure": "",
        "support_and_stop_procedure": "",
        "payload_and_tool_model": "",
        "operator_notes": "",
        "robot_side_execution_confirmed": False,
        "exclusive_arm_publisher_confirmed": False,
        "stock_lower_body_owner_confirmed": False,
        "uncommanded_waist_semantics_verified": False,
        "packet_loss_and_process_death_response_verified": False,
        "release_weight_zero_behavior_reviewed": False,
        "gains_limits_and_gravity_reviewed": False,
        "urdf_sha256": digest(urdf),
        "arm_sdk": asdict(cfg),
        "notice": "EXAMPLE VALUES ONLY. Not approved hardware gains, torque ratings or stop limits.",
    }


def validate_contract(doc, urdf, hardware_motion):
    if doc.get("schema") != 1:
        raise ValueError("Unsupported contract schema")
    cfg = G1ArmSDKConfig(**doc["arm_sdk"])
    if urdf is not None:
        if doc.get("urdf_sha256") != digest(urdf):
            raise ValueError("URDF hash differs from reviewed contract")
        limits = model_limits(urdf)
        if np.any(np.asarray(cfg.lower) < limits["lower"]) or np.any(np.asarray(cfg.upper) > limits["upper"]):
            raise ValueError("Contract joint bounds exceed model bounds")
        if np.any(np.asarray(cfg.torque_limits) > limits["effort"]):
            raise ValueError("Contract torque bounds exceed model effort limits")
        if cfg.gravity_urdf and Path(cfg.gravity_urdf).resolve() != urdf.resolve():
            raise ValueError("Gravity model must be the reviewed URDF")
    if hardware_motion:
        if urdf is None or cfg.read_only or cfg.expected_mode_machine is None:
            raise ValueError("Motion needs a reviewed URDF, expected mode and motion configuration")
        for key in (
            "reviewed",
            "robot_side_execution_confirmed",
            "exclusive_arm_publisher_confirmed",
            "stock_lower_body_owner_confirmed",
            "uncommanded_waist_semantics_verified",
            "packet_loss_and_process_death_response_verified",
            "release_weight_zero_behavior_reviewed",
            "gains_limits_and_gravity_reviewed",
        ):
            if doc.get(key) is not True:
                raise ValueError(f"Hardware motion gate not approved: {key}")
        for key in (
            "reviewer",
            "robot_serial",
            "firmware",
            "mode_and_manufacturer_procedure",
            "support_and_stop_procedure",
            "payload_and_tool_model",
        ):
            if not isinstance(doc.get(key), str) or not doc[key].strip():
                raise ValueError(f"Missing reviewed contract field: {key}")
    return cfg


def trajectory(start, index, displacement, fraction):
    target = np.asarray(start).copy()
    target[index] += displacement * (1 - math.cos(2 * math.pi * np.clip(fraction, 0, 1))) / 2
    return target


class SimulationTransport:
    """Physics-in-loop test fixture; only this adapter knows the MuJoCo layout."""

    def __init__(self, urdf):
        self.urdf = urdf
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.command = None
        self.thread = None
        self.sim = None
        self.error = None
        self.writes = 0
        self.last_weight = None

    def connect(self, cfg, callback):
        from lerobot.robots.unitree_g1.g1_vr_control import G1VRKinematics
        from lerobot.robots.unitree_g1.g1_vr_simulation import G1VRSimulation

        self.cfg = cfg
        self.sim = G1VRSimulation(self.urdf.absolute().parents[1], stationary_fixture=True)
        self.kinematics = G1VRKinematics(self.urdf.absolute().parents[1])
        self.initial = np.zeros(14)
        self.stock_tau = self.kinematics.gravity(self.initial)
        for _ in range(100):
            self.sim.step(self.initial, self.stock_tau)
        self.thread = threading.Thread(target=self._run, args=(callback,), daemon=True)
        self.thread.start()

    def enable_writer(self):
        pass

    def wait_ready(self, disabled_command):
        self.write(disabled_command)

    def write(self, command):
        with self.lock:
            self.command = command
            self.writes += 1
            self.last_weight = command.motor_cmd[29].q

    def _run(self, callback):
        sim = self.sim
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                with self.lock:
                    command = self.command
                weight = 0 if command is None else command.motor_cmd[29].q
                target, tau = self.initial.copy(), self.stock_tau.copy()
                if command is not None:
                    cmds = [command.motor_cmd[s] for s in ARM_SLOTS]
                    target = weight * np.array([c.q for c in cmds]) + (1 - weight) * target
                    tau = weight * np.array([c.tau for c in cmds]) + (1 - weight) * tau
                    for i, motor in enumerate(sim.command.motor_cmd[15:]):
                        motor.kp, motor.kd = cmds[i].kp, cmds[i].kd
                sim.step(target, tau)
                msg = sim.state()
                msg.mode_machine = self.cfg.expected_mode_machine or 0
                callback(msg)
                self.stop.wait(max(0.0, self.cfg.period_s - (time.monotonic() - started)))
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(2)
            if self.thread.is_alive():
                raise RuntimeError("Simulation worker did not stop")
        if self.sim is not None:
            self.sim.close()
        if self.error:
            raise RuntimeError(self.error)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("dry-run", "simulation", "hardware"), default="dry-run")
    p.add_argument("--phase", choices=("read-only", "hold", "joint"), default="read-only")
    p.add_argument("--contract", type=Path)
    p.add_argument("--write-template", type=Path)
    p.add_argument("--urdf", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--joint", choices=ARM_KEYS)
    p.add_argument("--delta-rad", type=float)
    p.add_argument("--duration-s", type=float, default=4.0)
    p.add_argument("--enable-motion", action="store_true")
    args = p.parse_args(argv)
    if args.write_template:
        if args.urdf is None or args.backend != "dry-run":
            p.error("Template generation requires --urdf and dry-run")
        return args
    if not args.contract or not args.output:
        p.error("--contract and a new --output file are required")
    if not math.isfinite(args.duration_s) or not 1 <= args.duration_s <= 30:
        p.error("duration-s must be finite and in [1, 30]")
    if args.phase == "joint" and (
        args.joint is None or args.delta_rad is None or not math.isfinite(args.delta_rad)
    ):
        p.error("Joint motion requires --joint and finite --delta-rad")
    if args.phase != "read-only" and args.backend == "hardware" and not args.enable_motion:
        p.error("Hardware motion requires --enable-motion and reviewed contract")
    if args.enable_motion and (args.backend != "hardware" or args.phase == "read-only"):
        p.error("--enable-motion is only for explicit hardware hold/joint tests")
    if args.backend == "simulation" and args.urdf is None:
        p.error("Simulation needs --urdf")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.write_template:
        with args.write_template.open("x") as out:
            json.dump(contract_template(args.urdf), out, indent=2)
        print("Wrote UNREVIEWED template; hardware motion remains disabled.")
        return 0
    report = {
        "status": "incomplete",
        "hardware_acceptance": "pending_operator_review",
        "backend": args.backend,
        "phase": args.phase,
        "motion_enabled": False,
        "python": platform.python_version(),
        "samples": [],
        "errors": [],
    }
    robot = None
    with args.output.open("x") as out:
        json.dump(report, out)
        out.flush()
        try:
            doc = json.loads(args.contract.read_text())
            hardware_motion = args.backend == "hardware" and args.phase != "read-only"
            cfg = validate_contract(doc, args.urdf, hardware_motion)
            cfg = replace(cfg, read_only=args.phase == "read-only")
            report["contract_sha256"] = digest(args.contract)
            report["configuration"] = asdict(cfg)
            import lerobot.robots.unitree_g1.g1_arm_sdk as source

            if (
                not Path(source.__file__)
                .resolve()
                .is_relative_to(Path(__file__).resolve().parents[2] / "src")
            ):
                raise RuntimeError("Diagnostic and imported LeRobot come from different checkouts")
            report["backend_source_sha256"] = digest(source.__file__)
            report["diagnostic_sha256"] = digest(__file__)
            report["source_revision"] = subprocess.check_output(
                ["git", "-C", str(Path(source.__file__).parents[4]), "rev-parse", "HEAD"], text=True
            ).strip()
            report["urdf_sha256"] = digest(args.urdf) if args.urdf else None
            if args.phase == "joint":
                if abs(args.delta_rad) > cfg.max_displacement * 0.8:
                    raise ValueError("Requested displacement needs 20% margin inside envelope")
                if abs(args.delta_rad) * math.pi / args.duration_s > cfg.max_velocity:
                    raise ValueError("Trajectory exceeds velocity bound")
                if 2 * abs(args.delta_rad) * (math.pi / args.duration_s) ** 2 > cfg.max_acceleration:
                    raise ValueError("Trajectory exceeds acceleration bound")
            if args.backend == "dry-run":
                report["status"] = "dry_run_passed"
            else:
                io = SimulationTransport(args.urdf) if args.backend == "simulation" else None
                if io is not None and cfg.expected_mode_machine is None:
                    cfg.expected_mode_machine = 0
                report["configuration"] = asdict(cfg)
                robot = G1ArmSDK(cfg, transport=io)
                robot.connect()
                observation = robot.observation()
                start = np.array([observation[key] for key in ARM_KEYS])
                if args.phase == "joint":
                    target = start.copy()
                    target[ARM_KEYS.index(args.joint)] += args.delta_rad
                    robot._check_pose(target)
                if not cfg.read_only:
                    robot.activate()
                    report["motion_enabled"] = True
                started = time.monotonic()
                ramp = 0 if cfg.read_only else cfg.blend_s + 0.5
                total = ramp + args.duration_s + (0 if cfg.read_only else 1.0)
                print(f"Running {args.backend}/{args.phase}; duration about {total:.1f}s", flush=True)
                while time.monotonic() - started < total:
                    stamp = time.monotonic()
                    elapsed = stamp - started
                    target = start.copy()
                    if args.phase == "joint":
                        target = trajectory(
                            start,
                            ARM_KEYS.index(args.joint),
                            args.delta_rad,
                            (elapsed - ramp) / args.duration_s,
                        )
                    if not cfg.read_only:
                        robot.send(dict(zip(ARM_KEYS, target.tolist(), strict=True)))
                    with robot.lock:
                        obs = robot.observation()
                        measured = [obs[key] for key in ARM_KEYS]
                        report["samples"].append(
                            {
                                "elapsed_s": elapsed,
                                "target": target.tolist(),
                                "measured": measured,
                                "observation": obs,
                                "executed_target": robot.previous.tolist() if robot.writer else None,
                                "tick": robot.tick,
                                "mode": robot.mode,
                                "state_age_s": time.monotonic() - robot.state_at,
                                "weight": robot.weight,
                            }
                        )
                    time.sleep(max(0, cfg.period_s - (time.monotonic() - stamp)))
                # Finish command authority before potentially expensive report analysis.
                robot.close()
                errors = [max(abs(np.array(s["target"]) - s["measured"])) for s in report["samples"]]
                report["max_requested_tracking_error_rad"] = float(max(errors))
                if not cfg.read_only and max(errors) > cfg.max_tracking_error:
                    raise RuntimeError("Requested trajectory tracking threshold exceeded")
                if args.phase == "joint" and abs(args.delta_rad) > 0:
                    index = ARM_KEYS.index(args.joint)
                    peak = max(
                        np.sign(args.delta_rad) * (s["measured"][index] - start[index])
                        for s in report["samples"]
                    )
                    report["directed_measured_motion_rad"] = float(peak)
                    if peak < abs(args.delta_rad) * 0.5:
                        raise RuntimeError("Insufficient measured motion in requested direction")
                    end_error = float(np.max(np.abs(np.array(report["samples"][-1]["measured"]) - start)))
                    other_motion = max(
                        float(np.max(np.abs(np.delete(np.array(s["measured"]) - start, index))))
                        for s in report["samples"]
                    )
                    tolerance = min(cfg.max_tracking_error, max(0.005, abs(args.delta_rad) * 0.25))
                    report.update(
                        final_return_error_rad=end_error,
                        max_other_arm_joint_motion_rad=other_motion,
                        diagnostic_return_tolerance_rad=tolerance,
                    )
                    if max(end_error, other_motion) > tolerance:
                        raise RuntimeError(
                            "Return error or unselected-arm-joint movement exceeded diagnostic tolerance"
                        )
                report["status"] = "software_checks_passed"
        except KeyboardInterrupt:
            report["status"] = "cancelled"
        except Exception as exc:
            report["status"] = "failed"
            report["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            if robot is not None:
                try:
                    robot.close()
                except Exception as exc:
                    report["status"] = "failed"
                    report["errors"].append(f"Cleanup: {exc}")
                report["release_packet_sent"] = robot.release_sent
                report["physical_stop_verified"] = False
            out.seek(0)
            json.dump(report, out, indent=2)
            out.truncate()
    print(f"{report['status']}: {args.output}", flush=True)
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return 0 if report["status"] in ("software_checks_passed", "dry_run_passed") else 1


if __name__ == "__main__":

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
