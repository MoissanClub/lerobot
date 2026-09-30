#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""G1-29 Unitree-mapped VR: isolated simulation, read-only shadow, gated hardware.

Terminal r + Enter starts/resumes, p + Enter pauses, q + Enter exits. No hardware
mode switching or motion on connection. Physical modes require a reviewed contract.
"""

import argparse
import json
import platform
import select
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from lerobot.robots.unitree_g1.g1_vr_control import (
    ARM_KEYS,
    G1VRKinematics,
    map_input,
    stop_buttons,
    synthetic_input,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=["simulation", "shadow", "arms", "walk", "combined"], default="simulation"
    )
    p.add_argument("--assets", type=Path, required=True, help="Prepared pinned Hub snapshot")
    p.add_argument("--input", choices=["live", "synthetic"], default="live")
    p.add_argument("--contract", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration-s", type=float, default=60)
    p.add_argument("--onscreen", action="store_true", help="Local MuJoCo viewer, simulation only")
    p.add_argument("--video", action="store_true")
    p.add_argument("--channel", type=Path, default=Path("/tmp/lerobot-physical-camera.rgb"))
    p.add_argument("--cloudxr-config", type=Path)
    p.add_argument("--accept-cloudxr-eula", action="store_true")
    p.add_argument("--enable-motion", action="store_true")
    p.add_argument("--enable-locomotion", action="store_true")
    p.add_argument("--synthetic-velocity", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    args = p.parse_args(argv)
    if not np.isfinite(args.duration_s) or not 0 < args.duration_s <= 3600:
        p.error("duration-s must be in (0, 3600]")
    physical = args.mode in ("arms", "walk", "combined")
    if physical and (not args.enable_motion or args.input != "live"):
        p.error("Physical control needs live XR, --enable-motion and reviewed contract")
    if args.mode != "simulation" and args.contract is None:
        p.error("Physical feedback/control requires --contract")
    if (args.mode in ("walk", "combined")) != args.enable_locomotion:
        p.error("--enable-locomotion is required only for walk/combined")
    if not physical and args.enable_motion:
        p.error("Read-only/simulation modes must not enable physical motion")
    if args.onscreen and args.mode != "simulation":
        p.error("--onscreen is only for simulation")
    if args.input == "synthetic" and (args.mode != "simulation" or args.video or args.cloudxr_config):
        p.error("Synthetic input is isolated simulation only, without XR video/runtime")
    if not np.isfinite(args.synthetic_velocity).all() or max(abs(v) for v in args.synthetic_velocity) > 0.3:
        p.error("Synthetic velocity components must be finite and <= 0.3")
    if args.cloudxr_config and not args.accept_cloudxr_eula:
        p.error("Explicit CloudXR EULA acceptance is required")
    return args


def record(output, **row):
    output.write(json.dumps(row, allow_nan=False) + "\n")
    output.flush()


def run(args, output):
    from validate_arm_sdk import digest, validate_contract

    from lerobot.robots.unitree_g1.g1_arm_sdk import G1ArmSDK
    from lerobot.robots.unitree_g1.g1_base_motion import G1BaseMotion

    physical = args.mode in ("arms", "walk", "combined")
    cfg, doc = None, None
    urdf = args.assets / "assets/g1_body29_hand14.urdf"
    if args.contract:
        doc = json.loads(args.contract.read_text())
        cfg = validate_contract(doc, urdf, physical)
        if physical and doc.get("vr_pose_mapping_reviewed") is not True:
            raise ValueError("Review VR target frames and starting pose before physical activation")
        if args.enable_locomotion:
            for flag in ("locomotion_reviewed", "damping_reviewed", "combined_ownership_reviewed"):
                if doc.get(flag) is not True:
                    raise ValueError(f"Required locomotion review: {flag}")
    ik = G1VRKinematics(args.assets)
    reader = sim = robot = base = writer = None
    damp_requested = False
    with ExitStack() as cleanup:
        if args.cloudxr_config:
            import socket

            from isaacteleop.cloudxr import CloudXRLauncher

            for port in (48322, 49100):
                with socket.socket() as probe:
                    probe.settimeout(0.3)
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        raise RuntimeError(f"CloudXR port {port} is in use; stop the other session first")
            cleanup.enter_context(CloudXRLauncher(env_config=str(args.cloudxr_config), accept_eula=True))
            print(
                "CloudXR ready: connect headset at https://nvidia.github.io/IsaacTeleop/client using this host IP",
                flush=True,
            )
        if args.input == "live":
            from lerobot.teleoperators.xr_controllers import XRControllers, XRControllersConfig
            from lerobot.teleoperators.xr_controllers.camera_display import VideoConfig
            from lerobot.teleoperators.xr_controllers.video_session import VideoControllerSession

            options = {}
            if args.video:
                source = "g1-29-simulation" if args.mode == "simulation" else "g1-29-physical-camera"
                options["session_factory"] = partial(
                    VideoControllerSession,
                    video=VideoConfig(channel=str(args.channel), expected_source=source),
                )
            reader = XRControllers(
                XRControllersConfig(full_input=True, base_T_anchor=np.eye(4).tolist()), **options
            )
            reader.connect()
            cleanup.callback(reader.disconnect)
        if args.mode == "simulation":
            from lerobot.robots.unitree_g1.g1_vr_simulation import G1VRSimulation

            sim = G1VRSimulation(args.assets, onscreen=args.onscreen, video=args.video)
            cleanup.callback(sim.close)
            q = np.zeros(14)
            for _ in range(100):
                sim.step(q, ik.gravity(q))
            if args.video:
                from lerobot.cameras.frame_channel import FrameWriter

                writer = FrameWriter(args.channel, 640, 480)
                cleanup.callback(writer.close)
            observation = sim.observation
        else:
            cfg = replace(cfg, read_only=args.mode in ("shadow", "walk"))
            robot = G1ArmSDK(cfg)
            cleanup.callback(robot.close)
            robot.connect()
            observation = robot.observation
        if args.enable_locomotion:
            limits = doc.get("base_limits", {})
            if set(limits) != {"max_speed", "max_yaw", "timeout"}:
                raise ValueError("Explicit reviewed base_limits required")

            def base_feedback():
                feedback = observation()
                if robot.mode != (cfg.expected_mode_machine, 0):
                    raise RuntimeError("Base motion requires reviewed matching feedback mode")
                return feedback

            base = G1BaseMotion(base_feedback, domain_id=cfg.domain_id, **limits)
            cleanup.callback(lambda: base.close(damp=damp_requested))
        initial_observation = observation()
        measured = np.array([initial_observation[key] for key in ARM_KEYS])
        initial_wrists = ik.fk(measured)
        held = measured.copy()
        started = args.input == "synthetic"
        stamp_last = -np.inf
        deadline = time.monotonic() + args.duration_s
        record(
            output,
            event="ready",
            physical_commands=physical,
            model_sha256=digest(urdf),
            contract_sha256=digest(args.contract) if args.contract else None,
        )
        print(
            f"{args.mode}: r + Enter starts, p + Enter pauses, q + Enter exits. Squeeze is NOT an arm clutch.",
            flush=True,
        )
        step = 0
        while time.monotonic() < deadline:
            loop = time.monotonic()
            obs = observation()
            measured = np.array([obs[key] for key in ARM_KEYS])
            if args.input == "live" and select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip().lower()
                if key == "q":
                    break
                if key == "r":
                    started = True
                if key == "p":
                    started = False
                    held = measured.copy()
            sample = (
                reader.get_action()
                if reader
                else synthetic_input(initial_wrists, step, args.synthetic_velocity)
            )
            quit_requested, damp_requested = stop_buttons(sample)
            if quit_requested or damp_requested:
                record(
                    output,
                    event="operator_stop",
                    damp_requested=damp_requested,
                    damping_rpc_enabled=base is not None and base.thread is not None,
                )
                if sim and damp_requested:
                    sim.step(held, np.zeros(14), damp=True)
                break
            invalid = None
            try:
                intent = map_input(sample)
                if sample["captured_at"] < stamp_last:
                    raise ValueError("XR time regressed")
                stamp_last = sample["captured_at"]
            except (ValueError, KeyError, TypeError) as exc:
                invalid = str(exc)
                intent = None
                started = False
                held = measured.copy()
            if intent and (intent["quit"] or intent["damp"]):
                damp_requested = intent["damp"]
                record(
                    output,
                    event="operator_stop",
                    damp_requested=damp_requested,
                    damping_rpc_enabled=base is not None,
                )
                if sim and damp_requested:
                    sim.step(held, np.zeros(14), damp=True)
                break
            velocity = np.zeros(3)
            target = held.copy()
            if started and intent:
                if args.mode != "walk":
                    try:
                        target, _ = ik.solve(intent["wrists"], measured)
                    except RuntimeError as exc:
                        started = False
                        target = measured.copy()
                        record(output, event="ik_pause", reason=str(exc))
                        print("IK failed: holding measured pose; review and press r to resume", flush=True)
                velocity = intent["velocity"]
                if not started:
                    velocity[:] = 0
                held = target.copy()
                if started and robot and physical and not robot.active and args.mode != "walk":
                    if max(abs(target - measured)) > cfg.max_displacement:
                        raise ValueError("Initial VR target exceeds reviewed displacement envelope")
                    robot.activate()
                if started and base and base.thread is None:
                    base.activate()
            tau = ik.gravity(target)
            if sim:
                sim.step(target, tau, velocity)
                if writer and step % 2 == 0:
                    captured = time.monotonic_ns()
                    writer.publish(
                        sim.render(), {"captured_monotonic_ns": captured, "embodiment": "g1-29-simulation"}
                    )
            if robot and robot.active:
                robot.send(dict(zip(ARM_KEYS, target.tolist(), strict=True)))
            if base and base.thread is not None:
                base.send(velocity)
            record(
                output,
                event="frame",
                started=started,
                invalid=invalid,
                measured=measured.tolist(),
                target=target.tolist(),
                velocity=velocity.tolist(),
                proposed_hand_slots=intent["hands"] if intent else None,
                base_position=sim.data.qpos[:3].tolist() if sim else None,
                motor_publication=bool(robot and robot.active),
                base_rpc=bool(base and base.thread),
            )
            if step % 50 == 0:
                print(
                    f"started={started} base={velocity.round(3).tolist()} arm_delta={max(abs(target - measured)):.4f} invalid={invalid}",
                    flush=True,
                )
            step += 1
            time.sleep(max(0, 0.02 - (time.monotonic() - loop)))


def main(argv=None):
    args = parse_args(argv)
    import lerobot

    if Path(lerobot.__file__).resolve().parents[2] != Path(__file__).resolve().parents[2]:
        raise RuntimeError("Wrong LeRobot import; set PYTHONPATH to this checkout's src")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with args.output.open("x") as output:
        record(
            output,
            event="start",
            source_revision=subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
            ).strip(),
            python=platform.python_version(),
            mode=args.mode,
            hardware_acceptance="pending_operator_review",
            parameters={
                key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            },
        )
        try:
            run(args, output)
        except BaseException as exc:
            record(output, event="failed_or_cancelled", error=str(exc) or type(exc).__name__)
            raise
        record(output, event="completed", hardware_acceptance="pending_operator_review")


if __name__ == "__main__":
    main()
