#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Combined-stack XR control example; --headless runs synthetic dual-arm input."""

import argparse
import os
import time
from functools import partial
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--embodiment", choices=["g1_29", "g1_23"], default="g1_29")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=300, help="0 runs continuously")
    parser.add_argument("--video", action="store_true", help="Share input/video OpenXR session (live only)")
    parser.add_argument("--channel", type=Path, default=Path("/tmp/lerobot-xr-example.rgb"))
    args = parser.parse_args()
    if args.steps < 0 or (args.headless and args.steps == 0):
        parser.error("Headless requires positive steps")
    if args.headless and args.video:
        parser.error("Headless replay publishes camera frames but does not open an XR session")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot.cameras.frame_channel import FrameWriter
    from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
    from lerobot.robots.unitree_g1.g1_xr_control import G1XRControl
    from lerobot.teleoperators.xr_controllers import XRControllers, XRControllersConfig
    from lerobot.teleoperators.xr_controllers.camera_display import VideoConfig
    from lerobot.teleoperators.xr_controllers.video_session import VideoControllerSession

    robot = UnitreeG1(
        UnitreeG1Config(
            embodiment=args.embodiment,
            simulation_urdf=str(args.assets / f"{args.embodiment}.urdf"),
            simulation_mesh_dir=str(args.assets / "meshes"),
            gravity_compensation=True,
        )
    )
    writer, reader = None, None
    try:
        robot.connect()
        sim = robot._native
        q = np.zeros(sim.ik.size)
        q[[0, sim.ik.size // 2]] = -0.4
        q[[3, sim.ik.size // 2 + 3]] = 0.7
        robot.send_action(sim.ik.arm_action(q))
        for _ in range(500):
            robot.step_simulation()
        control = G1XRControl(sim.ik)
        writer = FrameWriter(args.channel, 320, 240)
        if not args.headless:
            options = {}
            if args.video:
                options["session_factory"] = partial(
                    VideoControllerSession,
                    video=VideoConfig(channel=str(args.channel), expected_source=args.embodiment),
                )
            reader = XRControllers(XRControllersConfig(), **options)
            reader.connect()
        initial = sim.data.qpos[sim.qadr].copy()
        largest_motion = 0.0
        step = 0
        while args.steps == 0 or step < args.steps:
            started = time.monotonic()
            if reader is None:
                sample = {"captured_at": started}
                for side in ("left", "right"):
                    sample.update(
                        {
                            f"{side}.tracked": True,
                            f"{side}.grip_pos": [0, 0, 0.025 * np.sin(step / 35)],
                            f"{side}.grip_quat": [0, 0, 0, 1],
                            f"{side}.squeeze": 1.0,
                            f"{side}.trigger": 0.0,
                        }
                    )
            else:
                sample = reader.get_action()
            robot.send_action(control.action(sample, sim.data.qpos[sim.qadr].copy()))
            for _ in range(4):
                robot.step_simulation()
            largest_motion = max(largest_motion, float(np.max(abs(sim.data.qpos[sim.qadr] - initial))))
            if step % 2 == 0:
                captured = time.monotonic_ns()
                writer.publish(
                    robot.render_simulation(320, 240),
                    {"captured_monotonic_ns": captured, "embodiment": args.embodiment},
                )
            step += 1
            if reader is not None:
                time.sleep(max(0, 0.02 - (time.monotonic() - started)))
        if args.headless and largest_motion < 0.01:
            raise RuntimeError("Synthetic XR input did not move the robot")
        print(
            f"{args.embodiment}: {step} control frames, {writer.sequence} camera frames, motion={largest_motion:.4f}rad"
        )
    finally:
        try:
            if reader is not None:
                reader.disconnect()
        finally:
            if writer is not None:
                writer.close()
            robot.disconnect()


if __name__ == "__main__":
    main()
