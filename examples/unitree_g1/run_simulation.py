#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Run local G1 arm dynamics with scripted or keyboard Cartesian targets."""

import argparse
import os
from pathlib import Path

import numpy as np

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embodiment", choices=["g1_29", "g1_23"], default="g1_29")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument(
        "--viewer-frames", type=int, default=0, help="Close after N frames; 0 runs continuously"
    )
    parser.add_argument("--no-gravity-compensation", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    robot = UnitreeG1(
        UnitreeG1Config(
            embodiment=args.embodiment,
            simulation_urdf=str(args.assets / f"{args.embodiment}.urdf"),
            simulation_mesh_dir=None if args.headless else str(args.assets / "meshes"),
            gravity_compensation=not args.no_gravity_compensation,
        )
    )
    try:
        robot.connect()
        sim = robot._native
        q = np.zeros(sim.ik.size)
        q[[0, sim.ik.size // 2]] = -0.5
        q[[3, sim.ik.size // 2 + 3]] = 0.8
        robot.send_action(sim.ik.arm_action(q))
        for _ in range(args.steps):
            robot.step_simulation()
        if args.headless:
            error = np.max(abs(sim.data.qpos[sim.qadr] - q))
            if not np.isfinite(error) or error > 0.2:
                raise RuntimeError(f"Tracking failed: {error}")
            print(f"{args.embodiment}: time={sim.data.time:.3f}s max_error={error:.5f}rad")
            return
        import tkinter as tk

        from PIL import Image, ImageTk

        root = tk.Tk()
        root.title(f"LeRobot {args.embodiment}: native MuJoCo")
        label = tk.Label(root)
        label.pack()
        targets = list(sim.ik.fk(sim.data.qpos[sim.qadr]))
        frames = 0

        def key(event):
            axes = {
                "Up": (2, 0.01),
                "Down": (2, -0.01),
                "Left": (1, 0.01),
                "Right": (1, -0.01),
                "w": (0, 0.01),
                "s": (0, -0.01),
            }
            if event.keysym in axes:
                axis, delta = axes[event.keysym]
                for target in targets:
                    target[axis, 3] += delta

        def tick():
            nonlocal frames
            result = sim.ik.solve(*targets, sim.data.qpos[sim.qadr])
            robot.send_action(sim.ik.arm_action(result.q))
            for _ in range(9):
                robot.step_simulation()
            photo = ImageTk.PhotoImage(Image.fromarray(robot.render_simulation(head_camera=False)))
            label.configure(image=photo)
            label.image = photo
            frames += 1
            if args.viewer_frames and frames >= args.viewer_frames:
                root.after(40, root.destroy)
            else:
                root.after(40, tick)

        root.bind("<KeyPress>", key)
        print("Arrows: height/lateral targets; W/S: forward/back. Close window to stop.")
        tick()
        root.mainloop()
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
