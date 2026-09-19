#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Side-by-side native-URDF kinematic playback, not a motor dynamics simulation."""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np

from lerobot.robots.unitree_g1.g1_cartesian_control import G1ArmKinematics, G1CartesianConfig


class Panel:
    def __init__(self, assets, embodiment, camera_azimuth):
        import mujoco

        self.mj = mujoco
        self.ik = G1ArmKinematics(G1CartesianConfig(embodiment, assets / f"{embodiment}.urdf"))
        self.temp = tempfile.TemporaryDirectory(prefix=f"{embodiment}_kinematic_")
        root = ET.parse(self.ik.urdf_path).getroot()
        compiler = root.find("mujoco/compiler")
        if compiler is None:
            extension = ET.SubElement(root, "mujoco")
            compiler = ET.SubElement(extension, "compiler")
        compiler.set("meshdir", str(assets))
        compiler.set("discardvisual", "false")
        for link in root.findall("link"):
            for collision in link.findall("collision"):
                link.remove(collision)
        urdf_path = Path(self.temp.name) / "robot.urdf"
        ET.ElementTree(root).write(urdf_path)
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        scene = Path(self.temp.name) / "scene.xml"
        mujoco.mj_saveLastXML(str(scene), model)
        tree = ET.parse(scene)
        world = tree.getroot().find("worldbody")
        ET.SubElement(
            world, "light", pos="1 -1 3", dir="-0.2 0.2 -1", diffuse="0.8 0.8 0.8", ambient="0.4 0.4 0.4"
        )
        ET.SubElement(
            world,
            "geom",
            name="floor",
            type="plane",
            size="4 4 0.1",
            pos="0 0 -0.793",
            rgba="0.3 0.33 0.35 1",
        )
        tree.write(scene)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.addresses = [self.model.joint(n).qposadr[0] for n in self.ik.joint_names]
        self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.03, 0, -0.10]
        self.camera.distance = 2.2
        self.camera.azimuth = camera_azimuth
        self.camera.elevation = -10
        self.q = np.zeros(self.ik.size)
        half = self.ik.size // 2
        self.q[0], self.q[half] = -0.5, -0.5
        self.q[1], self.q[half + 1] = 0.2, -0.2
        self.q[3], self.q[half + 3] = 0.9, 0.9
        self.home = self.ik.fk(self.q)
        self.rows = deque(maxlen=2400)
        self.previous_image = None
        self.changed_frames = 0

    def frame(self, phase, wave):
        from PIL import Image, ImageDraw

        targets = [pose.copy() for pose in self.home]
        for target in targets:
            if phase < 3:
                target[phase, 3] += 0.06 * wave
            else:
                angle = 0.4 * wave
                rotation = np.array(
                    [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
                )
                target[:3, :3] = target[:3, :3] @ rotation
        previous_q = self.q.copy()
        result = self.ik.solve(*targets, self.q)
        self.q = result.q
        if (
            not np.all(np.isfinite(self.q))
            or np.any(self.q < self.ik.lower)
            or np.any(self.q > self.ik.upper)
            or np.max(np.abs(self.q - previous_q)) > self.ik.config.max_joint_step_rad + 1e-10
        ):
            raise RuntimeError("IK command violates finite joint/step bounds")
        self.data.qpos[self.addresses] = self.q
        self.mj.mj_forward(self.model, self.data)
        actual_poses = self.ik.fk(self.q)
        # mj_saveLastXML rounds transforms; allow 10 micrometres / 1e-5 rotation entries.
        for side, pose in zip(("left", "right"), actual_poses, strict=True):
            wrist = "yaw" if self.ik.config.embodiment == "g1_29" else "roll"
            joint_id = self.model.joint(f"{side}_wrist_{wrist}_joint").id
            body = self.data.body(int(self.model.jnt_bodyid[joint_id]))
            rotation = body.xmat.reshape(3, 3)
            if not np.allclose(
                body.xpos + rotation @ self.ik.ee_offset, pose[:3, 3], atol=1e-5, rtol=0
            ) or not np.allclose(rotation, pose[:3, :3], atol=1e-5, rtol=0):
                raise RuntimeError("Rendered URDF and Pinocchio FK disagree")
        self.renderer.update_scene(self.data, camera=self.camera)
        rgb = self.renderer.render().copy()
        # Check robot-region pixels, excluding the text added below.
        region = rgb[80:430, 170:470]
        if np.std(region) < 5:
            raise RuntimeError("Blank or uniform robot panel")
        if (
            self.previous_image is not None
            and np.mean(np.abs(region.astype(float) - self.previous_image)) > 0.02
        ):
            self.changed_frames += 1
        self.previous_image = region.astype(float)
        self.rows.append(
            {
                "phase": ["x", "y", "z", "roll"][phase],
                "wave": wave,
                "q": self.q.tolist(),
                "position_error_m": result.position_error_m,
                "orientation_error_rad": result.orientation_error_rad,
                "status": result.status,
                "actual_positions": [pose[:3, 3].tolist() for pose in actual_poses],
            }
        )
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 55), fill=(18, 22, 25))
        label = ["forward / back", "left / right", "up / down", "hand roll"][phase]
        draw.text((12, 8), f"{self.ik.config.embodiment} native URDF | {label}", fill="white")
        draw.text(
            (12, 30),
            f"{result.status} | error {max(result.position_error_m) * 1000:.1f} mm / {np.rad2deg(max(result.orientation_error_rad)):.1f} deg",
            fill="white",
        )
        return image

    def close(self):
        self.renderer.close()
        self.temp.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--samples-per-phase", type=int, default=60)
    parser.add_argument(
        "--cycles", type=int, default=0, help="0 loops in the viewer; headless defaults to one cycle"
    )
    parser.add_argument("--camera-azimuth", type=float, default=-135)
    parser.add_argument("--report", type=Path, default=Path("/tmp/g1-cartesian-report.json"))
    parser.add_argument("--frames", type=Path)
    args = parser.parse_args()
    if args.samples_per_phase < 8 or args.cycles < 0 or not np.isfinite(args.camera_azimuth):
        parser.error("samples-per-phase must be >=8, cycles >=0, and camera angle finite")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image

    panels = []
    window = None
    try:
        for name in ("g1_29", "g1_23"):
            panels.append(Panel(args.assets.resolve(), name, args.camera_azimuth))
        if not args.headless:
            import tkinter as tk

            from PIL import ImageTk

            window = tk.Tk()
            window.title("G1 Cartesian control: native G1-29 and G1-23")
            label = tk.Label(window)
            label.pack()
        count = 0
        cycles = (args.cycles or 1) if args.headless else args.cycles
        if args.frames:
            args.frames.mkdir(parents=True, exist_ok=True)
        while cycles == 0 or count < cycles * 4 * args.samples_per_phase:
            started = time.monotonic()
            phase = (count // args.samples_per_phase) % 4
            wave = float(np.sin(2 * np.pi * (count % args.samples_per_phase) / args.samples_per_phase))
            combined = Image.new("RGB", (1280, 480))
            for index, panel in enumerate(panels):
                combined.paste(panel.frame(phase, wave), (640 * index, 0))
            if args.frames and count % args.samples_per_phase in (
                0,
                args.samples_per_phase // 4,
                3 * args.samples_per_phase // 4,
            ):
                combined.save(args.frames / f"frame_{count:04d}.png")
            if window is not None:
                photo = ImageTk.PhotoImage(combined)
                label.configure(image=photo)
                label.image = photo
                window.update()
                time.sleep(max(0, 6 / args.samples_per_phase - (time.monotonic() - started)))
            count += 1
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if window is None or type(exc).__name__ != "TclError":
            raise
    finally:
        import lerobot.robots.unitree_g1.g1_cartesian_control as source

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        report = {
            "kind": "kinematic_playback_not_motor_dynamics",
            "commit": commit,
            "source": source.__file__,
            "source_sha256": hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
            "retained_frames_per_model": 2400,
            "models": {},
        }
        for panel in panels:
            report["models"][panel.ik.config.embodiment] = {
                "urdf_sha256": hashlib.sha256(panel.ik.urdf_path.read_bytes()).hexdigest(),
                "changed_frames": panel.changed_frames,
                "cases": list(panel.rows),
            }
            panel.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if window is not None:
            with contextlib.suppress(Exception):
                window.destroy()
    if args.headless and any(panel.changed_frames < 4 for panel in panels):
        raise RuntimeError("Insufficient motion in rendered panels")
    if args.headless and any(
        row["status"] == "optimizer_failed_hold" for panel in panels for row in panel.rows
    ):
        raise RuntimeError("Optimizer failure during verification sweep")
    for panel in panels:
        worst = max(max(row["position_error_m"]) for row in panel.rows)
        print(
            f"{panel.ik.config.embodiment}: frames={len(panel.rows)}, changed={panel.changed_frames}, max position error={worst:.4f} m"
        )
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
