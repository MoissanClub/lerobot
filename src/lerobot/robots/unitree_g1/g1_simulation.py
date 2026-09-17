# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Deterministic, fixed-base arm dynamics using the Cartesian layer's exact URDF.

No DDS participant, physical discovery, or background thread. Each action advances
one control period. Legs, waist and fingers are welded at URDF neutral.
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .g1_cartesian_control import G1ArmKinematics, G1CartesianConfig


class G1Simulation:
    def __init__(self, config):
        import mujoco

        self.mj, self.config = mujoco, config
        self.ik = G1ArmKinematics(G1CartesianConfig(config.embodiment, config.simulation_urdf))
        self.renderer = None
        self.closed = False
        self.indices = [j.value for j in self.ik.embodiment.arm_index]
        self.keys = [f"{j.name}.q" for j in self.ik.embodiment.arm_index]
        self.kp = np.asarray(config.kp)[self.indices]
        self.kd = np.asarray(config.kd)[self.indices]
        root = ET.parse(self.ik.urdf_path).getroot()
        efforts = {
            j.get("name"): float(j.find("limit").get("effort"))
            for j in root.findall("joint")
            if j.get("name") in self.ik.joint_names
        }
        self.torque_limits = np.array([efforts[n] for n in self.ik.joint_names])
        if not np.all(np.isfinite(self.torque_limits)) or np.any(self.torque_limits <= 0):
            raise ValueError("URDF must declare positive finite arm effort limits")
        for joint in root.findall("joint"):
            if joint.get("name") not in self.ik.joint_names:
                joint.set("type", "fixed")
        for link in root.findall("link"):
            for geom in list(link):
                if geom.tag == "collision" or (geom.tag == "visual" and config.simulation_mesh_dir is None):
                    link.remove(geom)
        extension = root.find("mujoco")
        if extension is None:
            extension = ET.SubElement(root, "mujoco")
        compiler = extension.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(extension, "compiler")
        compiler.set("discardvisual", "false")
        compiler.set("fusestatic", "false")
        if config.simulation_mesh_dir is not None:
            meshes = Path(config.simulation_mesh_dir).resolve(strict=True)
            compiler.set("meshdir", str(meshes))
            for mesh in root.iter("mesh"):
                mesh.set("filename", str(meshes / Path(mesh.get("filename")).name))
        with tempfile.TemporaryDirectory(prefix="g1-native-") as folder:
            path = Path(folder) / "robot.urdf"
            ET.ElementTree(root).write(path)
            self.model = mujoco.MjModel.from_xml_path(str(path))
        self.model.opt.gravity[:] = [0, 0, -9.81]
        self.substeps = max(1, int(np.ceil(config.control_dt / 0.002)))
        self.model.opt.timestep = config.control_dt / self.substeps
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        joints = [self.model.joint(n).id for n in self.ik.joint_names]
        self.qadr = self.model.jnt_qposadr[joints]
        self.vadr = self.model.jnt_dofadr[joints]
        # Explicit simulation regularization, not a calibrated hardware motor model.
        self.model.dof_armature[self.vadr] = 0.01
        self.model.dof_damping[self.vadr] = 0.05
        self.data = mujoco.MjData(self.model)
        self.last_torque = np.zeros(self.ik.size)
        self.reset()

    def _open(self):
        if self.closed:
            raise RuntimeError("Simulator is closed")

    def reset(self, body_positions=None):
        self._open()
        positions = np.asarray(
            self.config.default_positions if body_positions is None else body_positions, dtype=float
        )
        if positions.shape != (29,) or not np.all(np.isfinite(positions)):
            raise ValueError("Expected 29 finite body-slot positions")
        if np.any(np.delete(positions, self.indices) != 0):
            raise ValueError("Non-arm joints are fixed at zero")
        q = positions[self.indices]
        if np.any(q < self.ik.lower) or np.any(q > self.ik.upper):
            raise ValueError("Reset exceeds arm joint limits")
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = q
        self.target = q.copy()
        self.last_torque[:] = 0
        self.mj.mj_forward(self.model, self.data)

    def send_action(self, action):
        self._open()
        allowed = {f"{j.name}.q" for j in self.ik.embodiment.joint_index}
        if set(action) - allowed:
            raise ValueError("Unknown action keys")
        if any(not np.isscalar(v) or not np.isfinite(v) for v in action.values()):
            raise ValueError("Action values must be finite scalars")
        if any(v != 0 for k, v in action.items() if k not in self.keys):
            raise ValueError("Non-arm joints are fixed at zero")
        target = np.array([action.get(k, self.target[i]) for i, k in enumerate(self.keys)])
        if np.any(target < self.ik.lower) or np.any(target > self.ik.upper):
            raise ValueError("Action exceeds joint limits")
        self.target = target
        self.step()
        return dict(action)

    def step(self):
        self._open()
        for _ in range(self.substeps):
            q = self.data.qpos[self.qadr]
            tau = self.kp * (self.target - q) - self.kd * self.data.qvel[self.vadr]
            if self.config.gravity_compensation:
                tau += self.ik.gravity(q)
            self.last_torque = np.clip(tau, -self.torque_limits, self.torque_limits)
            self.data.qfrc_applied[self.vadr] = self.last_torque
            self.mj.mj_step(self.model, self.data)
            if not np.all(np.isfinite(self.data.qpos)) or np.any(self.data.warning.number):
                raise RuntimeError("Native MuJoCo physics warning or nonfinite state")

    def observation(self):
        self._open()
        obs = {
            f"{j.name}.{field}": 0.0 for j in self.ik.embodiment.joint_index for field in ("q", "dq", "tau")
        }
        for i, joint in enumerate(self.ik.embodiment.arm_index):
            obs[f"{joint.name}.q"] = float(self.data.qpos[self.qadr[i]])
            obs[f"{joint.name}.dq"] = float(self.data.qvel[self.vadr[i]])
            obs[f"{joint.name}.tau"] = float(self.last_torque[i])
        return obs

    def render(self, width=640, height=480, head_camera=True):
        self._open()
        if self.config.simulation_mesh_dir is None:
            raise ValueError("Rendering requires simulation_mesh_dir")
        if self.renderer is None or (self.renderer.width, self.renderer.height) != (width, height):
            if self.renderer is not None:
                self.renderer.close()
            self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
            self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
            self.renderer = self.mj.Renderer(self.model, width=width, height=height)
        self.mj.mj_forward(self.model, self.data)
        camera = self.mj.MjvCamera()
        camera.lookat[:] = [0, 0, -0.1]
        camera.distance, camera.azimuth, camera.elevation = 2.2, -135, -10
        self.renderer.update_scene(self.data, camera=camera)
        if head_camera:
            body = self.data.body("torso_link")
            rotation = body.xmat.reshape(3, 3)
            angle = np.deg2rad(35)
            for eye in self.renderer.scene.camera:
                eye.pos[:] = body.xpos + rotation @ [0.10, 0, 0.38]
                eye.forward[:] = rotation @ [np.cos(angle), 0, -np.sin(angle)]
                eye.up[:] = rotation @ [np.sin(angle), 0, np.cos(angle)]
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.closed = True
