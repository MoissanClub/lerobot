# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""In-process adapter to the pinned upstream Hub environment and GR00T policy.

Uses DefaultEnv, not BaseSimulator: no DDS factory, subscriber, publisher or RPC.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .g1_utils import G1_29_JointIndex
from .g1_vr_control import POLICY_REVISION


class G1VRSimulation:
    def __init__(self, assets, *, onscreen=False, video=False, stationary_fixture=False):
        import mujoco
        import yaml

        from .config_unitree_g1 import UnitreeG1Config
        from .controllers.gr00t_locomotion import GrootLocomotionController

        assets = Path(assets).resolve()
        if (
            "sim" in sys.modules
            and Path(sys.modules["sim"].__file__).absolute() != (assets / "sim/__init__.py").absolute()
        ):
            raise RuntimeError("Another Hub simulator is imported; use a fresh process")
        sys.path.insert(0, str(assets))
        try:
            environment_class = importlib.import_module("sim.base_sim").DefaultEnv
            select = importlib.import_module("sim.model_config").select_end_effector
        finally:
            sys.path.remove(str(assets))
        config = select(yaml.safe_load((assets / "config.yaml").read_text()), "dex3")
        config.update(ENABLE_ELASTIC_BAND=False, FREE_BASE=False, SIMULATE_DT=0.002)
        self.policy = GrootLocomotionController(policy_revision=POLICY_REVISION)
        cameras = {"head_camera": {"width": 640, "height": 480}} if video else {}
        self.env = environment_class(config, onscreen=False, offscreen=video, camera_configs=cameras)
        self.stationary_fixture = stationary_fixture
        spec = mujoco.MjSpec.from_file(str(assets / config["ROBOT_SCENE"]))
        # Match upstream IK's locked Dex3 inertias without enabling hand actuation.
        for joint in spec.joints:
            if "hand" in joint.name:
                spec.add_equality(
                    name=f"locked_{joint.name}",
                    type=mujoco.mjtEq.mjEQ_JOINT,
                    name1=joint.name,
                    data=[0.0] * 11,
                )
        if stationary_fixture:
            # Model-only support fixture, not simulated stock balance or firmware.
            spec.add_equality(
                name="pelvis_support",
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="pelvis",
                name2="world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
            )
            for i, address in enumerate(self.env.body_qpos_index[:15]):
                joint_id = int(np.flatnonzero(self.env.mj_model.jnt_qposadr == address)[0])
                name = self.env.mj_model.joint(joint_id).name
                spec.add_equality(
                    name=f"support_{name}",
                    type=mujoco.mjtEq.mjEQ_JOINT,
                    name1=name,
                    data=[float(self.policy.default_angles[i])] + [0.0] * 10,
                )
        self.env.mj_model = spec.compile()
        self.env.mj_model.opt.timestep = 0.002
        self.env.mj_data = mujoco.MjData(self.env.mj_model)
        if onscreen:
            self.env.viewer = mujoco.viewer.launch_passive(self.env.mj_model, self.env.mj_data)
            self.env.viewer.cam.azimuth = 120
            self.env.viewer.cam.elevation = -25
            self.env.viewer.cam.distance = 2.5
            self.env.viewer.cam.lookat[:] = [0, 0, 0.8]
        self.mj = mujoco
        self.data, self.model = self.env.mj_data, self.env.mj_model
        if self.model.nq != 50 or self.model.nv != 49:
            raise ValueError("Expected floating-base G1-29 with locked reference hand joints")
        cfg = UnitreeG1Config()
        motors = [
            SimpleNamespace(q=float(q), dq=0.0, tau=0.0, kp=kp, kd=kd)
            for q, kp, kd in zip(self.policy.default_angles, cfg.kp, cfg.kd, strict=True)
        ]
        self.command = SimpleNamespace(motor_cmd=motors)
        self.env.set_unitree_bridge(
            SimpleNamespace(
                low_cmd=self.command,
                num_body_motor=29,
                num_hand_motor=7,
                left_hand_cmd=SimpleNamespace(
                    motor_cmd=[SimpleNamespace(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0) for _ in range(7)]
                ),
                right_hand_cmd=SimpleNamespace(
                    motor_cmd=[SimpleNamespace(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0) for _ in range(7)]
                ),
                use_sensor=False,
                joystick=None,
                PublishLowState=lambda obs: None,
            )
        )
        self.data.qpos[2] = 0.79
        self.data.qpos[self.env.body_qpos_index] = self.policy.default_angles
        mujoco.mj_forward(self.model, self.data)
        self.tick = 0

    def state(self):
        from scipy.spatial.transform import Rotation

        obs = self.env.prepare_obs()
        quat = self.data.qpos[3:7].copy()
        return SimpleNamespace(
            tick=self.tick,
            mode_machine=0,
            mode_pr=0,
            motor_state=[
                SimpleNamespace(q=float(q), dq=float(dq), tau_est=float(tau))
                for q, dq, tau in zip(obs["body_q"], obs["body_dq"], obs["body_tau_est"], strict=True)
            ],
            imu_state=SimpleNamespace(
                quaternion=quat,
                gyroscope=self.data.qvel[3:6].copy(),
                rpy=Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz"),
            ),
        )

    def observation(self):
        state = self.state()
        return {
            f"{j.name}.{field}": getattr(state.motor_state[j.value], source)
            for j in G1_29_JointIndex
            for field, source in (("q", "q"), ("dq", "dq"), ("tau", "tau_est"))
        }

    def step(self, arm_q=None, arm_tau=None, velocity=(0.0, 0.0, 0.0), *, damp=False):
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all() or np.max(abs(velocity)) > 0.3:
            raise ValueError("Invalid simulation velocity")
        action = {"remote.ly": velocity[0], "remote.lx": -velocity[1], "remote.rx": -velocity[2]}
        lower = (
            {f"{j.name}.q": self.policy.default_angles[j.value] for j in G1_29_JointIndex}
            if self.stationary_fixture
            else self.policy.run_step(action, self.state())
        )
        for j in G1_29_JointIndex:
            if j.value < 15:
                self.command.motor_cmd[j.value].q = lower[f"{j.name}.q"]
        if arm_q is not None:
            for i, cmd in enumerate(self.command.motor_cmd[15:]):
                cmd.q = float(arm_q[i])
                cmd.tau = 0.0 if arm_tau is None else float(arm_tau[i])
        if damp:
            for cmd in self.command.motor_cmd:
                cmd.kp, cmd.tau = 0.0, 0.0
        for _ in range(10):
            self.env.sim_step()
            self.tick += 2
        if not np.isfinite(self.data.qpos).all() or np.any(self.data.warning.number):
            raise RuntimeError("MuJoCo numerical failure")
        if self.data.qpos[2] < 0.4:
            raise RuntimeError("Simulation fell; stop and review")
        self.env.update_viewer()

    def render(self):
        return self.env.update_render_caches()["head_camera_image"].copy()

    def close(self):
        if self.env.viewer is not None:
            self.env.viewer.close()
        for renderer in self.env.renderers.values():
            renderer.close()
        self.env.renderers.clear()
