# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Dual controller reader adapted from the Isaac Teleop SO-101 example.

CloudXR is externally owned unless explicitly configured for this session.
SDK imports are delayed until connect; replay and configuration need no OpenXR.
"""

import json
import select
import socket
import sys
import time
from contextlib import ExitStack

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_xr_controllers import XRControllersConfig


def controller_pose(action, side):
    """Return a validated base-frame pose or None. Never normalize a zero quaternion."""
    try:
        if not action[f"{side}.tracked"]:
            return None
        pos = np.asarray(action[f"{side}.grip_pos"], dtype=float)
        quat = np.asarray(action[f"{side}.grip_quat"], dtype=float)
        buttons = np.asarray([action[f"{side}.squeeze"], action[f"{side}.trigger"]], dtype=float)
        if (
            pos.shape != (3,)
            or quat.shape != (4,)
            or buttons.shape != (2,)
            or not np.all(np.isfinite(np.r_[pos, quat, buttons]))
            or np.linalg.norm(quat) < 1e-8
            or np.any(buttons < 0)
            or np.any(buttons > 1)
        ):
            return None
        from lerobot.utils.rotation import Rotation

        pose = np.eye(4)
        pose[:3, 3] = pos
        pose[:3, :3] = Rotation.from_quat(quat).as_matrix()
        return pose
    except (KeyError, TypeError, ValueError):
        return None


class IsaacControllerSession:
    """Small adapter around the optional NVIDIA SDK; also usable with shared XR handles."""

    def __init__(self, config, *, oxr_handles=None):
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HeadSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner, TensorGroup, ValueInput
        from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

        source = ControllersSource(name="controllers")
        transform = ValueInput("base_T_anchor", TransformMatrix())
        transformed = source.transformed(transform.output("value"))
        outputs = {side: transformed.output(f"controller_{side}") for side in ("left", "right")}
        if config.full_input:
            head = HeadSource(name="head").transformed(transform.output("value"))
            outputs["head"] = head.output("head")
        self.pipeline = OutputCombiner(outputs)
        value = TensorGroup(TransformMatrix())
        value[0] = np.asarray(config.base_T_anchor, dtype=np.float32)
        self.inputs = {"base_T_anchor": {"value": value}}
        self.config, self.oxr_handles = config, oxr_handles
        self.session = None
        self.entered = False

    def connect(self):
        from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig

        if self.entered:
            raise RuntimeError("Already connected")
        self.session = TeleopSession(
            TeleopSessionConfig(
                app_name=self.config.app_name, pipeline=self.pipeline, oxr_handles=self.oxr_handles
            )
        )
        try:
            self.session.__enter__()
        except BaseException as exc:
            # Isaac Teleop's __exit__ returns early before _setup_complete. Drain
            # its entered contexts when DeviceIO/plugin setup fails after OpenXR.
            # Keep this SDK-specific workaround here, not in the generic device.
            stack = getattr(self.session, "_exit_stack", None)
            try:
                if stack is not None:
                    stack.close()
                else:
                    self.session.__exit__(type(exc), exc, exc.__traceback__)
            except Exception as cleanup:
                exc.add_note(f"XR startup cleanup also failed: {cleanup}")
            self.session = None
            raise
        self.entered = True

    def read(self):
        from isaacteleop.retargeting_engine.interface import ExecutionEvents, ExecutionState
        from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex as Index

        result = self.session.step(
            execution_events=ExecutionEvents(execution_state=ExecutionState.RUNNING, reset=False),
            external_inputs=self.inputs,
        )
        info = self.session.last_step_info
        if info is not None and info.worker_exception is not None:
            raise RuntimeError("XR worker failed") from info.worker_exception
        fresh = info is None or not info.frame_deadline_miss
        action = {"captured_at": time.monotonic()}
        for side in ("left", "right"):
            values = {
                "grip_pos": np.zeros(3),
                "grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "squeeze": 0.0,
                "trigger": 0.0,
                "tracked": False,
            }
            if self.config.full_input:
                values.update(stick_x=0.0, stick_y=0.0, stick_click=False, primary=False, secondary=False)
            controller = result[side]
            if fresh and controller is not None and not getattr(controller, "is_none", False):
                try:
                    candidate = {
                        "grip_pos": np.asarray(controller[Index.GRIP_POSITION]),
                        "grip_quat": np.asarray(controller[Index.GRIP_ORIENTATION]),
                        "squeeze": float(controller[Index.SQUEEZE_VALUE]),
                        "trigger": float(controller[Index.TRIGGER_VALUE]),
                        "tracked": bool(controller[Index.GRIP_IS_VALID]),
                    }
                    if self.config.full_input:
                        candidate.update(
                            stick_x=float(controller[Index.THUMBSTICK_X]),
                            stick_y=float(controller[Index.THUMBSTICK_Y]),
                            stick_click=bool(controller[Index.THUMBSTICK_CLICK]),
                            primary=bool(controller[Index.PRIMARY_CLICK]),
                            secondary=bool(controller[Index.SECONDARY_CLICK]),
                        )
                        # Stop buttons do not require a valid grip pose.
                        for key in ("stick_click", "primary", "secondary"):
                            values[key] = candidate[key]
                    if controller_pose({f"{side}.{k}": v for k, v in candidate.items()}, side) is not None:
                        values = candidate
                except (KeyError, IndexError, TypeError, ValueError):
                    pass
            action.update({f"{side}.{key}": val for key, val in values.items()})
        if self.config.full_input:
            from isaacteleop.retargeting_engine.tensor_types.indices import HeadPoseIndex

            action.update(
                {"head.tracked": False, "head.pos": np.zeros(3), "head.quat": np.array([0.0, 0.0, 0.0, 1.0])}
            )
            head = result["head"]
            if fresh and head is not None and not getattr(head, "is_none", False):
                action.update(
                    {
                        "head.tracked": bool(head[HeadPoseIndex.IS_VALID]),
                        "head.pos": np.asarray(head[HeadPoseIndex.POSITION]),
                        "head.quat": np.asarray(head[HeadPoseIndex.ORIENTATION]),
                    }
                )
        return action

    def close(self):
        if self.entered:
            self.entered = False
            session, self.session = self.session, None
            session.__exit__(None, None, None)


class ReplaySession:
    """Explicit offline input fixture; never advertises live device provenance."""

    def __init__(self, config):
        self.path = config.replay_path
        self.stream = None

    def connect(self):
        self.stream = open(self.path)  # noqa: SIM115 - owned until disconnect

    def read(self):
        line = self.stream.readline()
        action = json.loads(line) if line else {"control.quit": True}
        action["captured_at"] = time.monotonic()
        return action

    def close(self):
        if self.stream:
            self.stream.close()
            self.stream = None


class XRControllers(Teleoperator):
    config_class = XRControllersConfig
    name = "xr_controllers"

    def __init__(self, config, *, session_factory=IsaacControllerSession):
        self._backend = None
        self._cleanup = None
        super().__init__(config)
        self.config, self._factory = config, session_factory

    @property
    def action_features(self):
        features = {"captured_at": float, "input.live": bool}
        features.update({f"control.{key}": bool for key in ("start", "pause", "quit")})
        for side in ("left", "right"):
            features.update(
                {
                    f"{side}.{key}": kind
                    for key, kind in {
                        "grip_pos": {"dtype": "float32", "shape": (3,), "names": None},
                        "grip_quat": {"dtype": "float32", "shape": (4,), "names": None},
                        "squeeze": float,
                        "trigger": float,
                        "tracked": bool,
                    }.items()
                }
            )
        if self.config.full_input:
            features.update(
                {
                    "head.tracked": bool,
                    "head.pos": {"dtype": "float32", "shape": (3,), "names": None},
                    "head.quat": {"dtype": "float32", "shape": (4,), "names": None},
                }
            )
            for side in ("left", "right"):
                features.update(
                    {
                        f"{side}.{key}": kind
                        for key, kind in (
                            ("stick_x", float),
                            ("stick_y", float),
                            ("stick_click", bool),
                            ("primary", bool),
                            ("secondary", bool),
                        )
                    }
                )
        return features

    @property
    def feedback_features(self):
        return {}

    @property
    def is_connected(self):
        return self._backend is not None

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def connect(self, calibrate=True):
        if self.is_connected:
            raise RuntimeError("Already connected")
        cleanup = ExitStack()
        try:
            if self.config.cloudxr_config:
                from isaacteleop.cloudxr import CloudXRLauncher

                for port in (48322, 49100):
                    with socket.socket() as probe:
                        probe.settimeout(0.3)
                        if probe.connect_ex(("127.0.0.1", port)) == 0:
                            raise RuntimeError(f"CloudXR port {port} already in use")
                cleanup.enter_context(
                    CloudXRLauncher(
                        env_config=self.config.cloudxr_config, accept_eula=self.config.accept_cloudxr_eula
                    )
                )
            self._connect_backend(cleanup)
        except BaseException:
            cleanup.close()
            raise
        self._cleanup = cleanup

    def _connect_backend(self, cleanup):
        if self.config.replay_path:
            backend = ReplaySession(self.config)
        elif self.config.video_channel:
            from .camera_display import VideoConfig
            from .video_session import VideoControllerSession

            backend = VideoControllerSession(
                self.config,
                VideoConfig(channel=self.config.video_channel, expected_source=self.config.video_source),
            )
        else:
            backend = self._factory(self.config)
        cleanup.callback(backend.close)
        backend.connect()
        self._backend = backend

    def get_action(self):
        if not self.is_connected:
            raise RuntimeError("Not connected")
        action = dict(self._backend.read())
        action["input.live"] = not bool(self.config.replay_path)
        for key in ("start", "pause", "quit"):
            action.setdefault(f"control.{key}", False)
        if self.config.terminal_control and select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            command = {"r": "start", "p": "pause", "q": "quit"}.get(line.strip())
            if not line:
                command = "quit"
            if command:
                action[f"control.{command}"] = True
        return action

    def send_feedback(self, feedback):
        if not self.is_connected:
            raise RuntimeError("Not connected")
        if feedback:
            raise NotImplementedError("XR haptics are not implemented")

    def disconnect(self):
        self._backend = None
        cleanup, self._cleanup = self._cleanup, None
        if cleanup is not None:
            cleanup.close()
