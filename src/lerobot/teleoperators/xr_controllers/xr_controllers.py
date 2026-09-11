# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Dual controller reader adapted from the Isaac Teleop SO-101 example.

CloudXR is externally owned: this device never installs or starts a runtime.
SDK imports are delayed until connect; replay and configuration need no OpenXR.
"""

import time

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
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner, TensorGroup, ValueInput
        from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

        source = ControllersSource(name="controllers")
        transform = ValueInput("base_T_anchor", TransformMatrix())
        transformed = source.transformed(transform.output("value"))
        self.pipeline = OutputCombiner(
            {side: transformed.output(f"controller_{side}") for side in ("left", "right")}
        )
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
            controller = result[side]
            if fresh and controller is not None and not getattr(controller, "is_none", False):
                try:
                    candidate = {
                        "grip_pos": np.asarray(controller[Index.GRIP_POSITION]),
                        "grip_quat": np.asarray(controller[Index.GRIP_ORIENTATION]),
                        "squeeze": float(controller[Index.SQUEEZE_VALUE]),
                        "trigger": float(controller[Index.TRIGGER_VALUE]),
                        "tracked": True,
                    }
                    if controller_pose({f"{side}.{k}": v for k, v in candidate.items()}, side) is not None:
                        values = candidate
                except (KeyError, IndexError, TypeError, ValueError):
                    pass
            action.update({f"{side}.{key}": val for key, val in values.items()})
        return action

    def close(self):
        if self.entered:
            self.entered = False
            session, self.session = self.session, None
            session.__exit__(None, None, None)


class XRControllers(Teleoperator):
    config_class = XRControllersConfig
    name = "xr_controllers"

    def __init__(self, config, *, session_factory=IsaacControllerSession):
        self._backend = None
        super().__init__(config)
        self.config, self._factory = config, session_factory

    @property
    def action_features(self):
        features = {"captured_at": float}
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
        backend = self._factory(self.config)
        try:
            backend.connect()
        except BaseException:
            backend.close()
            raise
        self._backend = backend

    def get_action(self):
        if not self.is_connected:
            raise RuntimeError("Not connected")
        return self._backend.read()

    def send_feedback(self, feedback):
        if not self.is_connected:
            raise RuntimeError("Not connected")
        if feedback:
            raise NotImplementedError("XR haptics are not implemented")

    def disconnect(self):
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.close()
