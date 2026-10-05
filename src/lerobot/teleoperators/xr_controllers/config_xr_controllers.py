# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
from dataclasses import dataclass

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xr_controllers")
@dataclass(kw_only=True)
class XRControllersConfig(TeleoperatorConfig):
    app_name: str = "LeRobot XR"
    full_input: bool = False
    terminal_control: bool = False
    replay_path: str | None = None
    record_path: str | None = None
    video_channel: str | None = None
    video_source: str = "g1-29-simulation"
    video_openxr_composition: bool = True
    cloudxr_config: str | None = None
    accept_cloudxr_eula: bool = False
    # OpenXR (right, up, backward) -> robot (forward, left, up).
    base_T_anchor: list[list[float]] | None = None  # noqa: N815

    def __post_init__(self):
        if self.base_T_anchor is None:
            # Full input includes head/sticks for embodiment processors, which consume raw OpenXR poses.
            self.base_T_anchor = (
                np.eye(4).tolist()
                if self.full_input
                else [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
            )
        if self.cloudxr_config and not self.accept_cloudxr_eula:
            raise ValueError("Starting CloudXR requires explicit EULA acceptance")
        if self.replay_path and (self.video_channel or self.cloudxr_config):
            raise ValueError("Replay does not open a live headset session")
        transform = np.asarray(self.base_T_anchor, dtype=float)
        if (
            transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
        ):
            raise ValueError("base_T_anchor must be a finite rigid transform")
