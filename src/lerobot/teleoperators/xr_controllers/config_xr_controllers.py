# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
from dataclasses import dataclass, field

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xr_controllers")
@dataclass(kw_only=True)
class XRControllersConfig(TeleoperatorConfig):
    app_name: str = "LeRobot XR"
    # OpenXR (right, up, backward) -> robot (forward, left, up).
    base_T_anchor: list[list[float]] = field(  # noqa: N815
        default_factory=lambda: [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )

    def __post_init__(self):
        transform = np.asarray(self.base_T_anchor, dtype=float)
        if (
            transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
        ):
            raise ValueError("base_T_anchor must be a finite rigid transform")
