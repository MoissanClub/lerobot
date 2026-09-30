# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Optional installed-SDK schema audit; no OpenXR session is opened."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_full_input_reads_head_sticks_and_untracked_stop_buttons():
    pytest.importorskip("isaacteleop")
    from isaacteleop.retargeting_engine.tensor_types.indices import (
        ControllerInputIndex as C,
        HeadPoseIndex as H,
    )

    from lerobot.teleoperators.xr_controllers import XRControllersConfig
    from lerobot.teleoperators.xr_controllers.xr_controllers import IsaacControllerSession

    session = IsaacControllerSession(XRControllersConfig(full_input=True))
    controller = {
        C.GRIP_POSITION: [0, 0, 0],
        C.GRIP_ORIENTATION: [0, 0, 0, 1],
        C.GRIP_IS_VALID: False,
        C.SQUEEZE_VALUE: 0.2,
        C.TRIGGER_VALUE: 0.3,
        C.THUMBSTICK_X: 0.1,
        C.THUMBSTICK_Y: 0.4,
        C.THUMBSTICK_CLICK: True,
        C.PRIMARY_CLICK: True,
        C.SECONDARY_CLICK: False,
    }
    head = {H.IS_VALID: True, H.POSITION: [0, 1.6, 0], H.ORIENTATION: [0, 0, 0, 1]}
    session.session = SimpleNamespace(
        last_step_info=None, step=lambda **kw: {"left": controller, "right": controller, "head": head}
    )
    action = session.read()
    assert action["head.tracked"] and not action["left.tracked"]
    assert action["right.primary"] and action["left.stick_click"]
    np.testing.assert_allclose(action["head.pos"], [0, 1.6, 0])
    controller[C.GRIP_IS_VALID] = True
    action = session.read()
    assert action["left.stick_y"] == 0.4 and action["left.trigger"] == 0.3
    session.session.last_step_info = SimpleNamespace(worker_exception=None, frame_deadline_miss=True)
    action = session.read()
    assert not action["head.tracked"] and not action["right.primary"]
