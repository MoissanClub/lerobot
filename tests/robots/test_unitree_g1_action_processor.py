# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.configs import PipelineFeatureType
from lerobot.processor import TransitionKey
from lerobot.robots.unitree_g1.g1_action_processor import G1CartesianActionProcessor
from lerobot.robots.unitree_g1.g1_embodiments import get_g1_embodiment


@pytest.mark.parametrize("embodiment", ["g1_29", "g1_23"])
def test_processor_contract_and_reset(embodiment):
    processor = G1CartesianActionProcessor(embodiment, "model.urdf")
    keys = [f"{joint.name}.q" for joint in get_g1_embodiment(embodiment).arm_index]
    measured = dict.fromkeys(keys, 0.1)
    solver = processor._kinematics = MagicMock()
    solver.solve.return_value = SimpleNamespace(q=np.full(len(keys), 0.2))
    solver.arm_action.return_value = dict.fromkeys(keys, 0.2)
    action = {"left.ee_pose": np.eye(4), "right.ee_pose": np.eye(4), "hand.grasp": 0.4}
    output = processor({TransitionKey.ACTION: action, TransitionKey.OBSERVATION: measured})
    assert output[TransitionKey.ACTION] == {**dict.fromkeys(keys, 0.2), "hand.grasp": 0.4}
    np.testing.assert_array_equal(solver.solve.call_args.kwargs["seed"], list(measured.values()))
    assert "left.ee_pose" in action
    assert output[TransitionKey.OBSERVATION] is measured
    features = processor.transform_features({PipelineFeatureType.ACTION: dict.fromkeys(action)})
    assert set(features[PipelineFeatureType.ACTION]) == {*keys, "hand.grasp"}
    assert G1CartesianActionProcessor(**processor.get_config()).get_config() == processor.get_config()
    processor.reset()
    solver.reset.assert_called_once()
    assert processor.last_result is None


def test_processor_requires_observation():
    with pytest.raises(ValueError, match="measured"):
        G1CartesianActionProcessor()({TransitionKey.ACTION: {}})
