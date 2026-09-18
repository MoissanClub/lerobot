# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Cartesian action adaptation; numerical IK remains independent of processors."""

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep, TransitionKey

from .g1_cartesian_control import G1ArmKinematics, G1CartesianConfig
from .g1_embodiments import get_g1_embodiment


@ProcessorStepRegistry.register("g1_cartesian_to_joints")
@dataclass
class G1CartesianActionProcessor(RobotActionProcessorStep):
    """Consume root-frame 4x4 targets and measured radians, emit named joint radians.

    ``left.ee_pose`` and ``right.ee_pose`` are intermediate robot actions, not
    flattened dataset features. Gravity feedforward is owned by robot execution.
    """

    embodiment: str = "g1_29"
    urdf_path: str = ""
    max_joint_step_rad: float = 0.1
    _kinematics: object = field(default=None, init=False, repr=False)
    last_result: object = field(default=None, init=False, repr=False)

    @property
    def kinematics(self):
        if self._kinematics is None:
            self._kinematics = G1ArmKinematics(
                G1CartesianConfig(self.embodiment, self.urdf_path, max_joint_step_rad=self.max_joint_step_rad)
            )
        return self._kinematics

    def action(self, action):
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("G1 Cartesian control requires measured joint observations")
        spec = get_g1_embodiment(self.embodiment)
        keys = [f"{joint.name}.q" for joint in spec.arm_index]
        measured = np.array([observation[key] for key in keys], dtype=float)
        if not np.all(np.isfinite(measured)):
            raise ValueError("Measured arm joints must be finite")
        targets = [action.pop(f"{side}.ee_pose") for side in ("left", "right")]
        self.last_result = self.kinematics.solve(*targets, seed=measured)
        action.update(self.kinematics.arm_action(self.last_result.q))
        return action

    def transform_features(self, features):
        actions = features[PipelineFeatureType.ACTION]
        for side in ("left", "right"):
            actions.pop(f"{side}.ee_pose", None)
        for joint in get_g1_embodiment(self.embodiment).arm_index:
            actions[f"{joint.name}.q"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features

    def get_config(self):
        return {
            "embodiment": self.embodiment,
            "urdf_path": self.urdf_path,
            "max_joint_step_rad": self.max_joint_step_rad,
        }

    def reset(self):
        self.last_result = None
        if self._kinematics is not None:
            self._kinematics.reset()
