# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-free validation of the public Cartesian input contract."""

import numpy as np
import pytest

from lerobot.robots.unitree_g1.g1_cartesian_control import G1CartesianConfig, _pose, _vector


@pytest.mark.parametrize("embodiment", ["g1_29", "g1_23"])
def test_config_does_not_load_optional_dependencies_or_assets(embodiment):
    assert G1CartesianConfig(embodiment, "not-downloaded.urdf").embodiment == embodiment


@pytest.mark.parametrize("field", ["position_tolerance_m", "orientation_tolerance_rad", "max_joint_step_rad"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_tolerances(field, value):
    with pytest.raises(ValueError):
        G1CartesianConfig("g1_23", "unused", **{field: value})


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_iteration_budget(value):
    with pytest.raises(ValueError):
        G1CartesianConfig("g1_23", "unused", max_iterations=value)


def test_unknown_embodiment():
    with pytest.raises(ValueError, match="Unknown"):
        G1CartesianConfig("g1_27", "unused")


@pytest.mark.parametrize(
    "value", [np.zeros((3, 3)), np.zeros((4, 4)), np.full((4, 4), np.nan), np.diag([-1, 1, 1, 1])]
)
def test_invalid_pose(value):
    with pytest.raises(ValueError):
        _pose(value)


def test_pose_and_vector_are_copied():
    pose, vector = np.eye(4), np.zeros(10)
    assert not np.shares_memory(pose, _pose(pose))
    assert not np.shares_memory(vector, _vector(vector, 10, "q"))


@pytest.mark.parametrize("value", [np.zeros(14), np.zeros((10, 1)), np.full(10, np.inf)])
def test_invalid_arm_vector(value):
    with pytest.raises(ValueError):
        _vector(value, 10, "q")
