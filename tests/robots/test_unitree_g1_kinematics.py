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

"""Real-model numerical acceptance; opt in with G1_KINEMATICS_ASSETS.

When enabled, missing models or dependencies are failures, not skips.
See examples/unitree_g1/prepare_cartesian_assets.py for the pinned fixture assets.
"""

import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.unitree_g1.g1_cartesian_control import G1ArmKinematics, G1CartesianConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("G1_KINEMATICS_ASSETS"), reason="Set G1_KINEMATICS_ASSETS for model acceptance"
)


@pytest.fixture(params=["g1_29", "g1_23"])
def model(request):
    root = Path(os.environ["G1_KINEMATICS_ASSETS"])
    path = root / f"{request.param}.urdf"
    hashes = {
        "g1_29": "8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735",
        "g1_23": "b1af86fb023c0b6f8e52723d224be6cad70916eaff2778e7dbe09e6f91faa9b9",
    }
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[request.param]
    return G1ArmKinematics(G1CartesianConfig(request.param, path))


def posture(model):
    q = np.zeros(model.size)
    half = model.size // 2
    q[0], q[half] = -0.4, -0.35
    q[1], q[half + 1] = 0.2, -0.2
    q[3], q[half + 3] = 0.7, 0.8
    return q


def test_fk_matches_unreduced_urdf(model):
    pin = model._pin
    q = posture(model)
    full_q = pin.neutral(model.full_model)
    for name, value in zip(model.joint_names, q, strict=True):
        full_q[model.full_model.joints[model.full_model.getJointId(name)].idx_q] = value
    data = model.full_model.createData()
    pin.forwardKinematics(model.full_model, data, full_q)
    for actual, name in zip(
        model.fk(q), [model.joint_names[model.size // 2 - 1], model.joint_names[-1]], strict=True
    ):
        expected = data.oMi[model.full_model.getJointId(name)] * pin.SE3(np.eye(3), model.ee_offset)
        np.testing.assert_allclose(actual, expected.homogeneous, atol=1e-12)


def test_gravity_matches_potential_energy_gradient(model):
    q = posture(model)
    eps = 1e-6
    gradient = []
    for i in range(model.size):
        delta = np.zeros(model.size)
        delta[i] = eps
        energy = [
            model._pin.computePotentialEnergy(
                model.model, model.model.createData(), model._to_model(q + sign * delta)
            )
            for sign in (1, -1)
        ]
        gradient.append((energy[0] - energy[1]) / (2 * eps))
    np.testing.assert_allclose(model.gravity(q), gradient, atol=1e-7, rtol=1e-6)
    assert np.linalg.norm(model.gravity(q)) > 0.1


def test_nonidentity_motor_to_model_order(model):
    permuted = copy.copy(model)
    permuted._q_indices = model._q_indices[::-1].copy()
    permuted._v_indices = model._v_indices[::-1].copy()
    q = posture(model)
    np.testing.assert_allclose(permuted.fk(q), model.fk(q[::-1]), atol=1e-12)
    np.testing.assert_allclose(permuted.gravity(q), model.gravity(q[::-1])[::-1], atol=1e-12)


def test_arm_action_and_sparse_scatter(model):
    q = posture(model)
    action = model.arm_action(q)
    assert len(action) == model.size
    assert not any("Hip" in key or "Waist" in key for key in action)
    body = model.scatter_arm(q)
    slots = [j.value for j in model.embodiment.arm_index]
    np.testing.assert_array_equal(body[slots], q)
    np.testing.assert_array_equal(np.delete(body, slots), 0)


def test_reachable_ik_round_trip_and_continuity(model):
    goal = posture(model)
    targets = model.fk(goal)
    seed = np.clip(goal + 0.03, model.lower, model.upper)
    for _ in range(15):
        result = model.solve(*targets, seed=seed)
        assert np.max(np.abs(result.q - seed)) <= model.config.max_joint_step_rad + 1e-12
        assert np.all(result.q >= model.lower) and np.all(result.q <= model.upper)
        seed = result.q
    assert result.converged, (result.position_error_m, result.orientation_error_rad)
    np.testing.assert_allclose(result.gravity_tau, model.gravity(result.q))


def test_unreachable_target_is_bounded_and_not_success(model):
    seed = posture(model)
    left, right = model.fk(seed)
    left[:3, 3] += 10
    result = model.solve(left, right, seed)
    assert not result.converged
    assert np.max(np.abs(result.q - seed)) <= model.config.max_joint_step_rad + 1e-12
    assert np.all(np.isfinite(result.gravity_tau))


def test_optimizer_failure_holds_seed_and_does_not_poison_state(model, monkeypatch):
    seed = posture(model)
    model.reset(seed)

    def fail(*args):
        raise RuntimeError("forced failure")

    with monkeypatch.context() as patch:
        patch.setattr(model, "_optimize", fail)
        result = model.solve(*model.fk(seed))
        assert result.status == "optimizer_failed_hold"
        assert not result.converged
        np.testing.assert_array_equal(result.q, seed)
    assert model.solve(*model.fk(seed)).converged


def test_nonfinite_optimizer_output_holds_seed(model, monkeypatch):
    seed = posture(model)
    monkeypatch.setattr(model, "_optimize", lambda *args: np.full(model.size, np.nan))
    result = model.solve(*model.fk(seed), seed)
    assert result.status == "optimizer_failed_hold"
    np.testing.assert_array_equal(result.q, seed)


def test_invalid_input_does_not_mutate_state(model):
    seed = posture(model)
    model.reset(seed)
    with pytest.raises(ValueError):
        model.solve(np.zeros((4, 4)), np.eye(4))
    np.testing.assert_array_equal(model._last_q, seed)
    with pytest.raises(ValueError):
        model.solve(*model.fk(seed), model.upper + 1)


def test_wrong_embodiment_model_is_rejected(model):
    other = "g1_23" if model.config.embodiment == "g1_29" else "g1_29"
    with pytest.raises(ValueError, match="does not match"):
        G1ArmKinematics(G1CartesianConfig(other, model.urdf_path))


def test_reachable_moving_targets_and_independent_arms(model):
    base = posture(model)
    seed = base.copy()
    right = model.fk(base)[1]
    for angle in np.linspace(0, 2 * np.pi, 41):
        goal = base.copy()
        goal[0] += 0.10 * np.sin(angle)
        goal[1] += 0.08 * np.sin(angle)
        goal[3] += 0.10 * np.sin(angle)
        goal[4] += 0.20 * np.sin(angle)
        result = model.solve(model.fk(goal)[0], right, seed)
        assert result.converged, result.status
        assert np.max(np.abs(result.q - seed)) <= model.config.max_joint_step_rad + 1e-12
        np.testing.assert_allclose(model.fk(result.q)[1], right, atol=1e-5)
        seed = result.q


def test_ik_with_nonidentity_model_order(model):
    permuted = copy.copy(model)
    permuted._q_indices = model._q_indices[::-1].copy()
    permuted._v_indices = model._v_indices[::-1].copy()
    permuted.lower, permuted.upper = model.lower[::-1].copy(), model.upper[::-1].copy()
    permuted._opti = None
    goal = posture(model)[::-1].copy()
    seed = goal + 0.02
    for _ in range(5):
        result = permuted.solve(*permuted.fk(goal), seed)
        seed = result.q
    assert result.converged
    np.testing.assert_allclose(result.gravity_tau, model.gravity(result.q[::-1])[::-1], atol=1e-12)


def test_cartesian_rank_exposes_underactuated_g1_23(model):
    jacobian = model._pin.computeFrameJacobian(
        model.model,
        model.model.createData(),
        model._to_model(posture(model)),
        model.frame_ids[0],
        model._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    assert np.linalg.matrix_rank(jacobian[:, model._v_indices[: model.size // 2]], tol=1e-7) == min(
        6, model.size // 2
    )
