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

"""Fixed-base, dual-arm Cartesian control independent of XR, DDS, and simulation.

Public joint vectors are compact active-arm vectors in DDS order (14 or 10).
Poses are homogeneous transforms in the URDF root frame, in metres/radians.
Legs, waist, and finger joints are locked at neutral. No collision checking or
physical safety certification is implied by the joint/step bounds here.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .g1_embodiments import get_g1_embodiment


def _vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array.copy()


def _pose(value) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("Target must be a finite 4x4 transform")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Target must contain a proper rotation and homogeneous bottom row")
    return pose.copy()


@dataclass(frozen=True)
class G1CartesianConfig:
    embodiment: str
    urdf_path: str | Path
    position_tolerance_m: float = 0.005
    orientation_tolerance_rad: float = 0.05
    max_joint_step_rad: float = 0.1
    max_iterations: int = 100

    def __post_init__(self):
        get_g1_embodiment(self.embodiment)
        for name in ("position_tolerance_m", "orientation_tolerance_rad", "max_joint_step_rad"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")


@dataclass(frozen=True)
class G1IKResult:
    q: np.ndarray
    gravity_tau: np.ndarray
    position_error_m: tuple[float, float]
    orientation_error_rad: tuple[float, float]
    converged: bool
    status: str


class G1ArmKinematics:
    """One solver instance per control loop; FK/gravity use independent work data.

    ``solve`` returns bounded best-effort commands on a pose mismatch and holds
    the seed on optimizer failure. ``converged`` describes the *returned* pose,
    after step limiting, not just the optimizer's termination status.
    """

    def __init__(self, config: G1CartesianConfig):
        # Conda-forge Pinocchio has no Python distribution metadata for the usual
        # require_package guard. Import only on construction, as the legacy solver does.
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise ImportError("Install conda-forge pinocchio and casadi; see the G1 Cartesian guide") from exc

        self.config = config
        self.embodiment = get_g1_embodiment(config.embodiment)
        self._pin = pin
        urdf = Path(config.urdf_path).expanduser().resolve(strict=True)
        self.urdf_path = urdf
        self.full_model = pin.buildModelFromUrdf(str(urdf))
        joints = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
        if config.embodiment == "g1_29":
            joints += ("wrist_pitch", "wrist_yaw")
        self.joint_names = tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in joints)
        expected_body = len(self.embodiment.joint_index)
        body_names = [name for name in self.full_model.names[1:] if "hand" not in name]
        if len(body_names) != expected_body or not all(
            self.full_model.existJointName(n) for n in self.joint_names
        ):
            raise ValueError(f"URDF does not match {config.embodiment}: expected {expected_body} body joints")
        locked = [i for i, name in enumerate(self.full_model.names) if i and name not in self.joint_names]
        self.model = pin.buildReducedModel(self.full_model, locked, pin.neutral(self.full_model))
        self.size = len(self.joint_names)
        if self.model.nq != self.size or self.model.nv != self.size:
            raise ValueError("Expected scalar revolute arm joints")
        self._q_indices = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_q for n in self.joint_names]
        )
        self._v_indices = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_v for n in self.joint_names]
        )
        self.lower = self.model.lowerPositionLimit[self._q_indices].copy()
        self.upper = self.model.upperPositionLimit[self._q_indices].copy()
        if not np.all(np.isfinite([self.lower, self.upper])) or np.any(self.lower > self.upper):
            raise ValueError("Arm joints must have finite ordered limits")
        offset = 0.05 if config.embodiment == "g1_29" else 0.20
        self.ee_offset = np.array([offset, 0, 0])
        self.frame_ids = tuple(
            self.model.addFrame(
                pin.Frame(
                    f"{side}_cartesian_ee",
                    self.model.getJointId(f"{side}_{joints[-1]}_joint"),
                    pin.SE3(np.eye(3), self.ee_offset),
                    pin.FrameType.OP_FRAME,
                )
            )
            for side in ("left", "right")
        )
        self._last_q = np.clip(np.zeros(self.size), self.lower, self.upper)
        self._opti = None

    def _to_model(self, q) -> np.ndarray:
        q = _vector(q, self.size, "Arm positions in DDS order")
        result = np.empty(self.size)
        result[self._q_indices] = q
        return result

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        data = self.model.createData()
        self._pin.framesForwardKinematics(self.model, data, self._to_model(q))
        return tuple(data.oMf[i].homogeneous.copy() for i in self.frame_ids)

    def gravity(self, q) -> np.ndarray:
        """Static feedforward in Nm; no velocity, acceleration, or payload term."""
        data = self.model.createData()
        tau = self._pin.computeGeneralizedGravity(self.model, data, self._to_model(q))
        return tau[self._v_indices].copy()

    def arm_action(self, q) -> dict[str, float]:
        """Arm-only action; never implicitly zero legs or waist."""
        values = _vector(q, self.size, "Arm positions")
        return {
            f"{joint.name}.q": float(value)
            for joint, value in zip(self.embodiment.arm_index, values, strict=True)
        }

    def scatter_arm(self, values) -> np.ndarray:
        """Scatter compact arm values into the 29-slot body transport layout."""
        compact = _vector(values, self.size, "Arm values")
        body = np.zeros(29)
        body[[joint.value for joint in self.embodiment.arm_index]] = compact
        return body

    def reset(self, q=None) -> None:
        values = (
            np.clip(np.zeros(self.size), self.lower, self.upper)
            if q is None
            else _vector(q, self.size, "Seed")
        )
        if np.any(values < self.lower) or np.any(values > self.upper):
            raise ValueError("Seed exceeds joint limits")
        self._last_q = values.copy()

    def _build_optimizer(self):
        try:
            import casadi as ca
            from pinocchio import casadi as cpin
        except ImportError as exc:
            raise ImportError("IK needs conda-forge pinocchio with CasADi bindings and casadi") from exc

        model = cpin.Model(self.model)
        data = model.createData()
        opti = ca.Opti()
        q = opti.variable(self.size)
        seed = opti.parameter(self.size)
        targets = [opti.parameter(4, 4), opti.parameter(4, 4)]
        # Build expressions in model order; all optimizer variables remain DDS ordered.
        symbolic_q = ca.SX.sym("arm_q", self.size)
        cpin.framesForwardKinematics(model, data, symbolic_q[np.argsort(self._q_indices).tolist()])
        cost = 1e-3 * ca.sumsqr(q - seed)
        for frame, target in zip(self.frame_ids, targets, strict=True):
            placement = ca.Function(f"frame_{frame}", [symbolic_q], [data.oMf[frame].homogeneous])(q)
            cost += 100 * ca.sumsqr(placement[:3, 3] - target[:3, 3])
            # Chordal SO(3) residual is smooth at identity, unlike a naive log map.
            cost += 0.5 * ca.sumsqr(placement[:3, :3] - target[:3, :3])
        opti.subject_to(opti.bounded(self.lower, q, self.upper))
        opti.subject_to(
            opti.bounded(-self.config.max_joint_step_rad, q - seed, self.config.max_joint_step_rad)
        )
        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {
                "print_time": False,
                "calc_lam_p": False,
                "ipopt": {"print_level": 0, "sb": "yes", "max_iter": self.config.max_iterations, "tol": 1e-8},
            },
        )
        self._opti, self._q, self._seed, self._targets = opti, q, seed, targets

    def _optimize(self, targets, seed):
        self._opti.set_initial(self._q, seed)
        self._opti.set_value(self._seed, seed)
        for param, target in zip(self._targets, targets, strict=True):
            self._opti.set_value(param, target)
        return np.asarray(self._opti.solve().value(self._q)).reshape(self.size)

    def solve(self, left, right, seed=None) -> G1IKResult:
        targets = (_pose(left), _pose(right))
        seed = self._last_q.copy() if seed is None else _vector(seed, self.size, "Seed")
        if np.any(seed < self.lower) or np.any(seed > self.upper):
            raise ValueError("Seed exceeds joint limits")
        if self._opti is None:
            self._build_optimizer()
        failed = False
        try:
            candidate = _vector(self._optimize(targets, seed), self.size, "IK result")
        except (RuntimeError, ValueError):
            failed = True
            candidate = seed.copy()
        q = np.clip(
            seed + np.clip(candidate - seed, -self.config.max_joint_step_rad, self.config.max_joint_step_rad),
            self.lower,
            self.upper,
        )
        poses = self.fk(q)
        position = tuple(
            float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            for actual, target in zip(poses, targets, strict=True)
        )
        orientation = tuple(
            float(np.linalg.norm(self._pin.log3(actual[:3, :3] @ target[:3, :3].T)))
            for actual, target in zip(poses, targets, strict=True)
        )
        converged = (
            not failed
            and max(position) <= self.config.position_tolerance_m
            and max(orientation) <= self.config.orientation_tolerance_rad
        )
        self._last_q = q.copy()
        status = "optimizer_failed_hold" if failed else ("converged" if converged else "bounded_best_effort")
        return G1IKResult(q, self.gravity(q), position, orientation, converged, status)
