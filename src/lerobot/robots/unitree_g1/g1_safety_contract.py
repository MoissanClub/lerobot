# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Reviewed arm SDK configuration shared by Robot and diagnostics."""

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .g1_arm_sdk import G1ArmSDKConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_limits(urdf):
    joints = (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
    root = ET.parse(urdf).getroot()
    names = [f"{side}_{joint}_joint" for side in ("left", "right") for joint in joints]
    limits = {j.get("name"): j.find("limit") for j in root.findall("joint")}
    return {
        field: [float(limits[name].get(field)) for name in names] for field in ("lower", "upper", "effort")
    }


def contract_template(urdf):
    limits = model_limits(urdf)
    cfg = G1ArmSDKConfig(
        network_interface="REPLACE_WITH_ROBOT_INTERFACE",
        read_only=False,
        expected_mode_machine=None,
        kp=[60.0] * 14,
        kd=[1.5] * 14,
        lower=limits["lower"],
        upper=limits["upper"],
        torque_limits=[min(15.0, effort) for effort in limits["effort"]],
        gravity_urdf=str(urdf.absolute()),
        max_feedforward=15.0,
    )
    return {
        "schema": 1,
        "reviewed": False,
        "vr_pose_mapping_reviewed": False,
        "locomotion_reviewed": False,
        "damping_reviewed": False,
        "combined_ownership_reviewed": False,
        "base_limits": {"max_speed": 0.1, "max_yaw": 0.1, "timeout": 0.2},
        "reviewer": "",
        "robot_serial": "",
        "firmware": "",
        "mode_and_manufacturer_procedure": "",
        "support_and_stop_procedure": "",
        "payload_and_tool_model": "",
        "operator_notes": "",
        "robot_side_execution_confirmed": False,
        "exclusive_arm_publisher_confirmed": False,
        "stock_lower_body_owner_confirmed": False,
        "uncommanded_waist_semantics_verified": False,
        "packet_loss_and_process_death_response_verified": False,
        "release_weight_zero_behavior_reviewed": False,
        "gains_limits_and_gravity_reviewed": False,
        "urdf_sha256": digest(urdf),
        "arm_sdk": asdict(cfg),
        "notice": "EXAMPLE VALUES ONLY. Not approved hardware gains, torque ratings or stop limits.",
    }


def validate_arm_model(cfg: G1ArmSDKConfig, urdf: Path) -> None:
    """Check configured bounds and gravity model against the selected robot model."""
    limits = model_limits(urdf)
    if np.any(np.asarray(cfg.lower) < limits["lower"]) or np.any(np.asarray(cfg.upper) > limits["upper"]):
        raise ValueError("Configured joint bounds exceed model bounds")
    if np.any(np.asarray(cfg.torque_limits) > limits["effort"]):
        raise ValueError("Configured torque bounds exceed model effort limits")
    if cfg.gravity_urdf and Path(cfg.gravity_urdf).resolve() != urdf.resolve():
        raise ValueError("Gravity model must be the reviewed URDF")


def validate_contract(doc, urdf, hardware_motion):
    if doc.get("schema") != 1:
        raise ValueError("Unsupported contract schema")
    cfg = G1ArmSDKConfig(**doc["arm_sdk"])
    if urdf is not None:
        if doc.get("urdf_sha256") != digest(urdf):
            raise ValueError("URDF hash differs from reviewed contract")
        validate_arm_model(cfg, urdf)
    if hardware_motion:
        if urdf is None or cfg.read_only or cfg.expected_mode_machine is None:
            raise ValueError("Motion needs a reviewed URDF, expected mode and motion configuration")
        for key in (
            "reviewed",
            "robot_side_execution_confirmed",
            "exclusive_arm_publisher_confirmed",
            "stock_lower_body_owner_confirmed",
            "uncommanded_waist_semantics_verified",
            "packet_loss_and_process_death_response_verified",
            "release_weight_zero_behavior_reviewed",
            "gains_limits_and_gravity_reviewed",
        ):
            if doc.get(key) is not True:
                raise ValueError(f"Hardware motion gate not approved: {key}")
        for key in (
            "reviewer",
            "robot_serial",
            "firmware",
            "mode_and_manufacturer_procedure",
            "support_and_stop_procedure",
            "payload_and_tool_model",
        ):
            if not isinstance(doc.get(key), str) or not doc[key].strip():
                raise ValueError(f"Missing reviewed contract field: {key}")
    return cfg
