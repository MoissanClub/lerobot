# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Resolve the pinned G1 model on demand, with an optional local asset override."""

from huggingface_hub import snapshot_download

from .g1_vr_control import MODEL_REVISION


def resolve_g1_vr_assets(assets: str | None) -> str:
    return assets or snapshot_download("lerobot/unitree-g1-mujoco", revision=MODEL_REVISION)
