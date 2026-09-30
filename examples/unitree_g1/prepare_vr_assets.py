#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Download pinned G1-29 simulation/IK assets and GR00T policies. No robot I/O."""

from huggingface_hub import hf_hub_download, snapshot_download

from lerobot.robots.unitree_g1.g1_vr_control import MODEL_REVISION, POLICY_REVISION


def main():
    assets = snapshot_download("lerobot/unitree-g1-mujoco", revision=MODEL_REVISION)
    for name in ("Balance", "Walk"):
        hf_hub_download(
            "nepyope/GR00T-WholeBodyControl_g1",
            revision=POLICY_REVISION,
            filename=f"GR00T-WholeBodyControl-{name}.onnx",
        )
    print(assets)


if __name__ == "__main__":
    main()
