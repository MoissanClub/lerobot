#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Download pinned G1-29 simulation/IK assets and GR00T policies. No robot I/O."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from lerobot.robots.unitree_g1.g1_vr_control import MODEL_REVISION, POLICY_REVISION


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--replay-frames", type=int, default=0)
    args = parser.parse_args()
    assets = snapshot_download("lerobot/unitree-g1-mujoco", revision=MODEL_REVISION)
    for name in ("Balance", "Walk"):
        hf_hub_download(
            "nepyope/GR00T-WholeBodyControl_g1",
            revision=POLICY_REVISION,
            filename=f"GR00T-WholeBodyControl-{name}.onnx",
        )
    print(assets)
    if args.config_dir:
        import numpy as np

        from lerobot.robots.unitree_g1.g1_vr_control import G1VRKinematics, synthetic_input

        directory = args.config_dir.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        processor = directory / "processor.json"
        processor.write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "class": "lerobot.robots.unitree_g1.g1_vr_processor.G1VRActionProcessor",
                            "config": {"assets": assets},
                        }
                    ]
                },
                indent=2,
            )
        )
        teleop = {
            "type": "xr_controllers",
            "full_input": True,
            "base_T_anchor": np.eye(4).tolist(),
            "terminal_control": not bool(args.replay_frames),
        }
        if args.replay_frames:
            replay = directory / "input.jsonl"
            wrists = G1VRKinematics(assets).fk(np.zeros(14))
            with replay.open("w") as stream:
                for step in range(args.replay_frames):
                    action = synthetic_input(wrists, step)
                    action["control.start"] = step == 0
                    stream.write(json.dumps(action, default=lambda x: x.tolist()) + "\n")
            teleop["replay_path"] = str(replay)
        config = directory / "teleoperate.json"
        config.write_text(
            json.dumps(
                {
                    "robot": {
                        "type": "unitree_g1_motion",
                        "assets": assets,
                        "report_path": str(directory / "report.jsonl"),
                    },
                    "teleop": teleop,
                    "fps": 50,
                    "display_data": False,
                    "teleop_action_processor_path": str(processor),
                },
                indent=2,
            )
        )
        print(f"lerobot-teleoperate --config_path={config}")


if __name__ == "__main__":
    main()
