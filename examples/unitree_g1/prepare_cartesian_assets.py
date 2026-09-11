#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Download pinned, checksum-verified URDFs; meshes are optional for playback."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

HUB_REVISION = "a38dc8617f0fca51b38e9354dc58ee35ad850fb5"
G1_23_REVISION = "41c2805742de879ddab2d8d6beaeaf215f876395"
G1_23_URL = f"https://raw.githubusercontent.com/unitreerobotics/unitree_lerobot/{G1_23_REVISION}/unitree_lerobot/eval_robot/assets/g1/g1_body23.urdf"
CHECKSUMS = {
    "g1_29": "8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735",
    "g1_23": "b1af86fb023c0b6f8e52723d224be6cad70916eaff2778e7dbe09e6f91faa9b9",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meshes", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path29 = hf_hub_download(
        "lerobot/unitree-g1-mujoco", "assets/g1_body29_hand14.urdf", revision=HUB_REVISION
    )
    content = {"g1_29": Path(path29).read_bytes()}
    cached23 = args.output / "g1_23.urdf"
    if cached23.is_file() and hashlib.sha256(cached23.read_bytes()).hexdigest() == CHECKSUMS["g1_23"]:
        content["g1_23"] = cached23.read_bytes()
    else:
        with urllib.request.urlopen(G1_23_URL, timeout=30) as response:
            content["g1_23"] = response.read()
    for name, data in content.items():
        if hashlib.sha256(data).hexdigest() != CHECKSUMS[name]:
            raise ValueError(f"{name}: pinned URDF checksum mismatch")
        (args.output / f"{name}.urdf").write_bytes(data)
    if args.meshes:
        snapshot = snapshot_download(
            "lerobot/unitree-g1-mujoco", revision=HUB_REVISION, allow_patterns=["assets/meshes/*"]
        )
        shutil.copytree(Path(snapshot) / "assets/meshes", args.output / "meshes", dirs_exist_ok=True)
    manifest = {
        "hub_revision": HUB_REVISION,
        "g1_23_revision": G1_23_REVISION,
        "g1_23_url": G1_23_URL,
        "sha256": CHECKSUMS,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
