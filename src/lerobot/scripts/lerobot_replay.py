# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Replays the actions of an episode from a dataset on a robot.

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=<USER>/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import cv2
import numpy as np
import rerun as rr

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    display_data: bool = False
    display_recorded_cameras: bool = False
    recorded_preview_width: int = 320
    recorded_preview_max_bytes: int = 256_000_000
    video_backend: str = "pyav"


def prepare_recorded_previews(dataset, episode, width, max_bytes):
    """Decode/resize before robot connection; keep bounded JPEG thumbnails in memory."""
    if width < 1 or max_bytes < 1:
        raise ValueError("Invalid recorded preview limits")
    previews = []
    used = 0
    for index, row in enumerate(dataset.hf_dataset):
        if int(row["episode_index"]) != episode:
            continue
        sample = dataset[index]
        frame = {}
        for key in dataset.meta.video_keys:
            value = sample[key]
            image = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            if image.ndim != 3:
                raise ValueError(f"Invalid recorded camera shape: {key}")
            if image.shape[0] in (1, 3, 4):
                image = np.transpose(image, (1, 2, 0))
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            image = cv2.resize(image, (width, max(1, round(image.shape[0] * width / image.shape[1]))))
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("Cannot encode reference thumbnail")
            payload = encoded.tobytes()
            used += len(payload)
            if used > max_bytes:
                raise ValueError(
                    "Recorded preview memory budget exceeded; disable recorded previews or shorten episode"
                )
            frame[key.removeprefix("observation.images.")] = payload
        previews.append(frame)
    return previews


def show_replay_frame(observation, action, recorded, index):
    rr.set_time_sequence("replay_frame", index)
    images = dict(observation)
    for key, encoded in recorded.items():
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Cannot decode reference preview")
        images[f"recorded.{key}"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    log_rerun_data(images, action, compress_images=True)


def replay_layout(camera_names, recorded_names):
    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(name=f"Live {name}", origin=f"observation.{name}") for name in camera_names]
    views += [
        rrb.Spatial2DView(name=f"Recorded {name}", origin=f"observation.recorded.{name}")
        for name in recorded_names
    ]
    if views:
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*views)))


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.episode],
        video_backend=cfg.video_backend,
    )

    # Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)
    actions = episode_frames.select_columns(ACTION)

    if cfg.display_recorded_cameras and not cfg.display_data:
        raise ValueError("Recorded camera preview requires display_data=true")
    previews = (
        prepare_recorded_previews(
            dataset, cfg.dataset.episode, cfg.recorded_preview_width, cfg.recorded_preview_max_bytes
        )
        if cfg.display_recorded_cameras
        else None
    )
    if previews is not None and len(previews) != len(episode_frames):
        raise ValueError("Reference preview/action frame mismatch")
    if cfg.display_data:
        init_rerun("so101_replay_comparison")
        replay_layout(getattr(robot, "cameras", {}), previews[0] if previews else {})
    robot.connect()

    try:
        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        for idx in range(len(episode_frames)):
            start_episode_t = time.perf_counter()

            action_array = actions[idx][ACTION]
            action = {}
            for i, name in enumerate(dataset.features[ACTION]["names"]):
                action[name] = action_array[i]

            robot_obs = robot.get_observation()

            processed_action = robot_action_processor((action, robot_obs))

            _ = robot.send_action(processed_action)
            if cfg.display_data:
                show_replay_frame(
                    robot_obs, processed_action, previews[idx] if previews is not None else {}, idx
                )

            dt_s = time.perf_counter() - start_episode_t
            precise_sleep(max(1 / dataset.fps - dt_s, 0.0))
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
