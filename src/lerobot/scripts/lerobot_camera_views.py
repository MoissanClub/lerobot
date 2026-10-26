"""Camera-only view identification/orientation. Never imports robot or motor code."""

import argparse
import json
import select
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import rerun as rr
import rerun.blueprint as rrb

from lerobot.utils.rerun_viewer import prepare_rerun_path


def rotate_image(image, degrees):
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, -90: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(image, codes[degrees]) if degrees else image


def save_views(directory, views):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = [directory / "record.json", directory / "replay.json"]
    configs = [json.loads(path.read_text()) for path in paths]
    cameras = {}
    for index, view in enumerate(views):
        width, height = view["width"], view["height"]
        if view["rotation"] in (90, -90):
            width, height = height, width
        cameras[f"camera_{index}"] = {
            "type": "opencv",
            "index_or_path": view["index"],
            "width": width,
            "height": height,
            "fps": 30,
            "rotation": view["rotation"],
        }
    name = "so101_cameras_" + stamp
    root = Path(configs[0]["dataset"]["root"]).parent / name
    for path, config in zip(paths, configs, strict=True):
        shutil.copy2(path, path.with_name(path.stem + ".before-cameras-" + stamp + ".json"))
        config["robot"]["cameras"] = cameras
        config["dataset"]["repo_id"] = "local/" + name
        config["dataset"]["root"] = str(root)
        config["display_data"] = True
        if path.stem == "record":
            config["dataset"].update(
                video=True, vcodec="h264", single_task="SO101 camera recording and replay comparison"
            )
            config.setdefault("catch_up_warning", {})["telemetry_path"] = str(
                directory / (name + ".telemetry.jsonl")
            )
        else:
            config["display_recorded_cameras"] = True
            config["video_backend"] = "pyav"
        path.write_text(json.dumps(config, indent=2) + "\n")
    (directory / "camera_views.json").write_text(
        json.dumps(
            {
                "confirmed_at": stamp,
                "views": views,
                "cameras": cameras,
                "scope": "device mapping and orientation only; no lens or camera-to-robot calibration",
            },
            indent=2,
        )
        + "\n"
    )
    return cameras


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indices", type=int, nargs=2, default=[0, 1])
    p.add_argument("--config-dir", type=Path, required=True)
    args = p.parse_args()
    if len(set(args.indices)) != 2:
        p.error("Choose two distinct indices")
    for filename in ("record.json", "replay.json"):
        if not (args.config_dir / filename).is_file():
            p.error(f"Missing {filename}")
    if not sys.stdin.isatty():
        p.error("Use an interactive Terminal for view confirmation")
    captures = []
    views = []
    try:
        for index in args.indices:
            capture = cv2.VideoCapture(index)
            captures.append(capture)
            if not capture.isOpened():
                raise RuntimeError(
                    f"Cannot open camera {index}; check connection/Terminal camera permission or choose --indices"
                )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, 30)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Camera {index} produced no frame")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if abs(fps - 30) > 0.5:
                raise RuntimeError(
                    f"Camera {index} reports {fps} FPS, not required 30; select a supported mode before recording"
                )
            views.append({"index": index, "rotation": 0, "width": frame.shape[1], "height": frame.shape[0]})
            print(f"Camera {index}: {frame.shape[1]}x{frame.shape[0]}, reported FPS={fps}", flush=True)
        prepare_rerun_path()
        rr.init("SO101_camera_view_setup", spawn=True)
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    *[rrb.Spatial2DView(name=f"camera_{i}", origin=f"camera_{i}") for i in range(2)]
                )
            )
        )
        print(
            "Cover one lens to identify its view. Commands + Enter: swap | rotate 0 | rotate 1 | save | quit",
            flush=True,
        )
        frames = 0
        while True:
            images = {}
            for capture, index in zip(captures, args.indices, strict=True):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Lost camera {index}")
                images[index] = image
            rr.set_time_sequence("frame", frames)
            for slot, view in enumerate(views):
                image = rotate_image(images[view["index"]], view["rotation"])
                label = f"camera_{slot}: device {view['index']}, rotation {view['rotation']}"
                cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                rr.log(f"camera_{slot}", rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).compress())
            frames += 1
            if select.select([sys.stdin], [], [], 0)[0]:
                command = sys.stdin.readline().strip().lower()
                if command == "swap":
                    views.reverse()
                elif command in ("rotate 0", "rotate 1"):
                    slot = int(command[-1])
                    rotations = [0, 90, 180, -90]
                    view = views[slot]
                    view["rotation"] = rotations[(rotations.index(view["rotation"]) + 1) % 4]
                elif command == "save":
                    print(json.dumps(save_views(args.config_dir, views), indent=2))
                    print("Saved mapping/orientation to record.json and replay.json. No arm was connected.")
                    break
                elif command == "quit":
                    break
                else:
                    print("Commands: swap | rotate 0 | rotate 1 | save | quit")
            time.sleep(0.01)
    finally:
        for capture in captures:
            capture.release()
        rr.rerun_shutdown()


if __name__ == "__main__":
    main()
