"""Run editable LeRobot teleop/record commands with the local SO101 configuration.

Examples:
    python so101_dev.py teleop
    python so101_dev.py record --dataset.repo_id=local/trial01 \
        --dataset.single_task="Pick up the block" --dataset.num_episodes=2
    python so101_dev.py record --help
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("teleop", "record"):
        raise SystemExit(
            "Usage: python so101_dev.py {teleop|record} [--check] [LeRobot CLI options]"
        )
    mode, extra = sys.argv[1], sys.argv[2:]
    check = "--check" in extra
    extra = [arg for arg in extra if arg != "--check"]
    # The NumPy-1-compatible Rerun 0.22 names single scalar logs Scalar.
    import draccus
    import rerun as rr

    if not hasattr(rr, "Scalars"):
        rr.Scalars = rr.Scalar
    import os

    config_path = Path(os.environ.get("SO101_CONFIG", str(ROOT / "configs/so101.json")))
    config = json.loads(config_path.read_text())
    if mode == "teleop":
        from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig as Config
        from lerobot.scripts.lerobot_teleoperate import teleoperate as run
    else:
        from lerobot.scripts.lerobot_record import RecordConfig as Config
        from lerobot.scripts.lerobot_record import record as run

        config["dataset"] = {
            "fps": config.pop("fps"),
            "push_to_hub": False,
            "vcodec": "h264",
        }
    # Load device fields from a config file: upstream CLI parsing has conflicting
    # port field types across robot choices (serial path vs integer network port).
    import tempfile

    with tempfile.TemporaryDirectory(prefix="so101-config-") as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(config))
        cfg = draccus.parse(Config, config_path=str(path), args=extra)
    if check:
        print(
            f"PASS {mode} config: {type(cfg.robot).__name__}, {type(cfg.teleop).__name__}"
        )
        if mode == "record":
            print(
                f"Dataset: {cfg.dataset.repo_id}; push_to_hub={cfg.dataset.push_to_hub}"
            )
        return
    run(cfg)


if __name__ == "__main__":
    main()
