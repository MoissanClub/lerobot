"""Interactive recalibration of one arm; prompts require operator positioning."""

import json
import sys
from pathlib import Path
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig

root = Path(__file__).resolve().parent
config = json.loads((root / "hardware.json").read_text())
arm = sys.argv[1]
if arm == "leader":
    options = config["teleop"].copy()
    options.pop("type")
    options["calibration_dir"] = Path(options["calibration_dir"])
    cfg = CalibrateConfig(teleop=SO101LeaderConfig(**options))
elif arm == "follower":
    options = config["robot"].copy()
    options.pop("type")
    options["calibration_dir"] = Path(options["calibration_dir"])
    cfg = CalibrateConfig(robot=SO101FollowerConfig(**options))
else:
    raise SystemExit("Choose leader or follower")
calibrate(cfg)
