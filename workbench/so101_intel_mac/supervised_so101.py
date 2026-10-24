"""Run from the parent workbench: .venv/bin/python supervised_so101.py capture|check|run."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
config = root / "outputs/lab/so101-20261023/hardware.json"
rest = root / "outputs/lab/so101-20261023/rest-pose.json"
mode = sys.argv[1] if len(sys.argv) > 1 else "help"
base = [sys.executable, "-m", "lerobot.scripts.lerobot_so101_supervised", "--config", str(config)]
if mode == "capture":
    args = ["--capture-rest", str(rest)]
elif mode in ("check", "run"):
    args = ["--rest", str(rest)] + (["--check"] if mode == "check" else [])
else:
    print(__doc__)
    raise SystemExit(0 if mode == "help" else 2)
raise SystemExit(subprocess.call(base + args + sys.argv[2:]))
