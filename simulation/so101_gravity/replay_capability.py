import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lerobot.utils.so101_capability_monitor import CapabilityMonitor, GravityModel

parser = argparse.ArgumentParser(description="Offline advisory monitor replay; no motor access")
parser.add_argument("--log", type=Path, required=True)
parser.add_argument("--calibration", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.log
r = [json.loads(line) for line in source.open()]
cal = json.loads(args.calibration.read_text())
g = GravityModel(Path(__file__).parent / "vendor/so101_new_calib.urdf", cal)
monitor = CapabilityMonitor(g, r[0]["bounds"])
latest = {}
reports = []
durations = []
last = -1e10
for record in r:
    if record["type"] == "motor_register":
        latest[record["register"]] = record
    if record["type"] != "sample" or record["monotonic_s"] - last < 0.1:
        continue
    t = record["monotonic_s"]
    cmd = record.get("sent_action") or {}
    cmd = {j: cmd.get(j + ".pos", record["follower"][j]) for j in record["follower"]}
    start = time.perf_counter()
    report = monitor.update(
        t,
        record["follower"],
        record["leader"],
        cmd,
        latest,
        active=record["state"] in ("tracking", "lagging"),
    )
    durations.append(time.perf_counter() - start)
    reports.append(report)
    last = t
out = args.output
out.mkdir(exist_ok=False, parents=True)
(out / "replay.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reports))
transition = next(x for x in r if x["type"] == "pre_alignment_motor_history")["transition_monotonic_s"]
before = max((x for x in reports if x["monotonic_s"] < transition), key=lambda x: x["monotonic_s"])
summary = {
    "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "monitor_samples": len(reports),
    "benchmark_ms": {
        "median": float(np.median(durations) * 1000),
        "p95": float(np.percentile(durations, 95) * 1000),
        "max": float(max(durations) * 1000),
    },
    "last_prediction_before_alignment": before["joints"]["shoulder_lift"],
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
