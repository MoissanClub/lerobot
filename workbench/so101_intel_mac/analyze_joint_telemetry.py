"""Analyze an instrumented SO101 run: python analyze_joint_telemetry.py RUN.telemetry.jsonl."""

import argparse
import csv
import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from lerobot.utils.catch_up_analysis import project_catch_up


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--window-ms", type=float, default=100)
    parser.add_argument("--tolerance-deg", type=float, default=0.5)
    args = parser.parse_args()
    path = args.telemetry
    logfile = path.with_name(path.name.replace(".telemetry.jsonl", ".log"))
    announced = set()
    if logfile.exists():
        announced = {
            int(x) for x in re.findall(r"slow down event=(\d+)", logfile.read_text())
        }
    history = deque()
    events = []
    metadata = None
    summary = None
    with path.open() as stream:
        for line in stream:
            sample = json.loads(line)
            if sample["type"] == "metadata":
                metadata = sample
                if metadata["joint_units"] != "degrees":
                    raise SystemExit(
                        "This analyzer expects degree-based joint coordinates"
                    )
            elif sample["type"] == "summary":
                summary = sample
            elif sample["type"] == "sample":
                history.append(sample)
                while len(history) > 1 and sample["monotonic_s"] - history[0][
                    "monotonic_s"
                ] > max(1, args.window_ms / 1000):
                    history.popleft()
                if sample["warning_due"] or sample["sample_index"] in announced:
                    result = project_catch_up(
                        list(history),
                        tolerance=args.tolerance_deg,
                        velocity_window_s=args.window_ms / 1000,
                    )
                    result.update(
                        utc=datetime.fromtimestamp(
                            sample["wall_time_ns"] / 1e9, timezone.utc
                        ).isoformat(),
                        monotonic_s=sample["monotonic_s"],
                        notification_logged=sample["sample_index"] in announced,
                        worst_error_joint=sample["worst_joint"],
                        worst_error=sample["worst_error"],
                    )
                    events.append(result)
    report = {
        "telemetry": str(path),
        "metadata": metadata,
        "capture_summary": summary,
        "capture_complete_without_drops": summary is not None
        and summary["dropped"] == 0,
        "warning_requests": len(events),
        "logged_notifications": len(announced),
        "notifications_missing_samples": sorted(
            announced - {e["sample_index"] for e in events}
        ),
        "velocity_window_ms": args.window_ms,
        "position_tolerance_deg": args.tolerance_deg,
        "interpretation": "Frozen-current-leader-target, constant filtered follower speed; not measured recovery time or a collision detector.",
        "events": events,
    }
    output = path.with_name(path.name.replace(".telemetry.jsonl", ".catch-up.json"))
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    csvpath = output.with_suffix(".csv")
    with csvpath.open("w", newline="") as f:
        fields = [
            "sample_index",
            "utc",
            "notification_logged",
            "worst_error_joint",
            "worst_error",
            "arm_catch_up_time_s",
            "limiting_joint",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
    print(
        f"Saved {len(events)} warning requests, {len(announced)} logged notifications: {output}\n{csvpath}"
    )
    if not report["capture_complete_without_drops"]:
        print(
            "Capture is incomplete or dropped samples; inspect the report before interpreting estimates."
        )


if __name__ == "__main__":
    main()
