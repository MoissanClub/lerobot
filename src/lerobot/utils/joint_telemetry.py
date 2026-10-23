"""Bounded background JSONL logging of already-read joint state and targets."""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JointTelemetryWriter:
    """Never wait for disk in the control loop; count any lost samples explicitly."""

    def __init__(self, path: str, metadata: dict[str, Any], capacity: int = 4096):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8")
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=capacity)
        self._closing = threading.Event()
        self.submitted = 0
        self.written = 0
        self.dropped = 0
        self.error: str | None = None
        self._metadata = metadata
        self._thread = threading.Thread(target=self._run, name="lerobot-joint-telemetry", daemon=True)
        self._thread.start()

    def submit(self, sample: dict[str, Any]) -> None:
        self.submitted += 1
        if self._closing.is_set() or self.error is not None:
            self.dropped += 1
            return
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            self.dropped += 1

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def _run(self) -> None:
        try:
            self._write({"type": "metadata", "schema_version": 1, **self._metadata})
            last_flush = time.monotonic()
            while not self._closing.is_set() or not self._queue.empty():
                try:
                    sample = self._queue.get(timeout=0.1)
                except queue.Empty:
                    self._file.flush()
                    continue
                self._write(sample)
                self.written += 1
                if time.monotonic() - last_flush >= 0.5:
                    self._file.flush()
                    last_flush = time.monotonic()
            self._write(
                {
                    "type": "summary",
                    "submitted": self.submitted,
                    "written": self.written,
                    "dropped": self.dropped,
                }
            )
        except Exception as exc:
            self.error = repr(exc)
            logger.exception("Joint telemetry writer failed; catch-up analysis may be incomplete")
        finally:
            self._file.close()

    def close(self) -> None:
        self._closing.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            logger.error("Joint telemetry did not finish flushing within five seconds")
        logger.info(
            "Joint telemetry: path=%s submitted=%s written=%s dropped=%s error=%s",
            self.path,
            self.submitted,
            self.written,
            self.dropped,
            self.error,
        )


def finite_joint_values(values: dict, keys: dict) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in keys:
        try:
            value = float(values[key])
            result[key] = value if math.isfinite(value) else None
        except (KeyError, TypeError, ValueError):
            result[key] = None
    return result
