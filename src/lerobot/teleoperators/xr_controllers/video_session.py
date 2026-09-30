# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Isolated shared input/video OpenXR session with a bounded controller mailbox."""

import multiprocessing as mp
import queue
import time
import traceback
from contextlib import ExitStack, suppress

import numpy as np

from .camera_display import CameraDisplay
from .xr_controllers import IsaacControllerSession


def video_worker(config, video, frames, stop, ready, errors):
    try:
        from isaacteleop.oxr import OpenXRSessionHandles
        from isaacteleop.teleop_session_manager import get_required_oxr_extensions_from_pipeline

        with ExitStack() as cleanup:
            reader = IsaacControllerSession(config)
            display = CameraDisplay(video, get_required_oxr_extensions_from_pipeline(reader.pipeline))
            cleanup.callback(display.close)
            reader.oxr_handles = OpenXRSessionHandles(*display.session.get_oxr_handles())
            cleanup.callback(reader.close)
            reader.connect()
            ready.set()
            while not stop.is_set():
                started = time.monotonic()
                if display.session.should_close():
                    raise RuntimeError("XR session ended; disconnect and reconnect")
                display.update_camera()
                display.render()
                sample = reader.read()
                # A full mailbox drops frames instead of blocking.
                with suppress(queue.Full):
                    frames.put_nowait(sample)
                stop.wait(max(0, 1 / 90 - (time.monotonic() - started)))
    except BaseException:
        if not stop.is_set():
            errors.put(traceback.format_exc())
        ready.set()


class VideoControllerSession:
    """Session factory for XRControllers. No graphics work executes in read()."""

    def __init__(self, config, video, *, worker=video_worker, startup_timeout=30):
        self.config, self.video, self.worker = config, video, worker
        self.timeout = startup_timeout
        self.process = None
        self.snapshot = None

    def connect(self):
        if self.process is not None:
            raise RuntimeError("Already connected")
        context = mp.get_context("spawn")
        self.frames, self.errors = context.Queue(1), context.Queue(1)
        self.stop, ready = context.Event(), context.Event()
        self.snapshot = None
        self.process = context.Process(
            target=self.worker,
            args=(self.config, self.video, self.frames, self.stop, ready, self.errors),
            name="lerobot-xr-video",
        )
        try:
            self.process.start()
            if not ready.wait(self.timeout):
                raise TimeoutError("XR video startup timed out")
            self._check_error(0.1)
        except BaseException:
            self.close()
            raise

    def _check_error(self, wait=0):
        try:
            error = self.errors.get(timeout=wait) if wait else self.errors.get_nowait()
        except queue.Empty:
            if self.process is None or not self.process.is_alive():
                raise RuntimeError("XR video worker stopped; reconnect required") from None
        else:
            raise RuntimeError(error)

    def read(self):
        if self.process is None:
            raise RuntimeError("Not connected")
        self._check_error()
        with suppress(queue.Empty):
            self.snapshot = self.frames.get_nowait()
        # Preserve acquisition time: a cached frame must never appear newly tracked.
        if self.snapshot is None:
            action = {"captured_at": 0.0}
            for side in ("left", "right"):
                action.update(
                    {
                        f"{side}.{key}": value
                        for key, value in {
                            "tracked": False,
                            "grip_pos": np.zeros(3),
                            "grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                            "squeeze": 0.0,
                            "trigger": 0.0,
                        }.items()
                    }
                )
            return action
        return dict(self.snapshot)

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        self.stop.set()
        if process.pid is not None:
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(3)
            if process.is_alive():
                process.kill()
                process.join()
        for channel in (self.frames, self.errors):
            channel.cancel_join_thread()
            channel.close()
        self.snapshot = None
