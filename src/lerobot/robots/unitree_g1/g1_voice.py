# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Best-effort robot-speaker feedback, isolated from the arm control loop."""

import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any


def make_audio_client() -> Any:
    """Reuse the DDS participant initialized by the arm feedback connection."""
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    client = AudioClient()
    client.SetTimeout(1.0)
    client.Init()
    return client


class _PromptHandler(logging.Handler):
    """Forward only operator prompts, never SDK errors or routine status logs."""

    def __init__(self, voice: "G1FollowingVoice"):
        super().__init__(logging.INFO)
        self.voice = voice
        self.last_guidance = -float("inf")

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.module not in ("g1_vr_processor", "g1_vr_front_box"):
            return
        now = time.monotonic()
        if message == "Robot start tracking in 5 second":
            self.last_guidance = now
            self.voice.say(message)
        elif message == "Lost track. Slow down.":
            if now - self.last_guidance >= 10.0:
                self.last_guidance = now
                self.voice.say(message)
        elif message.startswith("Tracking blocked:"):
            self.voice.say("Tracking blocked. Check the terminal.")
        elif message.startswith("Start to get picked up"):
            self.last_guidance = now
            self.voice.say("Start to get picked up. Move both hands near the robot wrists.")
        elif message.startswith(("Waiting for VR pickup", "VR workspace boundary")):
            if now - self.last_guidance >= 10.0:
                candidates = []
                for line in message.splitlines()[1:]:
                    if "move " not in line:
                        continue
                    hand = line.strip().split(" hand:", 1)[0]
                    for direction, distance in re.findall(
                        r"(forward|backward|left|right|up|down) ([0-9.]+) cm", line
                    ):
                        candidates.append((float(distance), hand, direction))
                if candidates:
                    _, hand, direction = max(candidates, key=lambda item: item[0])
                    self.last_guidance = now
                    self.voice.say(f"Move {hand.lower()} hand {direction}.")
        elif message.startswith("Waiting for tracking"):
            if now - self.last_guidance >= 10.0:
                self.last_guidance = now
                self.voice.say(
                    message.split(". No valid pose", 1)[0]
                    + ". Wake both controllers and keep them visible to the headset."
                )
        elif message.startswith("VR workspace regained"):
            self.voice.say("Hands back in range. Robot continues to follow.")
        elif message.startswith("XR tracking paused"):
            self.voice.say("Tracking lost. Following paused. Restore tracking and press R to pick up again.")
        elif message.startswith("IK failed"):
            self.voice.say("Arm target could not be solved. Following paused. Press R to pick up again.")
        elif message.startswith("VR pickup cancelled"):
            self.voice.say("Pickup cancelled. Following paused.")


class G1FollowingVoice:
    """Speak on following transitions without accumulating stale announcements."""

    def __init__(self, client_factory: Callable[[], Any] = make_audio_client):
        self._factory = client_factory
        self._following = False
        self._generation = 0
        self._pending: str | None = None
        self._prompt = False
        self._idle = threading.Event()
        self._idle.set()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="g1-following-voice", daemon=True)
        self._worker.start()
        self._handler = _PromptHandler(self)
        logging.getLogger().addHandler(self._handler)

    def update(self, following: bool) -> None:
        """Queue only a new following transition; a pause cancels pending speech."""
        with self._lock:
            if self._stop.is_set() or following == self._following:
                return
            self._following = following
            if following:
                self._queue("Robot starts to follow", prompt=False)
            elif not self._prompt:
                self._generation += 1
                self._pending = None

    def _queue(self, message: str, *, prompt: bool) -> None:
        self._generation += 1
        self._pending = message
        self._prompt = prompt
        self._idle.clear()
        self._wake.set()

    def say(self, message: str) -> None:
        """Replace pending guidance with the latest prompt; never block control."""
        with self._lock:
            if not self._stop.is_set():
                self._queue(message, prompt=True)

    def _run(self) -> None:
        client = None
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                if self._stop.is_set():
                    return
                if self._pending is None:
                    self._idle.set()
                    continue
                generation = self._generation
                message = self._pending
            try:
                if client is None:
                    client = self._factory()
                with self._lock:
                    if self._stop.is_set() or generation != self._generation:
                        continue
                code = client.TtsMaker(message, 1)
                if code != 0:
                    logging.warning("Robot voice announcement failed (SDK status %s)", code)
            except Exception:
                logging.exception("Robot voice unavailable; arm control is unaffected")
                client = None
            finally:
                with self._lock:
                    if generation == self._generation:
                        self._pending = None
                        self._idle.set()

    def close(self) -> None:
        """Cancel pending speech; call after motor authority has been released."""
        logging.getLogger().removeHandler(self._handler)
        # Motor resources have already closed. Give final failure feedback a
        # bounded chance to reach the speaker before cancelling the worker.
        self._idle.wait(timeout=1.5)
        self._stop.set()
        self._wake.set()
        self._worker.join(timeout=2.5)
        if self._worker.is_alive():
            logging.warning("Robot voice worker is still finishing an SDK call")
