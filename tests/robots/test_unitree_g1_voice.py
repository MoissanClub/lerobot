# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Voice tests use fake clients; no DDS or audible speech."""

import threading
from unittest.mock import Mock

from lerobot.robots.unitree_g1.g1_voice import G1FollowingVoice


def test_voice_is_nonblocking_and_deduplicates_following_updates():
    entered, release = threading.Event(), threading.Event()

    def speak(*_):
        entered.set()
        assert release.wait(2)
        return 0

    client = Mock()
    client.TtsMaker.side_effect = speak
    voice = G1FollowingVoice(lambda: client)
    try:
        voice.update(False)
        client.TtsMaker.assert_not_called()
        voice.update(True)
        assert entered.wait(2)
        for _ in range(20):
            voice.update(True)
        client.TtsMaker.assert_called_once_with("Robot starts to follow", 1)
        # update has returned even though the SDK request is still blocked.
        voice.update(False)
    finally:
        release.set()
        voice.close()
    client.TtsMaker.assert_called_once()


def test_pause_cancels_speech_while_client_initializes():
    entered, release = threading.Event(), threading.Event()
    client = Mock()

    def factory():
        entered.set()
        assert release.wait(2)
        return client

    voice = G1FollowingVoice(factory)
    try:
        voice.update(True)
        assert entered.wait(2)
        voice.update(False)
    finally:
        release.set()
        voice.close()
    client.TtsMaker.assert_not_called()


def test_voice_error_does_not_escape_into_control_loop(caplog):
    called = threading.Event()

    def fail(*_):
        called.set()
        raise RuntimeError("audio service offline")

    client = Mock()
    client.TtsMaker.side_effect = fail
    voice = G1FollowingVoice(lambda: client)
    try:
        voice.update(True)
        assert called.wait(2)
    finally:
        voice.close()
    assert "Robot voice unavailable" in caplog.text


def test_prompt_survives_disabled_update_and_shutdown():
    client = Mock()
    client.TtsMaker.return_value = 0
    voice = G1FollowingVoice(lambda: client)
    voice.say("Arm activation blocked. Check torso posture.")
    voice.update(False)
    voice.close()
    client.TtsMaker.assert_called_once_with("Arm activation blocked. Check torso posture.", 1)


def test_operator_prompt_filter_and_guidance_throttle(monkeypatch):
    import logging

    from lerobot.robots.unitree_g1.g1_voice import _PromptHandler

    voice = Mock()
    handler = _PromptHandler(voice)
    now = [0.0]
    monkeypatch.setattr("lerobot.robots.unitree_g1.g1_voice.time.monotonic", lambda: now[0])

    def emit(message):
        handler.emit(logging.LogRecord("root", logging.INFO, "g1_vr_processor.py", 1, message, (), None))

    emit("Start to get picked up: waiting for both VR hands near the robot wrists.")
    voice.say.assert_called_once()
    guidance = (
        "Waiting for VR pickup\n  Left hand: move up 8 cm, forward 3 cm"
        "\n  Right hand: move backward 12 cm, down 2 cm"
    )
    emit(guidance)
    assert voice.say.call_count == 1
    now[0] = 9.9
    emit(guidance)
    assert voice.say.call_count == 1
    now[0] = 10.0
    emit(guidance)
    assert voice.say.call_count == 2
    voice.say.assert_called_with("Move right hand backward.")
    emit("VR pickup reached: requesting following without a target jump.")
    assert voice.say.call_count == 2  # Only confirmed activation announces following.
    emit("XR tracking paused: lost")
    assert "Tracking lost" in voice.say.call_args.args[0]
