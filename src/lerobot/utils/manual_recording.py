"""Terminal-local explicit episode boundaries; no global keyboard permissions."""

import logging
import os
import select
import sys
import termios
import tty


def apply_key(events, phase, key):
    if key == "q":
        events["stop_recording"] = True
        events["exit_early"] = True
    elif phase == "waiting" and key == "s":
        events["exit_early"] = True
    elif phase == "recording" and key in ("f", "d"):
        events["exit_early"] = True
        if key == "d":
            events["rerecord_episode"] = True


class ManualKeyboard:
    def __init__(self, events):
        self.events = events
        self.phase = "waiting"
        self.original = None

    def start(self):
        if not sys.stdin.isatty():
            raise ValueError("Manual recording requires an interactive Terminal (stdin must be a TTY)")
        self.original = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        return self

    def set_phase(self, phase):
        self.phase = phase
        self.events["exit_early"] = False
        self.events["rerecord_episode"] = False
        # Keys typed during saving must not silently start the next recording.
        if self.original is not None:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        logging.info("MANUAL %s | S=start F=finish/save D=discard Q=save-and-exit", phase.upper())

    def poll(self):
        while select.select([sys.stdin], [], [], 0)[0]:
            data = os.read(sys.stdin.fileno(), 64)
            if not data:
                raise EOFError("Recording control terminal closed")
            for key in data.decode("utf-8", errors="ignore").lower():
                apply_key(self.events, self.phase, key)

    def stop(self):
        if self.original is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.original)
            self.original = None


def manual_episodes(dataset, events, keyboard, run_phase, max_episodes, say):
    """Reuse the regular control loop/writer; waiting follows but saves no frames."""
    saved = 0
    while saved < max_episodes and not events["stop_recording"]:
        keyboard.set_phase("waiting")
        if not getattr(keyboard, "launcher_controlled", False):
            say(f"Ready for episode {dataset.num_episodes}. Press S to start")
        run_phase(None, keyboard.poll)
        if events["stop_recording"]:
            break
        keyboard.set_phase("recording")
        say(f"Recording episode {dataset.num_episodes}")
        run_phase(dataset, keyboard.poll)
        if events["rerecord_episode"]:
            if dataset.episode_buffer is not None:
                dataset.clear_episode_buffer()
            say("Discarded")
            continue
        if dataset.episode_buffer is None or dataset.episode_buffer["size"] == 0:
            say("No frames recorded")
            return saved
        keyboard.set_phase("saving")
        say("Saving")
        dataset.save_episode()
        saved += 1
        logging.info("SAVED episode %s (%s/%s this launch)", dataset.num_episodes - 1, saved, max_episodes)
    return saved


class FileKeyboard:
    """Private append-only command file owned by the demo launcher, no terminal hooks."""

    launcher_controlled = True

    def __init__(self, events, path):
        self.events = events
        self.path = path
        self.phase = "waiting"
        self.file = None

    def start(self):
        self.file = open(self.path)  # noqa: SIM115 - owned until stop()
        return self

    def set_phase(self, phase):
        self.phase = phase
        self.events["exit_early"] = False
        self.events["rerecord_episode"] = False
        logging.info("MANUAL %s (launcher controls)", phase.upper())

    def poll(self):
        while True:
            position = self.file.tell()
            line = self.file.readline()
            if not line:
                return
            if not line.endswith("\n"):
                self.file.seek(position)
                return
            key = line.strip().lower()
            apply_key(self.events, self.phase, key)
            if key == "d" and self.phase == "recording":
                self.events["stop_recording"] = True

    def stop(self):
        if self.file:
            self.file.close()
            self.file = None
