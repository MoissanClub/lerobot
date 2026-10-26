"""One demo session, explicit episode keys, immutable checkpoints and replay logs."""

import argparse
import json
import os
import select
import subprocess
import sys
import termios
import threading
import tty
import uuid
from datetime import datetime
from pathlib import Path

from lerobot.utils.demo_session import DemoSession, atomic_json

WORKSPACE = Path(__file__).resolve().parents[4]


class Keys:
    def __init__(self):
        self.original = None
        self.buffer = ""

    def start(self):
        self.original = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        return self

    def stop(self):
        if self.original is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.original)
            self.original = None

    def read(self):
        result = []
        if select.select([sys.stdin], [], [], 0.1)[0]:
            data = os.read(sys.stdin.fileno(), 128)
            if not data:
                raise EOFError("Control terminal closed")
            self.buffer += data.decode(errors="ignore").lower()
        while self.buffer:
            if self.buffer.startswith("\x1b"):
                if self.buffer in ("\x1b", "\x1b["):
                    break
                if self.buffer.startswith("\x1b[c"):
                    result.append("f")
                    self.buffer = self.buffer[3:]
                    continue
                if self.buffer.startswith("\x1b[d"):
                    result.append("restart")
                    self.buffer = self.buffer[3:]
                    continue
                self.buffer = self.buffer[1:]
                continue
            key = self.buffer[0]
            self.buffer = self.buffer[1:]
            if key in "sfrq":
                result.append(key)
        return result

    def prompt(self, text):
        self.stop()
        try:
            return input(text)
        finally:
            self.start()


def token():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def menu(session):
    print(
        f"\nSESSION {session.folder.name} | saved={session.meta['committed_count']}/{session.meta['target_episodes']} | current episode={session.meta['current_episode']}",
        flush=True,
    )
    print("READY: S=Start next episode | R=Replay last saved | Q=Quit", flush=True)
    print(
        "RECORDING: Right Arrow/F=finish+save | Left Arrow=discard, then reset at Ready | Q=save-and-quit | maximum 5 minutes",
        flush=True,
    )


def send(attempt, key):
    with (attempt / "controls.txt").open("a") as f:
        f.write(key + "\n")
        f.flush()


def pipe_output(child, path, phase):
    with path.open("x") as log:
        for line in child.stdout:
            if "MANUAL RECORDING" in line:
                phase["value"] = "recording"
            if "MANUAL SAVING" in line:
                phase["value"] = "saving"
            print(line, end="", flush=True)
            log.write(line)
            log.flush()


def run_modal(command, log):
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        pipe_output(child, log, {})
        return child.wait()
    except KeyboardInterrupt:
        return child.wait()


def replay_episode(session, episode, keys):
    session.recover()
    if episode is None or not 0 <= episode < session.meta["committed_count"]:
        print("No such successfully saved episode.", flush=True)
        return
    config = json.loads((session.folder / "base_record.json").read_text())
    replay = {
        "robot": config["robot"],
        "dataset": {
            "repo_id": session.meta["repo_id"],
            "root": str(session.current_root),
            "episode": episode,
        },
        "play_sounds": True,
        "display_data": True,
        "display_recorded_cameras": True,
        "video_backend": "pyav",
    }
    name = "replay_ep" + str(episode).zfill(3) + "_" + token()
    path = session.folder / "logs" / (name + ".json")
    atomic_json(path, replay)
    keys.prompt(
        f"Replay episode {episode}: place follower near its starting pose, clear path, then Enter. Ctrl+C cancels: "
    )
    keys.stop()
    try:
        result = run_modal(
            [sys.executable, "-m", "lerobot.scripts.lerobot_replay", "--config_path=" + str(path)],
            session.folder / "logs" / (name + ".log"),
        )
    finally:
        keys.start()
    atomic_json(session.folder / "logs" / (name + ".result.json"), {"episode": episode, "exit_code": result})


def launch(session):
    if not sys.stdin.isatty():
        raise ValueError("Launch from an interactive Terminal")
    keys = Keys().start()
    child = attempt = reader = preview = preview_reader = None
    phase = {}
    quit_after = False

    def stop_preview():
        nonlocal preview, preview_reader
        if preview is not None:
            if preview.poll() is None:
                preview.terminate()
                try:
                    preview.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    preview.kill()
                    preview.wait()
            if preview_reader:
                preview_reader.join(timeout=2)
            preview = None

    def start_preview():
        nonlocal preview, preview_reader
        config = json.loads((session.folder / "base_record.json").read_text())
        cameras = list(config["robot"]["cameras"].values())
        command = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_camera_views",
            "--config-dir",
            str(session.folder),
            "--indices",
            *[str(c["index_or_path"]) for c in cameras],
            "--rotations",
            *[str(c.get("rotation", 0)) for c in cameras],
            "--preview-only",
            "--no-controls",
        ]
        preview = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        preview_reader = threading.Thread(
            target=pipe_output,
            args=(preview, session.folder / "logs" / ("preview_" + token() + ".log"), {}),
            daemon=True,
        )
        preview_reader.start()

    def start_recording():
        nonlocal attempt, phase, child, reader
        stop_preview()
        try:
            attempt = session.stage()
        except Exception as exc:
            print(f"Cannot start: {exc}", flush=True)
            start_preview()
            return
        print(
            f"Starting episode {session.meta['committed_count']}; wait for Recording, then Right Arrow finishes.",
            flush=True,
        )
        phase = {"value": "starting"}
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lerobot.scripts.lerobot_record",
                "--config_path=" + str(attempt / "record.json"),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            pass_fds=(session.lock.fileno(),),
        )
        reader = threading.Thread(
            target=pipe_output, args=(child, attempt / "record.log", phase), daemon=True
        )
        reader.start()

    try:
        menu(session)
        start_preview()
        while True:
            if child is not None and child.poll() is not None:
                result = child.wait()
                reader.join(timeout=5)
                try:
                    committed = session.commit(attempt, result, discarded=phase.get("discard", False))
                    print(
                        "COMMITTED"
                        if committed
                        else "Attempt not committed; last successful episode preserved.",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"Save validation failed: {exc}; previous committed episode remains current.",
                        flush=True,
                    )
                    session.recover()
                child = None
                if quit_after:
                    break
                menu(session)
                start_preview()
            for key in keys.read():
                if child is not None:
                    if key == "q":
                        quit_after = True
                        send(attempt, "q")
                    elif phase.get("value") == "recording":
                        if key == "f":
                            send(attempt, "f")
                            phase["value"] = "finishing"
                        elif key in ("restart", "r"):
                            phase["discard"] = True
                            send(attempt, "d")
                            phase["value"] = "discarding"
                    else:
                        print(f"{phase.get('value', 'starting')}: please wait; Q requests exit.", flush=True)
                elif key == "q":
                    return
                elif key == "s":
                    start_recording()
                elif key == "r":
                    stop_preview()
                    replay_episode(session, session.meta["current_episode"], keys)
                    menu(session)
                    start_preview()
    except KeyboardInterrupt:
        if child is not None:
            child.wait()
        print("Interrupted; last committed episode preserved.", flush=True)
    finally:
        stop_preview()
        keys.stop()
        session.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["launch", "list", "check"])
    p.add_argument("--config", type=Path, default=WORKSPACE / "outputs/lab/so101-20261025-12v/record.json")
    p.add_argument("--base", type=Path, default=WORKSPACE / "outputs/demo/so101")
    p.add_argument(
        "--session", help="Directory name or absolute path to reopen; no automatic creation if missing"
    )
    p.add_argument("--label", default="demo")
    p.add_argument("--episodes", type=int, default=25)
    args = p.parse_args()
    base = args.base.resolve()
    if args.mode == "list":
        for folder in sorted(base.glob("*"), reverse=True):
            if (folder / "session.json").exists():
                try:
                    session = DemoSession(folder)
                except BlockingIOError:
                    print(folder.name, "IN USE")
                    continue
                print(
                    folder.name,
                    "current=",
                    session.meta["current_episode"],
                    "saved=",
                    session.meta["committed_count"],
                )
                session.close()
        return
    if args.mode == "check":
        import draccus

        from lerobot.scripts.lerobot_record import RecordConfig

        cfg = draccus.parse(RecordConfig, config_path=str(args.config.resolve()), args=[])
        assert len(cfg.robot.cameras) == 2
        for d in (cfg.robot, cfg.teleop):
            if not (d.calibration_dir / (d.id + ".json")).is_file():
                raise ValueError("Missing calibration")
        print("PASS demo config/calibration; no hardware connected")
        return
    if args.session:
        folder = Path(args.session)
        folder = folder if folder.is_absolute() else base / folder
        session = DemoSession(folder)
    else:
        if args.episodes < 1:
            p.error("episodes must be positive")
        label = "".join(c for c in args.label if c.isalnum() or c in "_-") or "demo"
        session = DemoSession.create(
            base / (token() + "_" + label), json.loads(args.config.read_text()), args.episodes
        )
        session.meta["source_config"] = str(args.config.resolve())
        atomic_json(session.folder / "session.json", session.meta)
    print("Open session:", session.folder, flush=True)
    print(f"Resume this session: bash demo_so101.sh launch --session {session.folder.name}", flush=True)
    launch(session)


if __name__ == "__main__":
    main()
