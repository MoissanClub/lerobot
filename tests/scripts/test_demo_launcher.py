from lerobot.scripts.lerobot_so101_demo import Keys


def test_arrow_sequences_survive_split_reads(monkeypatch):
    monkeypatch.setattr("lerobot.scripts.lerobot_so101_demo.select.select", lambda *args: ([], [], []))
    keys = Keys()
    keys.buffer = "\x1b["
    assert keys.read() == []
    keys.buffer += "c"
    assert keys.read() == ["f"]
    keys.buffer = "\x1b[d"
    assert keys.read() == ["restart"]
    keys.buffer = "srq"
    assert keys.read() == ["s", "r", "q"]


def test_launcher_returns_to_last_saved_episode_after_next_record_failure(tmp_path, monkeypatch):
    import time

    import pyarrow as pa
    import pyarrow.parquet as pq

    from lerobot.scripts import lerobot_so101_demo as demo
    from lerobot.utils.demo_session import DemoSession, atomic_json

    cal = tmp_path / "cal"
    cal.mkdir()
    (cal / "arm.json").write_text("{}")
    config = {
        "robot": {"id": "arm", "calibration_dir": str(cal), "cameras": {}},
        "teleop": {"id": "arm", "calibration_dir": str(cal)},
        "dataset": {"fps": 30, "video": False},
    }
    session = DemoSession.create(tmp_path / "session", config, 25)
    children = []

    class Child:
        def __init__(self, command, **kwargs):
            self.record = "lerobot.scripts.lerobot_record" in command
            self.returncode = None
            self.stdout = iter(["MANUAL RECORDING\n"] if self.record else [])
            if self.record:
                self.attempt = __import__("pathlib").Path(command[-1].split("=", 1)[1]).parent
                if any(c.record for c in children):
                    self.returncode = 1
            children.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode or 0

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    class Keys:
        def __init__(self):
            self.commands = iter(["s", "f", "r", "s", "r", "q"])

        def start(self):
            return self

        def stop(self):
            pass

        def read(self):
            time.sleep(0.03)
            return [next(self.commands)]

    replayed = []

    def finish(attempt, key):
        assert key == "f"
        root = attempt / "data"
        (root / "meta/episodes").mkdir(parents=True)
        (root / "data").mkdir()
        atomic_json(root / "meta/info.json", {"total_episodes": 1, "total_frames": 2})
        pq.write_table(pa.table({"episode_index": [0]}), root / "meta/episodes/episodes.parquet")
        pq.write_table(pa.table({"episode_index": [0, 0]}), root / "data/frames.parquet")
        next(c for c in children if c.record and c.attempt == attempt).returncode = 0

    monkeypatch.setattr(demo, "Keys", Keys)
    monkeypatch.setattr(demo.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(demo.subprocess, "Popen", Child)
    monkeypatch.setattr(demo, "send", finish)
    monkeypatch.setattr(demo, "replay_episode", lambda s, episode, keys: replayed.append(episode))
    demo.launch(session)
    assert replayed == [0, 0]
    assert session.meta["current_episode"] == 0 and session.meta["committed_count"] == 1


def test_omitted_mode_defaults_to_launch_without_hardware(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lerobot.scripts import lerobot_so101_demo as demo

    config = tmp_path / "record.json"
    config.write_text("{}")
    launched = []

    def create(folder, record, episodes):
        folder.mkdir(parents=True)
        return SimpleNamespace(folder=folder, meta={})

    monkeypatch.setattr(demo.DemoSession, "create", create)
    monkeypatch.setattr(demo, "launch", lambda session: launched.append(session))
    monkeypatch.setattr(
        demo.sys, "argv", ["demo", "--config", str(config), "--base", str(tmp_path / "sessions")]
    )
    demo.main()
    assert len(launched) == 1
