import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.utils.demo_session import DemoSession, atomic_json, clone_or_copy


def source_config(tmp_path):
    cal = tmp_path / "cal"
    cal.mkdir(exist_ok=True)
    (cal / "arm.json").write_text("{}")
    return {
        "robot": {"id": "arm", "calibration_dir": str(cal), "cameras": {}},
        "teleop": {"id": "arm", "calibration_dir": str(cal)},
        "dataset": {"fps": 30, "video": False},
    }


def dataset(root, episodes, frames_per=2):
    (root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    atomic_json(
        root / "meta/info.json",
        {"total_episodes": episodes, "total_frames": episodes * frames_per, "fps": 30},
    )
    pq.write_table(
        pa.table({"episode_index": list(range(episodes))}), root / "meta/episodes/chunk-000/file-000.parquet"
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [i for i in range(episodes) for _ in range(frames_per)],
                "action": [[float(i)] for i in range(episodes * frames_per)],
            }
        ),
        root / "data/chunk-000/file-000.parquet",
    )


def test_stage_commit_failure_restart_replays_last_successful(tmp_path):
    s = DemoSession.create(tmp_path / "demo", source_config(tmp_path), 25)
    assert s.meta["current_episode"] is None
    first = s.stage()
    cfg = json.loads((first / "record.json").read_text())
    assert cfg["dataset"]["episode_time_s"] == 300 and not cfg["resume"]
    dataset(first / "data", 1)
    assert s.commit(first, 0)
    assert s.meta["current_episode"] == 0
    failed = s.stage()
    assert json.loads((failed / "record.json").read_text())["resume"]
    dataset(failed / "data", 2)
    assert not s.commit(failed, 1)
    assert s.meta["current_episode"] == 0
    # A damaged/stale session pointer cannot override the committed checkpoint.
    s.meta["current_episode"] = 99
    atomic_json(s.folder / "session.json", s.meta)
    s.close()
    s = DemoSession(tmp_path / "demo")
    assert s.meta["current_episode"] == 0 and (s.folder / "dataset").resolve() == (first / "data").resolve()
    second = s.stage()
    dataset(second / "data", 2)
    assert s.commit(second, 0)
    assert s.meta["current_episode"] == 1 and s.meta["committed_count"] == 2
    assert json.loads((first / "data/meta/info.json").read_text())["total_episodes"] == 1
    s.close()


def test_discard_does_not_touch_last_episode_and_no_path_reuse(tmp_path):
    s = DemoSession.create(tmp_path / "demo", source_config(tmp_path), 25)
    a = s.stage()
    b = s.stage()
    assert a != b
    dataset(a / "data", 1)
    assert s.commit(a, 0)
    c = s.stage()
    dataset(c / "data", 2)
    assert not s.commit(c, 0, discarded=True)
    assert s.meta["current_episode"] == 0
    s.close()


def test_session_lock_and_calibration_change(tmp_path):
    s = DemoSession.create(tmp_path / "demo", source_config(tmp_path), 25)
    with pytest.raises(BlockingIOError):
        DemoSession(tmp_path / "demo")
    (tmp_path / "cal/arm.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Calibration changed"):
        s.stage()
    s.close()


def test_cloned_files_remain_independent(tmp_path):
    a = tmp_path / "source"
    b = tmp_path / "copy"
    a.write_bytes(b"original")
    clone_or_copy(a, b)
    b.write_bytes(b"changed")
    assert a.read_bytes() == b"original"


def test_partial_save_is_not_promoted(tmp_path):
    s = DemoSession.create(tmp_path / "demo", source_config(tmp_path), 25)
    a = s.stage()
    dataset(a / "data", 1)
    atomic_json(a / "data/meta/info.json", {"total_episodes": 2, "total_frames": 4})
    with pytest.raises(ValueError):
        s.commit(a, 0)
    s.recover()
    assert s.meta["current_episode"] is None
    s.close()


def test_restart_recovers_save_completed_before_launcher_commit(tmp_path):
    s = DemoSession.create(tmp_path / "demo", source_config(tmp_path), 25)
    attempt = s.stage()
    dataset(attempt / "data", 1)
    s.close()  # Simulated launcher death after completed writer, before promotion.
    recovered = DemoSession(tmp_path / "demo")
    assert recovered.meta["current_episode"] == 0
    assert json.loads((attempt / "committed.json").read_text())["recovered"]
    recovered.close()
