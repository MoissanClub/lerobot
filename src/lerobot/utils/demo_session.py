"""Copy-on-write staged datasets and committed-episode recovery for a demo session."""

import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def clone_or_copy(source, dest):
    """APFS reflink on macOS; independent full copy on unsupported filesystems."""
    if os.uname().sysname == "Darwin":
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        clone = libc.clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(dest), 0) == 0:
            return str(dest)
    return shutil.copy2(source, dest)


def inspect_dataset(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    count = int(info["total_episodes"])
    frames = int(info["total_frames"])
    if count < 1 or frames < 1:
        return count, frames
    import pyarrow.parquet as pq

    rows = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    indices = {int(row["episode_index"]) for row in rows}
    if indices != set(range(count)):
        raise ValueError("Episode metadata indices do not match completed count")
    if not list((root / "data").rglob("*.parquet")):
        raise ValueError("No dataset action files")
    action_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in (root / "data").rglob("*.parquet"))
    if action_rows != frames:
        raise ValueError("Action frame count does not match committed metadata")
    for row in rows:
        for key, value in row.items():
            if key.startswith("videos/") and key.endswith("/file_index"):
                camera = key[len("videos/") : -len("/file_index")]
                video = root / info["video_path"].format(
                    video_key=camera,
                    chunk_index=int(row[f"videos/{camera}/chunk_index"]),
                    file_index=int(value),
                )
                if not video.is_file() or video.stat().st_size < 1:
                    raise ValueError(f"Missing video {video}")
    return count, frames


class DemoSession:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.lock = (self.folder / "session.lock").open("a+")
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.meta = json.loads((self.folder / "session.json").read_text())
        self.recover()

    def close(self):
        if self.lock:
            self.lock.close()
            self.lock = None

    @classmethod
    def create(cls, folder, record, episodes):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "attempts").mkdir()
        (folder / "logs").mkdir()
        atomic_json(folder / "base_record.json", record)
        hashes = {}
        for role in ("robot", "teleop"):
            d = record[role]
            file = Path(d["calibration_dir"]) / (d["id"] + ".json")
            hashes[role] = hashlib.sha256(file.read_bytes()).hexdigest()
        atomic_json(
            folder / "session.json",
            {
                "version": 1,
                "source_revision": subprocess.run(
                    ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "repo_id": "local/" + folder.name,
                "target_episodes": episodes,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "current_episode": None,
                "committed_count": 0,
                "calibration_sha256": hashes,
            },
        )
        return cls(folder)

    def recover(self):
        # A launcher can die after the recorder finalized but before promotion.
        # Recover completed saves from readable dataset metadata, never partial ones.
        for attempt in self.folder.glob("attempts/*"):
            if (attempt / "committed.json").exists() or not (attempt / "attempt.json").exists():
                continue
            try:
                state = json.loads((attempt / "attempt.json").read_text())
                if state.get("status") in ("failed", "discarded"):
                    continue
                count, frames = inspect_dataset(attempt / "data")
                if count == state["previous_count"] + 1:
                    atomic_json(
                        attempt / "committed.json",
                        {
                            "episodes": count,
                            "frames": frames,
                            "recovered": True,
                            "reason": "completed save found after interrupted launcher",
                        },
                    )
            except (ValueError, KeyError, OSError):
                continue
        valid = []
        for marker in self.folder.glob("attempts/*/committed.json"):
            try:
                entry = json.loads(marker.read_text())
                root = marker.parent / "data"
                count, frames = inspect_dataset(root)
                if count == entry["episodes"] and frames == entry["frames"]:
                    valid.append((count, marker.parent))
            except (ValueError, KeyError, OSError):
                continue
        count, attempt = max(valid, default=(0, None), key=lambda item: item[0])
        self.current_root = attempt / "data" if attempt else None
        self.meta.update(current_episode=count - 1 if count else None, committed_count=count)
        atomic_json(self.folder / "session.json", self.meta)
        if self.current_root:
            link = self.folder / "dataset"
            temp = self.folder / (".dataset-" + uuid.uuid4().hex)
            temp.symlink_to(self.current_root.relative_to(self.folder), target_is_directory=True)
            temp.replace(link)
        elif (self.folder / "dataset").is_symlink():
            (self.folder / "dataset").unlink()
        return self.meta["current_episode"]

    def stage(self):
        if self.meta["committed_count"] >= self.meta["target_episodes"]:
            raise ValueError("Target episode count reached")
        config = json.loads((self.folder / "base_record.json").read_text())
        # Calibration changes invalidate action meaning; do not append to an old dataset silently.
        for role, expected in self.meta["calibration_sha256"].items():
            d = config[role]
            actual = hashlib.sha256(
                (Path(d["calibration_dir"]) / (d["id"] + ".json")).read_bytes()
            ).hexdigest()
            if actual != expected:
                raise ValueError("Calibration changed; begin a new session")
        attempt = self.folder / "attempts" / f"ep{self.meta['committed_count']:03d}_{uuid.uuid4().hex[:10]}"
        attempt.mkdir()
        if self.current_root:
            shutil.copytree(self.current_root, attempt / "data", copy_function=clone_or_copy)
        config.update(
            manual_control=True,
            manual_control_file=str(attempt / "controls.txt"),
            resume=self.current_root is not None,
            display_data=True,
            display_compressed_images=True,
        )
        config["dataset"].update(
            episode_time_s=300,
            reset_time_s=0,
            repo_id=self.meta["repo_id"],
            root=str(attempt / "data"),
            num_episodes=1,
            push_to_hub=False,
        )
        config.setdefault("catch_up_warning", {})["telemetry_path"] = str(attempt / "joint_telemetry.jsonl")
        atomic_json(attempt / "record.json", config)
        (attempt / "controls.txt").write_text("s\n")
        atomic_json(
            attempt / "attempt.json", {"status": "prepared", "previous_count": self.meta["committed_count"]}
        )
        return attempt

    def commit(self, attempt, exit_code, discarded=False):
        if exit_code or discarded:
            atomic_json(
                attempt / "attempt.json",
                {"status": "discarded" if discarded else "failed", "exit_code": exit_code},
            )
            self.recover()
            return False
        count, frames = inspect_dataset(attempt / "data")
        if count != self.meta["committed_count"] + 1:
            raise ValueError("Expected exactly one additional completed episode")
        atomic_json(attempt / "committed.json", {"episodes": count, "frames": frames, "exit_code": exit_code})
        self.recover()
        return True
