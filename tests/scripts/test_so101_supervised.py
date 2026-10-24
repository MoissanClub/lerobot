import json

import pytest

from lerobot.scripts import lerobot_so101_supervised as runner


def test_replay_uses_observed_joints_and_records_decisions(tmp_path):
    cfg = runner.trial_config()
    cfg_values = {
        k: {j: v[j] for j in v if j != "gripper"} if isinstance(v, dict) else v for k, v in vars(cfg).items()
    }
    cfg = type(cfg)(**cfg_values)
    joints = cfg.alignment
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    with source.open("w") as f:
        f.write(json.dumps({"type": "metadata"}) + "\n")
        for k in range(50):
            f.write(
                json.dumps(
                    {
                        "type": "sample",
                        "sample_index": k,
                        "monotonic_s": k * 0.05,
                        "leader_target": {j + ".pos": 0 for j in joints},
                        "follower_position": {j + ".pos": 0 for j in joints},
                    }
                )
                + "\n"
            )
    runner.replay(source, output, dict.fromkeys(joints, 0), dict.fromkeys(joints, (-100, 100)), cfg)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["state"] == "lagging"
    assert records[-1]["state"] == "tracking"
    assert len(records) == 50
    with pytest.raises(FileExistsError):
        runner.replay(source, output, dict.fromkeys(joints, 0), dict.fromkeys(joints, (-100, 100)), cfg)


@pytest.mark.parametrize("serial_failure", [False, True])
def test_hardware_loop_with_fake_buses(monkeypatch, tmp_path, serial_failure):
    """Exercise the real runner: keyboard, read/send, telemetry and cleanup without USB."""
    joints = runner.JOINTS
    state = {j: 0.0 if j != "gripper" else 50.0 for j in joints}
    cal = {j: {"range_min": 0, "range_max": 4095} for j in joints}
    calibration_dir = tmp_path / "cal"
    calibration_dir.mkdir()
    for name in ("leader", "follower"):
        (calibration_dir / (name + ".json")).write_text(json.dumps(cal))
    config = tmp_path / "hardware.json"
    config.write_text(
        json.dumps(
            {
                "robot": {
                    "type": "so101_follower",
                    "port": "fake",
                    "id": "follower",
                    "calibration_dir": str(calibration_dir),
                },
                "teleop": {
                    "type": "so101_leader",
                    "port": "fake",
                    "id": "leader",
                    "calibration_dir": str(calibration_dir),
                },
            }
        )
    )
    rest = tmp_path / "rest.json"
    import hashlib

    rest.write_text(
        json.dumps(
            {
                "pose": state,
                "follower_calibration_sha256": hashlib.sha256(
                    (calibration_dir / "follower.json").read_bytes()
                ).hexdigest(),
            }
        )
    )
    instances = []

    class Bus:
        is_connected = False
        released = False

        def connect(self):
            self.is_connected = True

        def disconnect(self, disable_torque=True):
            self.is_connected = False
            self.released = disable_torque

        def sync_read(self, register, **kwargs):
            return dict.fromkeys(joints, 1 if register == "Torque_Enable" else 0)

    class Device:
        is_calibrated = True

        def __init__(self, config):
            self.bus = Bus()
            self.sent = []
            instances.append(self)

        def configure(self):
            pass

        def send_action(self, action):
            self.sent.append(action)
            return action

    class Keys:
        def __enter__(self):
            self.count = 0
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            self.count += 1
            return ["start"] if self.count == 2 else ["quit"] if self.count == 4 else []

    monkeypatch.setattr(runner, "SO101Follower", Device)
    monkeypatch.setattr(runner, "SO101Leader", Device)
    reads = 0

    def read(device):
        nonlocal reads
        reads += 1
        if serial_failure and reads == 6:
            raise OSError("simulated leader serial failure")
        return state.copy()

    monkeypatch.setattr(runner, "read_positions", read)

    def connect(device):
        device.bus.connect()
        return state.copy()

    monkeypatch.setattr(runner, "connect_follower_holding_present_pose", connect)
    monkeypatch.setattr(runner, "TerminalKeys", Keys)
    monkeypatch.setattr(runner.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *args: "")
    output = tmp_path / "live.jsonl"
    monkeypatch.setattr(
        runner.sys,
        "argv",
        ["trial", "--config", str(config), "--rest", str(rest), "--silent", "--output", str(output)],
    )
    if serial_failure:
        with pytest.raises(OSError, match="simulated leader"):
            runner.main()
    else:
        runner.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert any(r.get("state") == "lagging" for r in records)
    assert any(r.get("type") == "prompt_request" for r in records)
    assert records[-1]["dropped"] == 0
    if serial_failure:
        assert any(
            r.get("type") == "exception_hold" and r["result"] == "recent_measured_pose_sent" for r in records
        )
        assert any(r.get("type") == "fault" for r in records)
    assert instances[0].bus.released
    assert all(not d.bus.is_connected for d in instances)
