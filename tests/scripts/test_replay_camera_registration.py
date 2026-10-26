import json
import subprocess
import sys


def test_replay_camera_config_parses_in_fresh_process_without_record_import(tmp_path):
    config = {
        "robot": {
            "type": "so101_follower",
            "port": "/dev/fake",
            "id": "test",
            "cameras": {
                "camera_0": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
                "camera_1": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
            },
        },
        "dataset": {"repo_id": "local/test", "episode": 0},
        "display_data": True,
        "display_recorded_cameras": True,
    }
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(config))
    code = """import sys, draccus
from lerobot.scripts.lerobot_replay import ReplayConfig
cfg=draccus.parse(ReplayConfig, config_path=sys.argv[1], args=[])
assert len(cfg.robot.cameras)==2
assert cfg.robot.cameras['camera_0'].index_or_path==0
assert cfg.robot.cameras['camera_1'].index_or_path==1
print('PASS standalone replay camera registration; no hardware')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "PASS standalone" in result.stdout
