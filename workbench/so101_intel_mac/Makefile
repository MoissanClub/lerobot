PYTHON := $(CURDIR)/.venv/bin/python
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export HF_DATASETS_CACHE := $(CURDIR)/.cache/datasets

.PHONY: install-dev check test lint teleop
install-dev:
	.tools/uv pip install --python $(PYTHON) -r requirements-dev.in --override constraints-intel-mac.txt

check:
	$(PYTHON) verify_install.py
	$(PYTHON) teleop_so101_cameras.py --check
	$(PYTHON) teleop_pusht.py --check
	$(PYTHON) so101_dev.py record --help > /dev/null
	$(PYTHON) so101_dev.py teleop --check
	$(PYTHON) so101_dev.py record --check --dataset.repo_id=local/config_check --dataset.single_task="Configuration test"
	$(PYTHON) verify_recording.py

# Selected upstream tests use mocks and do not connect to physical hardware.
test:
	cd lerobot-src && $(PYTHON) -m pytest tests/robots/test_so100_follower.py tests/processor/test_policy_robot_bridge.py tests/scripts/test_edit_dataset_parsing.py -q --disable-warnings

lint:
	.venv/bin/ruff check so101_dev.py verify_recording.py
	cd lerobot-src && ../.venv/bin/ruff check src/lerobot/scripts/lerobot_teleoperate.py src/lerobot/scripts/lerobot_record.py src/lerobot/robots/so_follower src/lerobot/teleoperators/so_leader

teleop:
	$(PYTHON) so101_dev.py teleop
