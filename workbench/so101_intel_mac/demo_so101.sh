#!/bin/bash
# Copy into parent workbench alongside .venv and lerobot-src.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/.cache/huggingface/datasets"
if [[ $# -eq 0 ]]; then set -- launch; fi
exec "$ROOT/.venv/bin/python" -m lerobot.scripts.lerobot_so101_demo "$@"
