#!/bin/bash
# Supervised lab actions. Run one phase at a time from your Terminal.
set -euo pipefail
LAB="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$LAB/../../.." && pwd)"
cd "$ROOT"
export SO101_CONFIG="$LAB/hardware.json"
export PYTHONUNBUFFERED=1
MODE="${1:-}"
case "$MODE" in
  calibrate-leader|calibrate-follower)
    COMMAND=("$ROOT/.venv/bin/python" "$LAB/calibrate_arm.py" "${MODE#calibrate-}")
    ;;
  teleop-text|teleop-voice|teleop-default|record)
    for ARM in leader follower; do
      if [[ ! -s "$LAB/calibration/$ARM/$ARM.json" ]]; then
        echo "Recalibrate $ARM first: bash $LAB/run.sh calibrate-$ARM"
        exit 1
      fi
    done
    case "$MODE" in
      teleop-text)
        OPTIONS=(teleop --fps=30 --teleop_time_s=30 --catch_up_warning.joint_error_threshold=2 --play_sounds=false) ;;
      teleop-voice)
        OPTIONS=(teleop --fps=30 --teleop_time_s=45 --catch_up_warning.joint_error_threshold=2 --play_sounds=true) ;;
      teleop-default)
        OPTIONS=(teleop --fps=30 --teleop_time_s=45 --catch_up_warning.joint_error_threshold=10 --play_sounds=true) ;;
      record)
        DATASET="so101_warning_$(date +%Y%m%d_%H%M%S)_$$"
        OPTIONS=(record "--dataset.repo_id=local/$DATASET" "--dataset.root=$LAB/$DATASET"
          '--dataset.single_task=diagnostic validation of SO101 tracking warning without cameras'
          --dataset.num_episodes=2 --dataset.episode_time_s=15 --dataset.reset_time_s=10
          --dataset.video=false --dataset.fps=30 --catch_up_warning.joint_error_threshold=2 --play_sounds=true) ;;
    esac
    COMMAND=("$ROOT/.venv/bin/python" "$ROOT/so101_dev.py" "${OPTIONS[@]}")
    ;;
  *) echo "Usage: bash $0 {calibrate-leader|calibrate-follower|teleop-text|teleop-voice|teleop-default|record}"; exit 2 ;;
esac
RUN="${MODE}_$(date +%Y%m%d_%H%M%S)_$$"
if [[ "$MODE" != calibrate-* ]]; then
  COMMAND+=("--catch_up_warning.telemetry_path=$LAB/$RUN.telemetry.jsonl")
fi
# Record the exact command and code revision before any hardware connection.
{
  date -u '+start_utc=%Y-%m-%dT%H:%M:%SZ'
  git -C "$ROOT/lerobot-src" rev-parse HEAD
  git -C "$ROOT/lerobot-src" status --short --branch
  printf 'command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  shasum -a 256 "$LAB/hardware.json" "$LAB/calibrate_arm.py" "$ROOT/so101_dev.py" "$ROOT/analyze_joint_telemetry.py" "$LAB/run.sh"
  for ARM in leader follower; do
    FILE="$LAB/calibration/$ARM/$ARM.json"
    if [[ -f "$FILE" ]]; then shasum -a 256 "$FILE"; fi
  done
} > "$LAB/$RUN.manifest.txt"
trap 'RESULT=$?; printf "exit_code=%s\n" "$RESULT" >> "$LAB/$RUN.manifest.txt"; date -u "+end_utc=%Y-%m-%dT%H:%M:%SZ" >> "$LAB/$RUN.manifest.txt"' EXIT
"${COMMAND[@]}" 2>&1 | tee "$LAB/$RUN.log"

# Analyze after motor cleanup. A postprocessing failure does not rewrite the
# hardware process result; it is recorded separately in the manifest.
if [[ "$MODE" != calibrate-* && -f "$LAB/$RUN.telemetry.jsonl" ]]; then
  if "$ROOT/.venv/bin/python" "$ROOT/analyze_joint_telemetry.py" "$LAB/$RUN.telemetry.jsonl"; then
    echo 'analysis_exit_code=0' >> "$LAB/$RUN.manifest.txt"
  else
    echo 'analysis_exit_code=1' >> "$LAB/$RUN.manifest.txt"
    echo "Postprocessing failed; preserve $LAB/$RUN.telemetry.jsonl for review."
  fi
fi
