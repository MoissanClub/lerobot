# SO101 demo launcher

One session contains the camera preview, every successful recorded episode, replay logs, attempted/discarded recordings and configuration provenance. Current episode always means the last successful committed episode. No physical rehearsal of this new launcher has been claimed; rehearse one short record/replay before the demo.

## Launch

From `/Users/dwei/lerobot`:

```sh
bash demo_so101.sh check
bash demo_so101.sh launch --label tomorrow_demo --episodes 27
```

27 allows two demonstration episodes followed by25 collection episodes. Use --episodes25 if25 is the desired total. Limit counts successful saves; failed/discarded attempts do not count. Ready and replay remain available when the recording limit is reached.

The launcher prints the session directory and exact resume command. Each new launch without --session makes a unique timestamp/UUID folder; no existing dataset is overwritten. Launching requires a focused interactive Terminal but no Accessibility permissions. Rerun viewer path and workspace HuggingFace cache are configured automatically.

## Ready screen — three choices

- **S — Start next episode.** Stops camera-only preview, stages a dataset checkpoint, connects arms and starts recording when ready. There is no recording-start timer while idle. Listen for “Recording episode N.” Ensure leader/follower start near matching poses; standard teleop has no startup alignment ramp.
- **R — Replay last saved episode.** Replays only committed data, with two live views and two recorded reference views. It asks you to place the follower near the saved starting pose before replay. No episode index advances; press R again to repeat it. Ctrl+C interrupts playback.
- **Q — Quit.** Closes idle camera preview. Data remains available for resume/replay.

Camera-only preview runs automatically at Ready. No arm is connected during that preview. Close other camera setup/teleop/record programs before launch; do not close the launcher’s Rerun viewer during an active recording.

## During recording

- **Right Arrow (or F): finish and save.** Returns to Ready after save validation. No task/gripper success detection.
- **Left Arrow: discard the active attempt and return to Ready to reset.** Then S starts again at the same next episode index. It does not discard a previously saved episode or automatically return the follower to its initial pose. This deliberately avoids an immediate motion restart during task reset.
- **Q: finish/save and then quit.** Empty attempts are not counted.
- **300 seconds / five minutes:** automatically finish and save. The operator normally finishes earlier with Right Arrow. Waiting at Ready never times out or starts recording automatically.

Keys are read by the launcher from its Terminal, not by pynput or the Rerun window. Keep Terminal focused. Keys other than Q are ignored while starting/saving, preventing accidental next starts. Q during save exits once that save completes.

Saving may take time for two camera videos; progress remains in the console/log. The regular recorder holds its last target until it finishes saving and disconnects; then follower torque is released. Support the arm for disconnect/reset. There is no automatic home, obstacle recovery or continuous arm holding across separate recording/replay processes. Idle camera-only preview restarts after cleanup.

## Resume and failure handling

```sh
bash demo_so101.sh list
bash demo_so101.sh launch --session EXACT_SESSION_DIRECTORY_NAME
```

Use the directory name printed by list. A fresh invocation without --session creates a different session; it does not silently resume another operator’s work. Session locking prevents two launchers writing the same session; the recording child inherits the lock so it remains protected if the parent dies.

Each attempt gets a unique path and records into a staged snapshot of the preceding completed dataset. On successful finalize, the launcher checks basic save integrity and promotes the staged dataset. If recording fails or is discarded, the prior committed snapshot remains current. On reopen, finalized orphan saves can be recovered; partial metadata/files are ignored. Corrupt latest checkpoints fall back to earlier valid ones. Failed attempts remain preserved for diagnosis; no automatic delete/recalibration.

Validation checks readable metadata, episode/action consistency and referenced video-file presence. It does **not** reject low FPS, timing jitter, missing expected samples, tracking error or an unsuccessful task. Data-quality review/interpolation is separate. The launcher cannot guarantee recovery from lost/corrupted storage or physically stop an arm after process/power loss.

## Files

```
outputs/demo/so101/UNIQUE_SESSION/
  session.json                 # count, current episode, calibration hashes
  base_record.json             # original hardware/camera config
  dataset -> attempts/.../data  # latest committed, ordinary LeRobot dataset
  attempts/
    ep000_UNIQUE/
      controls.txt
      record.json
      record.log
      joint_telemetry.jsonl
      data/                    # cumulative dataset snapshot
      committed.json           # present only after completed save
    ep001_UNIQUE/...
  logs/
    preview_*.log
    replay_ep000_*.json
    replay_ep000_*.log
    replay_ep000_*.result.json
```

The logical dataset contains episode indices0,1,2,... (one dataset, not separate per-episode repo IDs). Older checkpoint directories are retained for recovery. APFS copy-on-write clones avoid copying unchanged video blocks; other filesystems fall back to full copies, which may take time and space. Do not edit files through the dataset symlink while recording.

## Suggested demo sequence

1. Launch; show high-mount/gripper views at Ready.
2. S; demonstrate task; Right Arrow; wait for COMMITTED.
3. R; position follower near start; Enter; replay.
4. S; record a different episode; Right Arrow; R to replay it. Repeat R as desired.
5. Continue S/Right Arrow until the required number of successful episodes is collected. Left Arrow discards a bad active attempt; S restarts it after reset.
6. Q. Keep the printed session name for the next visit.
