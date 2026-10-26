# Camera view setup, recording and replay

Run camera-only view identification from the parent workspace:

```sh
.venv/bin/python -m lerobot.scripts.lerobot_camera_views \
  --config-dir outputs/lab/so101-20261025-12v --indices 0 1
```

No robot/motor code is imported and no arm is connected. Index0/1 are candidates, not calibrated/stable hardware IDs. Run from a camera-authorized interactive Terminal. If the indices are wrong, quit and rerun with the two desired indices (for example `--indices 1 2`); physical camera identity is established by covering each lens and inspecting the view. Device indices can change after reconnection.

A Rerun window displays two labeled views. Terminal commands, followed by Enter:

- `swap`: exchange camera_0/camera_1 assignments.
- `rotate 0` / `rotate 1`: rotate the selected view clockwise by 90°.
- `save`: save the chosen views/orientations and exit.
- `quit`: exit without changing config files.

This identifies viewpoints and orientation only. No lens intrinsics, distortion or camera-to-robot extrinsics are measured. Existing calibration files/motor configs are preserved. Existing record/replay JSONs are backed up. Saved mappings include actual frame dimensions and rotation; requested capture is 640x480 at30FPS, and unsupported reported FPS aborts. Rotated output dimensions are saved appropriately for LeRobot's capture convention.

Save updates both `record.json` and `replay.json`, with a fresh matching local dataset path, H264 video recording, preview enabled and no upload. `camera_views.json` records confirmation. Existing datasets are not overwritten. Recording uses full configured resolution. Replay shows live camera_0/camera_1 plus matching recorded references (JPEG thumbnails,320px wide) in one viewer. References are decoded before robot connection and capped at256MB, avoiding source-video decode delays inside the motor loop. Replay still sends stored actions at dataset FPS, does not guarantee identical physical motion and has no initial alignment ramp.

After camera-only confirmation, the existing commands work:

```sh
.venv/bin/lerobot-record --config_path=outputs/lab/so101-20261025-12v/record.json
.venv/bin/lerobot-replay --config_path=outputs/lab/so101-20261025-12v/replay.json
```

Replay options `display_data` and `display_recorded_cameras` default false outside the saved camera config; recorded references require display_data. Set display_recorded_cameras=false for live views only. Leader is unused for replay. The user should return follower close to the first recorded target before replay. These commands actuate hardware; the view setup command does not.

Intel compatibility: scalar logging uses the installed Rerun0.22 Scalar fallback; camera frames are time-varying rather than static. Synthetic dual-video recording/encoding/reload and four-view image/scalar logging have passed locally. Actual camera views/permissions and physical replay remain operator validation tasks.

Viewer discovery: setup/record/replay prepend the active Python environment’s bin directory before spawning Rerun. Direct `.venv/bin/python` or CLI use no longer requires shell activation for the viewer to be found. Installed viewer version0.22.1 verified locally; actual GUI remains an operator-side check.
