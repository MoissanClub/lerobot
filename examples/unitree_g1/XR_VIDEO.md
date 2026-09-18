# Camera delivery

The generic Linux same-host RGB channel is `lerobot.cameras.frame_channel`.
It has no simulator, DDS, OpenXR, or gRPC dependency. A single producer publishes
bounded RGB frames with acquisition time and a session/sequence identity. Readers
reject stale frames and reopen the path after producer restart. Files are private
to the current user; this is not an authenticated network protocol.

`CameraDisplay` uploads frames to an Isaac Teleop Vulkan/CUDA quad layer.
`VideoControllerSession` supplies that graphics session's OpenXR handles to the
dual-controller pipeline. Both use **one OpenXR session**, in a spawned worker.
The IK loop reads a bounded mailbox and preserves acquisition timestamps.
Video/session failure is explicit; disconnect/reconnect and reset the clutch.
The mono monitor follows the head; it is not stereo depth or an HMD-driven camera.

With simulation + XR/video branches merged, pinned assets prepared, and the SDK
installed in the active environment:

```bash
# Automated control + camera production, no X/OpenXR/headset:
PYTHONPATH=src MUJOCO_GL=egl python examples/unitree_g1/run_xr_simulation.py \
  --assets /path/to/assets --embodiment g1_23 --headless --steps 100

# Manual headset acceptance after starting the external CloudXR runtime:
PYTHONPATH=src MUJOCO_GL=egl python examples/unitree_g1/run_xr_simulation.py \
  --assets /path/to/assets --embodiment g1_29 --video --steps 0
```

This example combines the local simulation loop with camera publication for easy
verification. Renderer work can reduce its control frequency; it is not the old
multiprocess DDS launcher. The XR display/input worker is isolated from that loop.

Automated suites:

- `tests/teleoperators/test_unitree_g1_xr_video.py`: frame validation, freshness,
  producer restart, worker teardown and reconnect.
- Set `G1_VIDEO_GPU_TESTS=1` to require real **offscreen** Vulkan/CUDA delivery,
  readback pixel checks, interruption/source-mismatch/recovery. Requires Isaac SDK
  and GPU access, no X window or headset.
- With combined stack: `G1_INTEGRATION_TESTS=1 G1_KINEMATICS_ASSETS=/path/to/assets
  MUJOCO_GL=egl PYTHONPATH=src python -m pytest
  tests/integration/test_unitree_g1_xr_video_mujoco.py -q`.

Manual headset acceptance remains required after merging. Headless tests cannot
verify the CloudXR client, headset rendering, or tracking quality.
