# G1-29 VR Teleoperation Examples

See the [G1 user guide](../../docs/source/unitree_g1.mdx#simulation-with-vr-isaacteleops)
for VR simulation setup and controls, and
[physical VR arm teleoperation](../../docs/source/unitree_g1.mdx#physical-vr-arm-teleoperation-isaac-teleop)
for camera viewing, the standard `lerobot-teleoperate` command, state transitions,
and fault recovery. Physical arm operation selects `--robot.arm_controller=unitree`;
it moves the arms during startup and enables VR following with `r` in the laptop
terminal. Simulation is the default and has no DDS participant.

- `prepare_vr_assets.py`: optional asset preparation and replay fixtures; the standard
  CLI resolves assets and selects the VR action processor automatically.
- `run_vr_teleop.py`: lower-level simulation, shadow and contract-based diagnostics;
  use the standard CLI in the guide for the current physical VR workflow.
- `validate_xr_readonly.py`: camera producer and headset-only display.
- `validate_arm_sdk.py`: passive feedback, reviewed hold and bounded joint diagnostics.

No G1-23 or hand actuation. Simulation, headset and physical acceptance are
separate gates. Never run physical commands solely because automated tests pass.
