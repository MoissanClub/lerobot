# G1-29 VR Teleoperation Examples

See the [G1 user guide](../../docs/source/unitree_g1.mdx#simulation-with-vr-isaacteleops)
for VR simulation setup and controls, and
[physical VR arm teleoperation](../../docs/source/unitree_g1.mdx#physical-vr-arm-teleoperation-isaac-teleop)
for camera viewing and hardware operation.
Simulation is the default and has no DDS participant. This workflow is separate
from the legacy whole-body server.

- `prepare_vr_assets.py`: pinned Hub simulator/IK assets and GR00T weights.
- `run_vr_teleop.py`: simulation, read-only shadow and explicitly gated physical modes.
- `validate_xr_readonly.py`: camera producer and headset-only display.
- `validate_arm_sdk.py`: passive feedback, reviewed hold and bounded joint diagnostics.

No G1-23 or hand actuation. Simulation, headset and physical acceptance are
separate gates. Never run physical commands solely because automated tests pass.
