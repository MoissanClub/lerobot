# G1-29 VR Validation Candidate

Branch: `work/g1-29-vr-teleop`. Start with the
[manual test plan](../../docs/source/g1_vr_manual_test_plan.mdx), not the legacy
whole-body server. Simulation is the default and has no DDS participant.

- `prepare_vr_assets.py`: pinned Hub simulator/IK assets and GR00T weights.
- `run_vr_teleop.py`: simulation, read-only shadow and explicitly gated physical modes.
- `validate_xr_readonly.py`: camera producer and headset-only display.
- `validate_arm_sdk.py`: passive feedback, reviewed hold and bounded joint diagnostics.

No G1-23 or hand actuation. Simulation, headset and physical acceptance are
separate gates. Never run physical commands solely because automated tests pass.
