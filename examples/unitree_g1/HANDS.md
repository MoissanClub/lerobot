# Optional hands

`UnitreeG1Config.hands` composes zero, one, or two `HandSystem` implementations
under the existing `unitree_g1` robot identity. Vendor SDKs remain optional and
are loaded only by their drivers. No second body publisher is created.
The old `G1WithHands` / `unitree_g1_with_hands` wrapper is retained for compatibility,
not recommended for new integrations. Do not configure hands in both layers.

Physical hand drivers cannot be attached to a simulated body. Articulated-hand
simulation needs its own model and execution implementation; fake hand tests do
not provide one. This branch does not enable the blocked physical G1-23 backend.
It depends on `g1/simulation` to verify the shared robot lifecycle in both modes.

Body features retain their names. Hand-local features are prefixed with
`hands.left.` or `hands.right.`. Hand actions use each implementation's declared
features; normalized position conventions belong to that implementation.

`HandConfig` is a draccus choice registry. A vendor provides a registered config
and `HandSystem` with side-effect-free action validation, action/observation
features, connect/disconnect, send_action and get_observation. The standard
device-class factory resolves the implementation from its config.

All hand commands are validated before writes. Dispatch is sequential and cannot
be atomic across devices. A read/write failure disconnects all children and raises;
it does not imply emergency stop, torque disable, or rollback of physical motion.
Failed connections clean up every child; teardown attempts all children even if
one fails. Reconnect is explicit. Optional hands are not controlled through XR
implicitly; policies and other teleoperators can use the same action namespace.

```bash
PYTHONPATH=src python -m pytest tests/robots/test_unitree_g1_hands.py tests/robots/test_unitree_g1.py -q
```

This suite uses fake body/hands to test lifecycle, namespaces, independent dispatch,
invalid-action rejection before writes, failure cleanup, and no-hand compatibility.
Vendor SDK and physical acceptance are separate.
