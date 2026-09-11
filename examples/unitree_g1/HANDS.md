# Optional hands

`G1WithHands` composes an unchanged `UnitreeG1` body with zero, one, or two
`HandSystem` implementations. It is a LeRobot `Robot`, registered through
`G1WithHandsConfig` as `unitree_g1_with_hands`. This wrapper avoids putting hand
dependencies into the existing body driver. It works with either embodiment's
supported body backend and does not enable the blocked physical G1-23 backend.

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
PYTHONPATH=src python -m pytest tests/robots/test_unitree_g1_hands.py -q
```

This suite uses fake body/hands to test lifecycle, namespaces, independent dispatch,
invalid-action rejection before writes, failure cleanup, and no-hand compatibility.
Vendor SDK and physical acceptance are separate.
