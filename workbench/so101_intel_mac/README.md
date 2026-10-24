# Intel Mac SO101 workbench checkpoint

Snapshot of the working development helpers from 2026-10-23. To reproduce the workspace layout, copy this directory's contents into a workspace containing this repository as `lerobot-src/`; create the local environment using INSTALLATION.txt and DEVELOPMENT.txt. Do not run these helpers from this nested checkpoint directory. Historical lock files contain the original absolute editable path and require adjustment on another machine.

The session runner expects its private hardware.json and calibration files under outputs/lab/so101-20261023. These are preserved with the private labbook evidence archive, together with original logs and manifests. Verify ports, calibration, and operator readiness before any hardware command.

Implemented: angle-error warning, short speech, per-cycle telemetry/offline estimates, fixed-target alignment and read-only motor diagnostics. Planned: validated catch-up-time threshold, operator-controlled recording, pickup recovery, microphone input. Physical recording integration and pickup recovery remain unvalidated/unimplemented respectively.
