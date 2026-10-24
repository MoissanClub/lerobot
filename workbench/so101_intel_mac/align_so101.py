"""Start the camera-free SO101 alignment/inspection check using this lab's calibration."""

import sys
from pathlib import Path
from lerobot.scripts.lerobot_align_so101 import main

if __name__ == "__main__":
    if not any(
        arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]
    ):
        sys.argv.extend(
            [
                "--config",
                str(
                    Path(__file__).resolve().parent
                    / "outputs/lab/so101-20261023/hardware.json"
                ),
            ]
        )
    main()
