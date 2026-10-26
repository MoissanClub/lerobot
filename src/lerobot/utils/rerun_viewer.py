"""Find the viewer installed with the active Python environment."""

import os
import shutil
import sys
from pathlib import Path


def prepare_rerun_path():
    # Do not resolve the Python symlink: its venv/bin parent is what we need.
    directory = Path(sys.executable).parent
    viewer = directory / ("rerun.exe" if os.name == "nt" else "rerun")
    if viewer.is_file() and os.access(viewer, os.X_OK):
        paths = os.environ.get("PATH", "").split(os.pathsep)
        os.environ["PATH"] = os.pathsep.join([str(directory), *[p for p in paths if p != str(directory)]])
    return shutil.which("rerun")
