import os

from lerobot.utils import rerun_viewer


def test_viewer_beside_python_found_without_activation(tmp_path, monkeypatch):
    bindir = tmp_path / "venv/bin"
    bindir.mkdir(parents=True)
    viewer = bindir / "rerun"
    viewer.write_text("#!/bin/sh\nexit 0\n")
    viewer.chmod(0o755)
    monkeypatch.setattr(rerun_viewer.sys, "executable", str(bindir / "python"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert rerun_viewer.prepare_rerun_path() == str(viewer)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bindir)
    rerun_viewer.prepare_rerun_path()
    assert os.environ["PATH"].split(os.pathsep).count(str(bindir)) == 1


def test_missing_neighbor_leaves_existing_path_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(rerun_viewer.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    rerun_viewer.prepare_rerun_path()
    assert os.environ["PATH"] == "/usr/bin:/bin"
