#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_opencv.py::test_connect
# ```

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode, Cv2Backends, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

RealVideoCapture = cv2.VideoCapture


class MockLoopingVideoCapture:
    """
    Wraps the real OpenCV VideoCapture.
    Motivation: cv2.VideoCapture(file.png) is only valid for one read.
    Strategy: Read the file once & return the cached frame for subsequent reads.
    Consequence: No recurrent I/O operations, but we keep the test artifacts simple.
    """

    def __init__(self, *args, **kwargs):
        args_clean = [str(a) if isinstance(a, Path) else a for a in args]
        self._real_vc = RealVideoCapture(*args_clean, **kwargs)
        self._cached_frame = None

    def read(self):
        ret, frame = self._real_vc.read()

        if ret:
            self._cached_frame = frame
            return ret, frame

        if not ret and self._cached_frame is not None:
            return True, self._cached_frame.copy()

        return ret, frame

    def __getattr__(self, name):
        return getattr(self._real_vc, name)


@pytest.fixture(autouse=True)
def patch_opencv_videocapture():
    """
    Automatically patches cv2.VideoCapture for all tests.
    """
    module_path = OpenCVCamera.__module__
    target = f"{module_path}.cv2.VideoCapture"

    with patch(target, new=MockLoopingVideoCapture):
        yield


# NOTE(Steven): more tests + assertions?
TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"
DEFAULT_PNG_FILE_PATH = TEST_ARTIFACTS_DIR / "image_160x120.png"
TEST_IMAGE_SIZES = ["128x128", "160x120", "320x180", "480x270"]
TEST_IMAGE_PATHS = [TEST_ARTIFACTS_DIR / f"image_{size}.png" for size in TEST_IMAGE_SIZES]


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    config = OpenCVCameraConfig(index_or_path=0)

    _ = OpenCVCamera(config)


def test_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        assert camera.is_connected


@pytest.mark.parametrize("buffer_size, behavior", [(None, True), (1, True), (1, False), (1, "raise")])
def test_connect_configures_optional_buffer_size(buffer_size, behavior, caplog):
    class Capture:
        def __init__(self, *_args, **_kwargs):
            self.properties = []
            self.opened = True
            self.values = {
                cv2.CAP_PROP_FRAME_WIDTH: 160,
                cv2.CAP_PROP_FRAME_HEIGHT: 120,
                cv2.CAP_PROP_FPS: 30,
                cv2.CAP_PROP_BUFFERSIZE: 4,
            }

        def get(self, property_id):
            return self.values.get(property_id, 0)

        def isOpened(self):  # noqa: N802 - OpenCV API compatibility
            return self.opened

        def release(self):
            self.opened = False

        def set(self, property_id, value):
            self.properties.append((property_id, value))
            if property_id == cv2.CAP_PROP_BUFFERSIZE:
                if behavior == "raise":
                    raise RuntimeError("unsupported")
                if behavior:
                    self.values[property_id] = value
                return behavior
            self.values[property_id] = value
            return True

    module_path = OpenCVCamera.__module__
    with (
        patch(f"{module_path}.cv2.VideoCapture", new=Capture),
        patch.object(OpenCVCamera, "_start_read_thread"),
        caplog.at_level("DEBUG", logger=module_path),
    ):
        camera = OpenCVCamera(
            OpenCVCameraConfig(index_or_path=0, width=320, height=240, fps=25, buffer_size=buffer_size)
        )
        camera.connect(warmup=False)

    assert camera.videocapture is not None
    configured = [
        value
        for property_id, value in camera.videocapture.properties
        if property_id == cv2.CAP_PROP_BUFFERSIZE
    ]
    assert configured == ([] if buffer_size is None else [buffer_size])
    assert (cv2.CAP_PROP_FRAME_WIDTH, 320.0) in camera.videocapture.properties
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 240.0) in camera.videocapture.properties
    assert (cv2.CAP_PROP_FPS, 25.0) in camera.videocapture.properties
    if buffer_size is None:
        assert camera.buffer_size_set is camera.buffer_size_actual is None
    elif behavior == "raise":
        assert camera.buffer_size_set is False
        assert camera.buffer_size_actual is None
        assert "raised while requesting buffer_size=1" in caplog.text
    else:
        assert camera.buffer_size_set is behavior
        assert camera.buffer_size_actual == (1 if behavior else 4)
        assert f"success={behavior}" in caplog.text


@pytest.mark.parametrize("buffer_size", [0, -1, True, 1.0, "1"])
def test_buffer_size_configuration_rejects_invalid_values(buffer_size):
    with pytest.raises(ValueError, match="`buffer_size` must be a positive integer"):
        OpenCVCameraConfig(index_or_path=0, buffer_size=buffer_size)


def test_buffer_size_preserves_existing_positional_field_order():
    config = OpenCVCameraConfig(
        0,
        ColorMode.BGR,
        Cv2Rotation.ROTATE_180,
        2,
        "MJPG",
        Cv2Backends.V4L2,
    )

    assert config.backend is Cv2Backends.V4L2
    assert config.buffer_size is None


def test_connect_already_connected():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera, pytest.raises(DeviceAlreadyConnectedError):
        camera.connect()


def test_connect_invalid_camera_path():
    config = OpenCVCameraConfig(index_or_path="nonexistent/camera.png")

    camera = OpenCVCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


def test_invalid_width_connect():
    config = OpenCVCameraConfig(
        index_or_path=DEFAULT_PNG_FILE_PATH,
        width=99999,  # Invalid width to trigger error
        height=480,
    )

    camera = OpenCVCamera(config)
    with pytest.raises(RuntimeError):
        camera.connect(warmup=False)


def test_connect_cleans_up_after_settings_failure_and_allows_retry():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)
    camera = OpenCVCamera(config)
    opened_captures = []

    def fail_settings():
        opened_captures.append(camera.videocapture)
        raise RuntimeError("settings failed")

    with (
        patch.object(camera, "_configure_capture_settings", side_effect=fail_settings),
        pytest.raises(RuntimeError, match="settings failed"),
    ):
        camera.connect(warmup=False)

    assert camera.videocapture is None
    assert camera.thread is None
    assert not camera.is_connected
    assert opened_captures[0] is not None
    assert not opened_captures[0].isOpened()

    camera.connect(warmup=False)
    assert camera.is_connected
    camera.disconnect()


def test_connect_cleans_up_after_warmup_failure_and_allows_retry():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=1)
    camera = OpenCVCamera(config)
    read_threads = []

    def fail_warmup(*_args, **_kwargs):
        read_threads.append(camera.thread)
        raise TimeoutError("no frame")

    with (
        patch.object(camera, "async_read", side_effect=fail_warmup),
        pytest.raises(TimeoutError, match="no frame"),
    ):
        camera.connect()

    assert camera.videocapture is None
    assert camera.thread is None
    assert not camera.is_connected
    assert read_threads[0] is not None
    assert not read_threads[0].is_alive()

    camera.connect(warmup=False)
    assert camera.is_connected
    camera.disconnect()


def test_find_cameras_releases_unopened_handles():
    module_path = OpenCVCamera.__module__
    unopened_capture = MagicMock()
    unopened_capture.isOpened.return_value = False

    with (
        patch(f"{module_path}.platform.system", return_value="Darwin"),
        patch(f"{module_path}.MAX_OPENCV_INDEX", 1),
        patch(f"{module_path}.cv2.VideoCapture", return_value=unopened_capture),
    ):
        assert OpenCVCamera.find_cameras() == []

    unopened_capture.release.assert_called_once_with()


@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
def test_read(index_or_path):
    config = OpenCVCameraConfig(index_or_path=index_or_path, warmup_s=0)

    with OpenCVCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
def test_color_mode_conversion(index_or_path):
    """RGB and BGR reads of the same frame must differ only by a channel-axis reversal."""
    rgb_config = OpenCVCameraConfig(index_or_path=index_or_path, color_mode=ColorMode.RGB, warmup_s=0)
    bgr_config = OpenCVCameraConfig(index_or_path=index_or_path, color_mode=ColorMode.BGR, warmup_s=0)
    with OpenCVCamera(rgb_config) as rgb_cam:
        rgb = rgb_cam.read()
    with OpenCVCamera(bgr_config) as bgr_cam:
        bgr = bgr_cam.read()

    assert rgb.shape == bgr.shape
    np.testing.assert_array_equal(rgb, bgr[..., ::-1])


def test_postprocess_invalid_color_mode():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)
    camera.color_mode = "invalid"
    with pytest.raises(ValueError):
        camera._postprocess_image(np.zeros((120, 160, 3), dtype=np.uint8))


def test_read_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)

    camera = OpenCVCamera(config)
    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


def test_disconnect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.disconnect()


@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
def test_async_read(index_or_path):
    config = OpenCVCameraConfig(index_or_path=index_or_path, warmup_s=0)

    with OpenCVCamera(config) as camera:
        img = camera.async_read()

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(img, np.ndarray)


@pytest.mark.skip("Skipping test: async_read  0 timeout behavior may be flaky/non-deterministic.")
def test_async_read_timeout():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera, pytest.raises(TimeoutError):
        camera.async_read(timeout_ms=0)  # consumes any available frame by then
        camera.async_read(timeout_ms=0)  # request immediately another one


def test_async_read_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


def test_read_latest():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        # ensure at least one fresh frame is captured
        frame = camera.read()
        latest = camera.read_latest()

        assert isinstance(latest, np.ndarray)
        assert latest.shape == frame.shape


def test_read_latest_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)

    camera = OpenCVCamera(config)
    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read_latest()


def test_read_latest_high_frequency():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        # prime to ensure frames are available
        ref = camera.read()

        for _ in range(20):
            latest = camera.read_latest()
            assert isinstance(latest, np.ndarray)
            assert latest.shape == ref.shape


def test_read_latest_too_old():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        # prime to ensure frames are available
        _ = camera.read()

        with pytest.raises(TimeoutError):
            _ = camera.read_latest(max_age_ms=0)  # immediately too old


def test_fourcc_configuration():
    """Test FourCC configuration validation and application."""

    # Test MJPG specifically (main use case)
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc="MJPG")
    camera = OpenCVCamera(config)
    assert camera.config.fourcc == "MJPG"

    # Test a few other common formats
    valid_fourcc_codes = ["YUYV", "YUY2", "RGB3"]

    for fourcc in valid_fourcc_codes:
        config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc=fourcc)
        camera = OpenCVCamera(config)
        assert camera.config.fourcc == fourcc

    # Test invalid FOURCC codes
    invalid_fourcc_codes = ["ABC", "ABCDE", ""]

    for fourcc in invalid_fourcc_codes:
        with pytest.raises(ValueError):
            OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc=fourcc)


def test_fourcc_with_camera():
    """Test FourCC functionality with actual camera connection."""
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc="MJPG", warmup_s=0)

    # Connect should work with MJPG specified
    with OpenCVCamera(config) as camera:
        assert camera.is_connected

        # Read should work normally
        img = camera.read()
        assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
@pytest.mark.parametrize(
    "rotation",
    [
        Cv2Rotation.NO_ROTATION,
        Cv2Rotation.ROTATE_90,
        Cv2Rotation.ROTATE_180,
        Cv2Rotation.ROTATE_270,
    ],
    ids=["no_rot", "rot90", "rot180", "rot270"],
)
def test_rotation(rotation, index_or_path):
    filename = Path(index_or_path).name
    dimensions = filename.split("_")[-1].split(".")[0]  # Assumes filenames format (_wxh.png)
    original_width, original_height = map(int, dimensions.split("x"))

    config = OpenCVCameraConfig(index_or_path=index_or_path, rotation=rotation, warmup_s=0)
    with OpenCVCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)

        if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
            assert camera.width == original_height
            assert camera.height == original_width
            assert img.shape[:2] == (original_width, original_height)
        else:
            assert camera.width == original_width
            assert camera.height == original_height
            assert img.shape[:2] == (original_height, original_width)
