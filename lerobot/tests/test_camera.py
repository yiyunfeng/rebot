import numpy as np
import pytest

from lerobot_robot_rebot.camera import ReBotRGBDCamera, ReBotRGBDConfig, depth_mm_to_model_image


def test_depth_image_uses_fixed_metric_range() -> None:
    depth_mm = np.array([[0, 99, 100, 550, 1000, 1001]], dtype=np.uint16)

    image = depth_mm_to_model_image(depth_mm, min_depth_mm=100, max_depth_mm=1000)

    assert image.shape == (1, 6, 3)
    np.testing.assert_array_equal(
        image[..., 0],
        np.array([[0, 0, 1, 128, 255, 0]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(image[..., 0], image[..., 1])
    np.testing.assert_array_equal(image[..., 1], image[..., 2])


@pytest.mark.parametrize("shape", [(2,), (1, 2, 1)])
def test_depth_image_rejects_non_image_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="HxW"):
        depth_mm_to_model_image(np.zeros(shape, dtype=np.uint16), 100, 1000)


def test_camera_config_rejects_invalid_depth_range() -> None:
    with pytest.raises(ValueError, match="深度范围"):
        ReBotRGBDConfig(min_depth_mm=1000, max_depth_mm=1000)


def test_camera_retries_incomplete_rgbd_frames() -> None:
    color_bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    depth_mm = np.full((2, 3), 500, dtype=np.uint16)

    class FakeDriver:
        def __init__(self) -> None:
            self.calls = 0

        def get_frame(self):
            self.calls += 1
            if self.calls < 3:
                return None, None
            return color_bgr, depth_mm

    camera = ReBotRGBDCamera(ReBotRGBDConfig(width=3, height=2, frame_retries=3))
    camera._driver = FakeDriver()

    rgb, model_depth = camera.read()

    assert camera._driver.calls == 3
    assert rgb.shape == (2, 3, 3)
    assert model_depth.shape == (2, 3, 3)
    np.testing.assert_array_equal(camera.read_depth_mm(), depth_mm)


def test_camera_times_out_after_configured_retries() -> None:
    class EmptyDriver:
        def __init__(self) -> None:
            self.calls = 0

        def get_frame(self):
            self.calls += 1
            return None, None

    camera = ReBotRGBDCamera(ReBotRGBDConfig(frame_retries=2))
    camera._driver = EmptyDriver()

    with pytest.raises(TimeoutError, match="连续 2 次"):
        camera.read()

    assert camera._driver.calls == 2
