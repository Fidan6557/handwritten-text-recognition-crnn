import cv2
import numpy as np
import pytest

from src.data.preprocessing import preprocess_handwritten_image


def _make_test_image(path, width=100, height=50):
    img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_output_shape(tmp_path):
    img_path = tmp_path / "test.png"
    _make_test_image(img_path)

    result = preprocess_handwritten_image(str(img_path), image_width=512, image_height=64)

    assert result.shape == (64, 512)


def test_pixel_values_in_zero_one_range(tmp_path):
    img_path = tmp_path / "test.png"
    _make_test_image(img_path)

    result = preprocess_handwritten_image(str(img_path))

    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        preprocess_handwritten_image("nonexistent_image.png")


def test_custom_dimensions(tmp_path):
    img_path = tmp_path / "test.png"
    _make_test_image(img_path)

    result = preprocess_handwritten_image(str(img_path), image_width=256, image_height=32)

    assert result.shape == (32, 256)
