import numpy as np
import pytest

from src.data.preprocessing import preprocess_handwritten_image


def test_preprocessing_returns_float_array(tmp_path):
    import cv2

    image_path = tmp_path / "test_image.png"
    dummy = np.ones((64, 512), dtype=np.uint8) * 200
    cv2.imwrite(str(image_path), dummy)

    result = preprocess_handwritten_image(str(image_path))

    assert result.dtype == np.float32
    assert result.shape == (64, 512)


def test_preprocessing_normalizes_between_0_and_1(tmp_path):
    import cv2

    image_path = tmp_path / "test_image.png"
    dummy = np.ones((64, 512), dtype=np.uint8) * 128
    cv2.imwrite(str(image_path), dummy)

    result = preprocess_handwritten_image(str(image_path))

    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocessing_inverts_pixel_values(tmp_path):
    import cv2

    image_path = tmp_path / "white_image.png"
    white = np.ones((64, 512), dtype=np.uint8) * 255
    cv2.imwrite(str(image_path), white)

    result = preprocess_handwritten_image(str(image_path))

    assert np.allclose(result, 0.0)


def test_preprocessing_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        preprocess_handwritten_image("nonexistent_path.png")


def test_preprocessing_custom_size(tmp_path):
    import cv2

    image_path = tmp_path / "test_image.png"
    dummy = np.ones((32, 256), dtype=np.uint8) * 100
    cv2.imwrite(str(image_path), dummy)

    result = preprocess_handwritten_image(str(image_path), image_width=256, image_height=32)

    assert result.shape == (32, 256)
