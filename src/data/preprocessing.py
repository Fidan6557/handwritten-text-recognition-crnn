import cv2
import numpy as np


def preprocess_handwritten_image(image_path, image_width=512, image_height=64):
    """
    Read and preprocess a handwritten text line image.

    Steps:
    - Convert to grayscale
    - Resize to fixed width and height
    - Normalize pixel values
    - Invert image so handwriting strokes become high-value pixels
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.resize(image, (image_width, image_height))
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image

    return image