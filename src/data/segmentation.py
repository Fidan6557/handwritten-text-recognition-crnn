from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image_as_grayscale(image):
    """
    Load image from path or PIL image and convert it to grayscale numpy array.
    """

    if isinstance(image, (str, Path)):
        grayscale = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)

        if grayscale is None:
            raise FileNotFoundError(f"Image not found: {image}")

        return grayscale

    image = np.array(image)

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image


def segment_text_lines(image, min_line_height=12, padding=8):
    """
    Segment a full handwritten page into individual text line crops.

    This function uses thresholding, dilation and contour detection.
    It is designed as a practical baseline for handwritten page OCR.

    Args:
        image: image path or PIL image
        min_line_height: minimum contour height to keep as a text line
        padding: extra pixels added around each detected line

    Returns:
        List of PIL images, one per detected text line.
    """

    grayscale = load_image_as_grayscale(image)

    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    line_boxes = []

    image_height, image_width = grayscale.shape

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if h < min_line_height:
            continue

        if w < image_width * 0.08:
            continue

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image_width)
        y2 = min(y + h + padding, image_height)

        line_boxes.append((x1, y1, x2, y2))

    line_boxes = sorted(line_boxes, key=lambda box: box[1])

    line_images = []

    for x1, y1, x2, y2 in line_boxes:
        crop = grayscale[y1:y2, x1:x2]
        crop = Image.fromarray(crop).convert("RGB")
        line_images.append(crop)

    return line_images


def save_segmented_lines(image, output_dir):
    """
    Segment a page image and save detected lines for debugging.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    line_images = segment_text_lines(image)

    saved_paths = []

    for index, line_image in enumerate(line_images, start=1):
        output_path = output_dir / f"line_{index:03d}.png"
        line_image.save(output_path)
        saved_paths.append(output_path)

    return saved_paths