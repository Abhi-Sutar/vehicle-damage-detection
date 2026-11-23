"""Image preprocessing before inference."""

import cv2
import numpy as np

CLIP_LIMIT = 2.0
TILE_GRID = (8, 8)


def preprocess(image_bgr: np.ndarray, clahe: bool = True) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("expected a uint8 BGR image shaped (height, width, 3)")
    if not clahe:
        return image_bgr.copy()
    lightness, a_channel, b_channel = cv2.split(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB))
    equalizer = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
    merged = cv2.merge((equalizer.apply(lightness), a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
