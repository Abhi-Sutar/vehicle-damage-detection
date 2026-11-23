"""Mask helpers: undo the letterbox and measure area."""

import cv2
import numpy as np


def mask_to_original(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask_h, mask_w = mask.shape
    mask = mask.astype(np.float32)
    if (mask_h, mask_w) != (height, width):
        gain = min(mask_h / height, mask_w / width)
        pad_x = (mask_w - width * gain) / 2
        pad_y = (mask_h - height * gain) / 2
        top, bottom = round(pad_y - 0.1), mask_h - round(pad_y + 0.1)
        left, right = round(pad_x - 0.1), mask_w - round(pad_x + 0.1)
        mask = cv2.resize(
            mask[top:bottom, left:right], (width, height), interpolation=cv2.INTER_LINEAR
        )
    return (mask > 0.5).astype(np.uint8) * 255


def area_pct(mask: np.ndarray) -> float:
    return 100.0 * cv2.countNonZero(mask) / mask.size
