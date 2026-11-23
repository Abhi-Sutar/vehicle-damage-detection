"""OpenCV drawing helpers."""

import cv2
import numpy as np

PALETTE = [(60, 60, 255), (0, 165, 255), (0, 220, 0), (255, 120, 0)]


def draw_polygons(
    image: np.ndarray,
    polygons: list[tuple[int, np.ndarray]],
    names: dict[int, str],
    alpha: float = 0.35,
) -> np.ndarray:
    fill = image.copy()
    for class_id, points in polygons:
        cv2.fillPoly(fill, [points], PALETTE[class_id % len(PALETTE)])
    out = cv2.addWeighted(fill, alpha, image, 1 - alpha, 0)
    for class_id, points in polygons:
        color = PALETTE[class_id % len(PALETTE)]
        cv2.polylines(out, [points], isClosed=True, color=color, thickness=2)
        x, y = points.min(axis=0)
        cv2.putText(
            out,
            names.get(class_id, str(class_id)),
            (int(x), max(int(y) - 4, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out
