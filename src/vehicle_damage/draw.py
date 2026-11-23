"""OpenCV drawing helpers."""

from typing import Any

import cv2
import numpy as np

from vehicle_damage.masks import area_pct, mask_to_original

PALETTE = [(60, 60, 255), (0, 165, 255), (0, 220, 0), (255, 120, 0)]


def color_for(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]


def put_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    origin = (x, max(y - 4, 14))
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_polygons(
    image: np.ndarray,
    polygons: list[tuple[int, np.ndarray]],
    names: dict[int, str],
    alpha: float = 0.35,
) -> np.ndarray:
    fill = image.copy()
    for class_id, points in polygons:
        cv2.fillPoly(fill, [points], color_for(class_id))
    out = cv2.addWeighted(fill, alpha, image, 1 - alpha, 0)
    for class_id, points in polygons:
        color = color_for(class_id)
        cv2.polylines(out, [points], isClosed=True, color=color, thickness=2)
        x, y = points.min(axis=0)
        put_label(out, names.get(class_id, str(class_id)), int(x), int(y), color)
    return out


def annotate(
    image_bgr: np.ndarray, result: Any, alpha: float = 0.35
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if result.masks is None:
        return image_bgr.copy(), []
    height, width = image_bgr.shape[:2]
    class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
    confidences = result.boxes.conf.cpu().numpy()
    raw_masks = result.masks.data.cpu().numpy()
    masks = [mask_to_original(m, height, width) for m in raw_masks]

    fill = image_bgr.copy()
    for mask, class_id in zip(masks, class_ids, strict=True):
        fill[mask > 0] = color_for(class_id)
    out = cv2.addWeighted(fill, alpha, image_bgr, 1 - alpha, 0)

    detections = []
    for mask, class_id, confidence in zip(masks, class_ids, confidences, strict=True):
        name = result.names[class_id]
        detections.append(
            {
                "class": name,
                "confidence": round(float(confidence), 4),
                "area_pct": round(area_pct(mask), 2),
            }
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        color = color_for(class_id)
        cv2.drawContours(out, contours, -1, color, 2)
        x, y = np.vstack(contours).reshape(-1, 2).min(axis=0)
        put_label(out, f"{name} {confidence:.2f}", int(x), int(y), color)
    return out, detections
