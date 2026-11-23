"""Dataset helpers shared by the scripts and the tests."""

import random
from pathlib import Path

import numpy as np

COCO_SPLITS = ("train", "valid", "test")


def source_name(filename: str) -> str:
    return filename.split(".rf.")[0]


def read_yolo_polygons(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    polygons = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
        points = np.round(coords * (width, height)).astype(np.int32)
        polygons.append((int(parts[0]), points))
    return polygons


def polygon_to_line(class_id: int, flat: list[float], width: int, height: int) -> str | None:
    if len(flat) < 6:
        return None
    values = []
    for x, y in zip(flat[0::2], flat[1::2], strict=True):
        values.append(min(max(x / width, 0.0), 1.0))
        values.append(min(max(y / height, 0.0), 1.0))
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in values)


def assign_splits(
    group_sizes: dict[str, int], train: float = 0.70, val: float = 0.19, seed: int = 0
) -> dict[str, str]:
    names = sorted(group_sizes)
    random.Random(seed).shuffle(names)
    total = sum(group_sizes.values())
    assignment = {}
    seen = 0
    for name in names:
        share = seen / total
        if share < train:
            assignment[name] = "train"
        elif share < train + val:
            assignment[name] = "val"
        else:
            assignment[name] = "test"
        seen += group_sizes[name]
    return assignment
