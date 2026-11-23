"""Draw the label polygons of random images from a YOLO folder and save them as one grid."""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from vehicle_damage.data import read_yolo_polygons
from vehicle_damage.draw import draw_polygons

TILE = 320


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="folder that holds the split folders")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data", type=Path, default=Path("data/data.yaml"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/label_check/grid.png"))
    args = parser.parse_args()

    names = yaml.safe_load(args.data.read_text(encoding="utf-8"))["names"]
    paths = sorted((args.root / args.split / "images").glob("*.jpg"))
    picked = random.Random(args.seed).sample(paths, args.count)

    tiles = []
    for path in picked:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        label = args.root / args.split / "labels" / f"{path.stem}.txt"
        drawn = draw_polygons(image, read_yolo_polygons(label, width, height), names)
        tiles.append(cv2.resize(drawn, (TILE, TILE), interpolation=cv2.INTER_AREA))

    while len(tiles) % args.cols:
        tiles.append(np.zeros((TILE, TILE, 3), dtype=np.uint8))
    rows = [cv2.hconcat(tiles[i : i + args.cols]) for i in range(0, len(tiles), args.cols)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.vconcat(rows))
    print(f"wrote {args.out} ({len(picked)} images from {args.split})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
