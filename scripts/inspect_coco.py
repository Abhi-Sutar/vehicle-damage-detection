"""Summarise a COCO segmentation export: images, classes, polygon sizes and split overlap."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from vehicle_damage.data import COCO_SPLITS, source_name


def shape(segmentation: object) -> str:
    if isinstance(segmentation, dict):
        return "rle mask"
    if not segmentation:
        return "no segmentation"
    return f"polygon, {len(segmentation)} part(s)"


def summarise(split_dir: Path, splits_per_source: dict[str, set[str]]) -> None:
    coco = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = coco["images"]
    annotations = coco["annotations"]

    on_disk = {p.name for p in split_dir.glob("*.jpg")}
    missing = sum(1 for i in images if i["file_name"] not in on_disk)
    unlabelled = len({i["id"] for i in images} - {a["image_id"] for a in annotations})
    sizes = Counter(f"{i['width']}x{i['height']}" for i in images)
    print(f"{split_dir.name}: {len(images)} images, {missing} missing, {unlabelled} unlabelled")
    print(f"  sizes      {dict(sizes)}")
    print(f"  categories {names}")

    classes = Counter(names[a["category_id"]] for a in annotations)
    shapes = Counter(shape(a.get("segmentation")) for a in annotations)
    for label, count in [*sorted(classes.items()), *sorted(shapes.items())]:
        print(f"  {label:<22} {count}")

    points = sorted(
        sum(len(part) // 2 for part in a["segmentation"])
        for a in annotations
        if isinstance(a.get("segmentation"), list)
    )
    if points:
        print(f"  points     min {points[0]}, median {median(points):g}, max {points[-1]}")
        print(f"  <=4 points {sum(1 for n in points if n <= 4)}")

    for image in images:
        splits_per_source[source_name(image["file_name"])].add(split_dir.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="folder with the train, valid and test folders")
    args = parser.parse_args()

    splits_per_source: dict[str, set[str]] = defaultdict(set)
    for split in COCO_SPLITS:
        summarise(args.root / split, splits_per_source)

    distinct = len(splits_per_source)
    shared = sum(1 for found_in in splits_per_source.values() if len(found_in) > 1)
    print(f"source names: {distinct} distinct, {shared} found in more than one split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
