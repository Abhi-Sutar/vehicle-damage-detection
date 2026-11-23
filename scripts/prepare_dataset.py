"""Convert the Roboflow COCO export to YOLO-seg labels, with a split grouped by source image."""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from vehicle_damage.data import COCO_SPLITS, assign_splits, polygon_to_line, source_name


def load_records(coco_root: Path) -> tuple[list[dict], list[str]]:
    records = []
    class_names: list[str] = []
    for split in COCO_SPLITS:
        coco = json.loads((coco_root / split / "_annotations.coco.json").read_text("utf-8"))
        real = sorted((c for c in coco["categories"] if c["id"] != 0), key=lambda c: c["id"])
        class_ids = {c["id"]: index for index, c in enumerate(real)}
        names = [c["name"] for c in real]
        if class_names and names != class_names:
            raise SystemExit(f"categories differ in {split}: {names} vs {class_names}")
        class_names = names

        by_id = {i["id"]: i for i in coco["images"]}
        lines_per_image = defaultdict(list)
        for ann in coco["annotations"]:
            parts = ann["segmentation"]
            if not isinstance(parts, list):
                continue
            image = by_id[ann["image_id"]]
            for flat in parts:
                line = polygon_to_line(
                    class_ids[ann["category_id"]], flat, image["width"], image["height"]
                )
                if line:
                    lines_per_image[ann["image_id"]].append(line)

        for image in coco["images"]:
            records.append(
                {
                    "path": coco_root / split / image["file_name"],
                    "source": source_name(image["file_name"]),
                    "lines": lines_per_image[image["id"]],
                }
            )
    return records, class_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coco_root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/yolo_seg"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists, delete it first")

    records, class_names = load_records(args.coco_root)
    sizes = Counter(r["source"] for r in records)
    assignment = assign_splits(dict(sizes), seed=args.seed)

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        split = assignment[record["source"]]
        images_dir = args.out / split / "images"
        labels_dir = args.out / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["path"], images_dir / record["path"].name)
        (labels_dir / f"{record['path'].stem}.txt").write_text(
            "\n".join(record["lines"]) + "\n", encoding="utf-8"
        )
        counts[split]["images"] += 1
        counts[split]["labels"] += len(record["lines"])

    data_yaml = {
        "train": f"{args.out.name}/train/images",
        "val": f"{args.out.name}/val/images",
        "test": f"{args.out.name}/test/images",
        "names": dict(enumerate(class_names)),
    }
    (args.out.parent / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )

    for split in ("train", "val", "test"):
        print(f"{split}: {counts[split]['images']} images, {counts[split]['labels']} labels")
    print(f"{len(sizes)} source names, one split each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
