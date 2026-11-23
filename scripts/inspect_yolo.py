"""Report class counts and points per label for a YOLO folder (<split>/images, <split>/labels)."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="folder that holds the split folders")
    args = parser.parse_args()

    splits = sorted(p.name for p in args.root.iterdir() if (p / "labels").is_dir())
    for split in splits:
        files = sorted((args.root / split / "labels").glob("*.txt"))
        images = len(list((args.root / split / "images").glob("*.jpg")))
        classes: Counter[str] = Counter()
        points = []
        for file in files:
            for line in file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    classes[parts[0]] += 1
                    points.append((len(parts) - 1) // 2)
        print(f"{split}: {images} images, {len(files)} label files, {len(points)} labels")
        print(f"  classes    {dict(sorted(classes.items()))}")
        print(f"  points     min {min(points)}, median {median(points):g}, max {max(points)}")
        quads = sum(1 for n in points if n == 4)
        print(f"  4 points   {quads} of {len(points)}")
        if quads == len(points):
            print("  verdict    every label has 4 points: boxes, not polygons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
