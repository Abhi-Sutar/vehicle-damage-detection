"""Report the Python, torch, CUDA, ultralytics and OpenCV setup."""

import argparse
import platform
import sys

import cv2
import numpy as np
import torch
import ultralytics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predict",
        action="store_true",
        help="download yolov8n-seg.pt and run one prediction on a blank image",
    )
    args = parser.parse_args()

    print(f"python       {platform.python_version()} ({sys.executable})")
    print(f"torch        {torch.__version__} (CUDA build {torch.version.cuda})")
    cuda_ok = torch.cuda.is_available()
    print(f"cuda ok      {cuda_ok}")
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        x = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        print(f"gpu          {props.name}, {props.total_memory / 2**30:.1f} GiB")
        print(f"gpu matmul   ok, mean {(x @ x).abs().mean().item():.1f}")
    print(f"ultralytics  {ultralytics.__version__}")
    print(f"opencv       {cv2.__version__}")
    print(f"numpy        {np.__version__}")

    if args.predict:
        model = ultralytics.YOLO("yolov8n-seg.pt")
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = model.predict(image, device=0 if cuda_ok else "cpu", verbose=False)[0]
        print(f"predict      ok, {len(result.boxes)} detections on a blank image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
