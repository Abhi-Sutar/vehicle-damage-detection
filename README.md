# Vehicle Damage Detection

Instance segmentation of vehicle damage (scratches, dents) with a fine-tuned YOLOv8n-seg model, served as a FastAPI service and packaged in Docker. OpenCV is used for preprocessing and for drawing the results.

Status: work in progress (weekend project).

## What it does

- Takes an image of a car via HTTP.
- Returns the detected damage types, confidences and bounding boxes as JSON.
- Returns an annotated image with the predicted masks drawn on top.

## Stack

- Python, PyTorch, ultralytics (YOLOv8n-seg)
- OpenCV for preprocessing and drawing
- FastAPI + uvicorn
- Docker
- pytest

## Results

To be filled in after training.

| Model | Split | mAP50 (box) | mAP50-95 (box) | mAP50 (mask) | mAP50-95 (mask) |
|-------|-------|-------------|----------------|--------------|-----------------|
| YOLOv8n-seg | held-out test | TBD | TBD | TBD | TBD |

Example outputs will go in `docs/examples/`.

## Dataset

TBD. Datasets are not stored in this repo. See `data/README.md` once it exists for download steps and licence notes.

## Quick start

TBD. Planned:

```
# run the service
docker build -t vehicle-damage .
docker run -p 8000:8000 vehicle-damage

# send an image
curl -F "file=@car.jpg" http://localhost:8000/predict
```

## Repo layout

TBD, will be added once the structure is settled.

## Limitations

TBD (dataset size, lighting and angle sensitivity, class imbalance).
