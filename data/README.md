# Data

The dataset is not stored in this repo. This file says where it comes from and how to get it.

## Source

- Name: Car Damage V5, version 5 (generated 2023-08-20)
- Author: Roboflow Universe workspace "Car Damage" (car-damage-kadad)
- URL: https://universe.roboflow.com/car-damage-kadad/car-damage-v5/dataset/5
- Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Downloaded: 2026-11-22, export format "COCO Segmentation", into `data/raw/car_damage_v5_5_coco/`

## Contents as exported

- 3,001 images, 640x640, stretched by Roboflow from the originals (aspect ratio not kept)
- Split: train 2105, valid 574, test 322, every image has at least one annotation
- One polygon per instance, median about 13 points, 26 of 5,012 polygons have 4 points or fewer
- Categories: 1 Broken Glass, 2 Dent, 3 Scratch, 4 Wreck (id 0 is an unused parent category)
- Instances (train / valid / test): Broken Glass 559 / 158 / 84, Dent 706 / 226 / 89, Scratch 1428 / 278 / 172, Wreck 914 / 249 / 149

## Known issues

- The 3,001 images come from 1,387 distinct original file names, and 467 of those names appear in more than one split. Some are the same photo in train and test. The split is rebuilt in M1, grouped by original name.
- Images are stretched to a square, so cars look wider or taller than they are.
- The YOLOv8 exports of this dataset (version 5 in `car_damage_v5_5`, version 6 in `car_damage_v5_6`) hold rotated boxes with 4 points per label, not outlines, so they are not used. Version 6 is also only 320x320.

## How to get it

1. Sign in at the URL above, choose Download Dataset, format "COCO Segmentation", download zip.
2. Unzip into `data/raw/car_damage_v5_5_coco/` so the `train`, `valid` and `test` folders sit directly inside it.
3. Check it: `uv run scripts/inspect_coco.py data/raw/car_damage_v5_5_coco`
4. Build the YOLO-seg copy: `uv run scripts/prepare_dataset.py data/raw/car_damage_v5_5_coco`. It writes `data/yolo_seg/` and `data/data.yaml`.

## Changes made in this project

Listed as per CC BY 4.0 requirements. The images are unchanged. The labels were changed as follows (2026-11-23):

- Converted from COCO polygons in pixels to YOLO-seg polygons (0 to 1), with the unused category 0 dropped and ids 1 to 4 shifted to 0 to 3.
- Split again, grouped by original file name, into train 2102 / val 570 / test 329 images (labels 3425 / 962 / 625), instead of Roboflow's 2105 / 574 / 322.
- Instances (train / val / test): Broken Glass 551 / 129 / 121, Dent 718 / 177 / 126, Scratch 1251 / 391 / 236, Wreck 905 / 265 / 142.
- All four classes kept.

