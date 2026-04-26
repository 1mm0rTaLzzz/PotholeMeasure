# PotholeMeasure

Pipeline for detecting road potholes and estimating their **depth** (as the
offset from a RANSAC-fitted road plane over a monocular metric-depth map) and
**area** (via a ground-plane homography to bird's-eye view). Potholes are
classified by severity in accordance with **GOST R 50597-2017**.

The main idea that makes the approach reliable with a single camera is to
**never trust absolute monocular depth** — instead, fit a local road plane
with RANSAC on points outside the pothole masks and report the pothole depth
as the signed offset from that plane. This cancels the scale drift that
plagues monocular metric depth models.

## Pipeline

```
image ─► YOLOv8-seg                    ──► instance masks
      ─► Depth Anything V2 Metric      ──► metric depth (m)
      ─► Homography (calibration/lanes)──► BEV (m)
                        │
                        ▼
                RANSAC road plane on non-pothole 3D points
                        │
                        ▼
     per mask:  depth_m = p95(|signed distance to plane|)
                area_m2 = polygon area in BEV
                severity = GOST R 50597 buckets (depth ∨ area)
                        │
                        ▼
                JSON + visualisation (2D overlay / 3-panel / Open3D)
```

## Project layout

```
configs/default.yaml           paths, thresholds, plane-fit params
data/
  raw/                         RDD2022 images
  annotations/                 COCO train/val/test
  calibration/camera.yaml      dashcam template (H, pitch, FOV)
src/
  data/                        VOC→COCO, SAM2 pseudo-masks, YOLO export
  models/                      segmentation (YOLOv8-seg), depth (DA V2)
  geometry/                    plane_fitting, homography, metrics
  pipeline.py                  PotholePipeline.process(image)
  classifier.py                GOST R 50597 severity buckets
  evaluation.py                ablation recipes + runner
  visualize.py                 overlays, 3-panel, Open3D
scripts/
  prepare_data.py              RDD2022 → COCO (+SAM2 masks)
  train_segmentation.py        YOLOv8 fine-tune
  run_inference.py             folder/image → JSON + visualisation
  evaluate.py                  ablation → CSV / LaTeX
experiments/
  notebooks/demo.ipynb         end-to-end walkthrough on a single image
  checkpoints/                 best.pt drops here
  results/                     ablation CSVs / .tex
tests/                         56 unit tests (see §Testing)
```

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. sanity check
python -c "import torch, ultralytics, transformers, cv2"

# 3. prepare data
# 3a. raw RDD2022 (VOC XML, needs stratified split)
python scripts/prepare_data.py \
    --rdd-root data/raw/RDD2022 \
    --out-root data \
    --generate-masks --sam2-model facebook/sam2-hiera-large

# 3b. pre-split YOLO layout (train/val/test/{images,labels}/)
python scripts/prepare_data.py \
    --rdd-root data/raw/rdd2022 --format yolo --class-id 3 \
    --out-root data \
    --generate-masks --sam2-model facebook/sam2-hiera-large

# 4. segmentation weights — pick ONE of the following:

# 4a. (recommended) drop in a pretrained pothole checkpoint and skip training.
#     keremberke/yolov8s-pothole-segmentation has mAP@0.5 ≈ 0.99 on its val set
#     and is single-class "pothole", so it loads straight into the pipeline.
python scripts/download_pretrained.py \
    --hf-repo keremberke/yolov8s-pothole-segmentation
# → writes experiments/checkpoints/seg/keremberke_yolov8s-pothole-segmentation.pt
# Update configs/default.yaml → segmentation.finetuned_weights to that path.

# 4b. fine-tune on your own data (start from the pretrained checkpoint above to
#     converge in ~20 epochs instead of 100 from COCO weights).
python scripts/train_segmentation.py --config configs/default.yaml

# 5. single image / folder inference
python scripts/run_inference.py \
    --config configs/default.yaml \
    --input path/to/image_or_folder \
    --output experiments/results/run1

# 6. ablation table for the paper
python scripts/evaluate.py \
    --config configs/default.yaml \
    --gt data/annotations/test_gt.json \
    --output experiments/results/ablation.csv --latex
```

## Configuration

`configs/default.yaml` drives every stage. The pieces that matter for the
measurement quality:

- `plane_fitting.ransac_threshold_m` (default `0.02`) — inlier distance in m.
- `plane_fitting.min_inlier_ratio` (`0.7`) — reject frames where the plane
  covers less than 70% of the non-pothole pixels.
- `plane_fitting.up_cos` (`0.9`) — the plane normal must be within ~25° of the
  gravity axis; a sanity check that avoids fitting walls / car hoods.
- `severity.thresholds.depth_m` / `area_m2` — GOST R 50597 bucket edges. The
  final severity is the worse of the two buckets.

Calibration lives in `data/calibration/camera.yaml`; replace the dashcam
defaults (H = 1.2 m, FOV = 60°, pitch = −5°) with a real calibration when one
is available.

## Ablation recipes

`scripts/evaluate.py` runs every recipe on the same detections and emits
MAE / RMSE / severity F1:

| recipe                       | depth signal                                        |
|------------------------------|-----------------------------------------------------|
| `midas_relative_scaled`      | relative depth rescaled by 5/95-percentiles (drift) |
| `metric_no_plane`            | `max − min` of the metric depth inside the mask     |
| `metric_plane_offset`        | **ours** — p95 of `|distance to RANSAC plane|`     |

Severity F1 is reported when a classifier is passed; area in m² is always the
BEV polygon area.

## Testing

```
pytest -q
# 56 passed
```

- `test_plane_fitting.py` — **critical**: synthetic plane + 8 cm pit,
  end-to-end recovery within 0.5 cm.
- `test_homography.py` — 1 m² target recovered within 15%.
- `test_classifier.py` — GOST buckets, worst-of-two rule.
- `test_pipeline.py` — end-to-end with mocked segmentor / depth model.
- `test_metrics_and_ablation.py` — regression + classification metrics,
  `run_ablation` sanity on a synthetic pit.

The heavy deps (`torch`, `ultralytics`, `transformers`, `open3d`) are imported
lazily inside functions, so the pure-logic tests and `--help` work on a
minimal install.

## References

- Depth Anything V2 (Metric Outdoor): https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
- Ultralytics YOLOv8 segmentation: https://docs.ultralytics.com/tasks/segment/
- SAM2: https://github.com/facebookresearch/sam2
- RDD2022 dataset: https://github.com/sekilab/RoadDamageDetector
- GOST R 50597-2017 — pavement defect thresholds.
