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
  download_pretrained.py       fetch a pothole-pretrained YOLOv8-seg .pt
  train_segmentation.py        YOLOv8 fine-tune (--init <weights>)
  run_inference.py             folder/image → JSON + visualisation
  evaluate.py                  ablation w/ ground truth → CSV / LaTeX
  relative_ablation.py         per-recipe descriptive stats (no GT needed)
  make_paper_figures.py        severity grid + hero from inference output
run_all.ps1                    one-shot driver (Windows PowerShell)
experiments/
  notebooks/demo.ipynb         end-to-end walkthrough on a single image
  checkpoints/                 best.pt drops here
  results/                     ablation CSVs / .tex
tests/                         56 unit tests (see §Testing)
```

## One-shot run for the paper (Windows PowerShell)

After `pip install -r requirements.txt`, drop your RDD2022 tree into
`data\raw\RDD2022\` and run:

```powershell
.\run_all.ps1
```

That orchestrates: `prepare_data` → `download_pretrained` → short fine-tune
→ inference on `data\processed\test` → relative ablation → severity-grid
figure. All artefacts land in `experiments\results\paper\`:

| file | purpose |
|---|---|
| `inference\json\<stem>.json` | per-image detections w/ depth_m, area_m2, severity |
| `inference\viz\<stem>.jpg`   | severity-coloured overlays |
| `inference\figures\severity_grid.jpg` | 2x2 paper figure (one example per bucket) |
| `inference\figures\hero.jpg` | teaser image |
| `ablation\relative_ablation.tex` | LaTeX table of per-recipe stats |
| `ablation\relative_ablation.png` | violin plot per recipe |

Common knobs:

```powershell
.\run_all.ps1 -SkipFinetune                              # keremberke weights only
.\run_all.ps1 -HfRepo keremberke/yolov8m-pothole-segmentation -Epochs 30
.\run_all.ps1 -SkipDataPrep -SkipMasks                   # iterate on inference only
```

Real MAE/RMSE numbers require ground-truth depth/area — see "Ground truth"
below.

## Quick start (manual, cross-platform)

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

# 6a. relative ablation (no GT needed — descriptive stats per recipe)
python scripts/relative_ablation.py \
    --input-dir data/processed/test \
    --output experiments/results/run1/ablation

# 6b. paper figures (severity grid + hero)
python scripts/make_paper_figures.py \
    --inference-dir experiments/results/run1

# 6c. ablation with ground truth → real MAE / RMSE / severity F1
python scripts/evaluate.py \
    --config configs/default.yaml \
    --gt data/annotations/test_gt.json \
    --output experiments/results/ablation.csv --latex
```

## Ground truth for the publishable MAE/RMSE table

`scripts/evaluate.py` needs per-pothole physical measurements that RDD2022
doesn't provide. Two ways to obtain them:

1. **Hand-measure ~30 potholes** with a tape (or chalk grid for area). Record
   `depth_m`, `area_m2`, `severity` per pothole, plus the path to the binary
   mask PNG. Save as `data/annotations/test_gt.json` — schema is in the
   docstring of `scripts/evaluate.py`. 30 points is enough for a workshop
   submission.
2. **Use a public RGB-D pothole dataset** (e.g. PothRGBD, arXiv 2505.04207)
   that ships per-pothole depth/area; convert to the same JSON schema and
   point `--gt` at it.

Without GT you can still report `relative_ablation.py` results — they show
that the three recipes produce different distributions on the same
detections, which is enough to motivate the method qualitatively.

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
