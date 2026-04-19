# PotholeMeasure

Pipeline for detecting road potholes and estimating their **depth** (as offset
from a RANSAC-fitted road plane over a monocular metric-depth map) and
**area** (via homography to bird's-eye view). Potholes are classified by
severity in accordance with GOST R 50597-2017.

## Pipeline

```
image ─► YOLOv8-seg  ──► masks
      ─► Depth Anything V2 Metric ──► depth map (m)
      ─► Homography (template or lanes) ──► BEV
                     │
                     ▼
            RANSAC road plane (on non-pothole points)
                     │
                     ▼
   per mask: depth = p95(|plane − depth|),
             area  = BEV polygon area,
             severity by GOST R 50597 thresholds
                     │
                     ▼
            JSON + visualisation
```

## Project layout

```
configs/        YAML configs
data/
  raw/          input images (RDD2022)
  annotations/  COCO JSON (train/val/test)
  calibration/  camera intrinsics + mounting
src/
  models/       segmentation, depth
  geometry/     plane fitting, homography, metrics
  pipeline.py   end-to-end class
  classifier.py severity thresholds
  visualize.py  overlays, 3D view
scripts/        prepare_data, train, run_inference, evaluate
experiments/    checkpoints, notebooks
tests/          unit tests
```

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. sanity check
python -c "import torch; print(torch.cuda.is_available())"

# 3. (later phases) prepare dataset, train, inference
python scripts/prepare_data.py --dataset rdd2022
python scripts/train_segmentation.py --config configs/default.yaml
python scripts/run_inference.py --input path/to/image.jpg --output results/
```

## Phased development

| # | Phase | Status |
|---|---|---|
| 1 | Environment scaffold | in progress |
| 2 | Data preparation (RDD2022 + SAM2 pseudo-masks) | todo |
| 3 | Instance segmentation (YOLOv8-seg) | todo |
| 4 | Metric depth (Depth Anything V2) | todo |
| 5 | RANSAC road plane + depth offset | todo |
| 6 | Homography + area in m² | todo |
| 7 | Severity classifier (GOST R 50597) | todo |
| 8 | End-to-end pipeline + inference script | todo |
| 9 | Visualisation | todo |
| 10 | Experiments, ablations, metrics | todo |
| 11 | Tests + docs | todo |

Execution order: `1 → 2 → 3 → 4 → 5 → 6 → 8 → 7 → 9 → 10 → 11`.

## References

- Depth Anything V2 (Metric Outdoor): https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
- Ultralytics YOLOv8 segmentation: https://docs.ultralytics.com/tasks/segment/
- RDD2022 dataset: https://github.com/sekilab/RoadDamageDetector
- GOST R 50597-2017 — pavement defect thresholds.
