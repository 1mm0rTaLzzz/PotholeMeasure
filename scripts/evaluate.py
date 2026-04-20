"""Ablation evaluation: runs multiple depth/area recipes on a ground-truth
dataset and emits a CSV + LaTeX table of MAE / RMSE / severity F1.

Ground truth JSON schema (one file for the whole test set):

    [
      {
        "image": "path/to/img.jpg",
        "mask_path": "path/to/mask_0.png",   # binary 0/255
        "depth_m": 0.08,
        "area_m2": 0.12,
        "severity": "major"
      },
      ...
    ]

Example:
    python scripts/evaluate.py \\
        --config configs/default.yaml \\
        --gt data/annotations/test_gt.json \\
        --output experiments/results/ablation.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--gt", type=Path, required=True, help="ground-truth JSON file")
    p.add_argument("--output", type=Path, default=Path("experiments/results/ablation.csv"))
    p.add_argument("--latex", action="store_true", help="also emit .tex alongside CSV")
    return p.parse_args()


def load_gt(path: Path):
    import cv2

    from src.evaluation import PotholeGroundTruth

    records = json.loads(path.read_text())
    by_image: dict[str, list[PotholeGroundTruth]] = defaultdict(list)
    for rec in records:
        mask = cv2.imread(rec["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.warning("missing mask %s", rec["mask_path"])
            continue
        by_image[rec["image"]].append(
            PotholeGroundTruth(
                image_id=rec["image"],
                mask=(mask > 127),
                depth_m=float(rec["depth_m"]),
                area_m2=float(rec["area_m2"]),
                severity=str(rec["severity"]),
            )
        )
    return by_image


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import cv2
    import yaml

    from src.classifier import make_severity_classifier
    from src.evaluation import run_ablation
    from src.geometry.homography import (
        compute_homography_from_calibration,
        load_calibration,
        mask_to_area_m2,
    )
    from src.geometry.metrics import classification_metrics, regression_metrics
    from src.geometry.plane_fitting import depth_to_pointcloud, fit_road_plane
    from src.models.depth import load_from_yaml_config as load_depth

    cfg = yaml.safe_load(Path(args.config).read_text())
    K, calib = load_calibration(cfg["paths"]["calibration"])
    H = compute_homography_from_calibration(
        K,
        camera_height_m=float(calib["mount"]["height_m"]),
        pitch_deg=float(calib["mount"]["pitch_deg"]),
    )
    classifier = make_severity_classifier(cfg["severity"]["thresholds"])

    depth_estimator = load_depth(cfg)
    plane_cfg = cfg.get("plane_fitting", {})
    gt_by_image = load_gt(args.gt)

    per_image_signals = []
    gt_depth, gt_area, gt_severity = [], [], []
    for image_path, gts in gt_by_image.items():
        image = cv2.imread(image_path)
        if image is None:
            logging.warning("missing image %s", image_path)
            continue
        depth_map = depth_estimator.predict(image)
        pc = depth_to_pointcloud(depth_map, K)
        plane = fit_road_plane(
            pc,
            depth_map.shape,
            exclude_masks=[g.mask for g in gts],
            threshold_m=float(plane_cfg.get("ransac_threshold_m", 0.02)),
            min_inlier_ratio=float(plane_cfg.get("min_inlier_ratio", 0.7)),
            min_points=int(plane_cfg.get("min_points", 1000)),
        )
        per_image_signals.append((depth_map, pc, plane, gts))
        gt_depth.extend(g.depth_m for g in gts)
        gt_area.extend(g.area_m2 for g in gts)
        gt_severity.extend(g.severity for g in gts)

    runs = run_ablation(
        per_image_signals,
        area_fn=lambda mask: mask_to_area_m2(mask, H),
        classifier=classifier,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recipe", "n", "depth_mae_m", "depth_rmse_m", "area_mae_m2", "severity_acc", "severity_f1"])
        for run in runs:
            depth_m = regression_metrics(run.depth_preds, gt_depth)
            area_m = regression_metrics(run.area_preds, gt_area)
            sev_m = classification_metrics(run.severity_preds, gt_severity) if run.severity_preds else None
            writer.writerow([
                run.recipe,
                depth_m.n,
                round(depth_m.mae, 4),
                round(depth_m.rmse, 4),
                round(area_m.mae, 4),
                round(sev_m.accuracy, 3) if sev_m else "",
                round(sev_m.macro_f1, 3) if sev_m else "",
            ])

    logging.info("wrote %s", args.output)

    if args.latex:
        tex_path = args.output.with_suffix(".tex")
        with tex_path.open("w") as f:
            f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
            f.write("recipe & n & MAE depth (m) & RMSE depth (m) & MAE area (m²) & acc & F1 \\\\\n\\midrule\n")
            for run in runs:
                depth_m = regression_metrics(run.depth_preds, gt_depth)
                area_m = regression_metrics(run.area_preds, gt_area)
                sev_m = classification_metrics(run.severity_preds, gt_severity) if run.severity_preds else None
                f.write(
                    f"{run.recipe} & {depth_m.n} & {depth_m.mae:.3f} & {depth_m.rmse:.3f}"
                    f" & {area_m.mae:.3f} & {sev_m.accuracy if sev_m else 0:.3f}"
                    f" & {sev_m.macro_f1 if sev_m else 0:.3f} \\\\\n"
                )
            f.write("\\bottomrule\n\\end{tabular}\n")
        logging.info("wrote %s", tex_path)


if __name__ == "__main__":
    main()
