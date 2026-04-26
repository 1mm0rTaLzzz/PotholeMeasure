"""Import PothRGBD (Kaggle release) and emit ground-truth JSON for evaluate.py.

Pipeline:
    1. Auto-detect the dataset layout under ``--src`` (Roboflow exports come
       in a few flavours).
    2. For every pothole instance:
       a. Load the GT depth map (D415 16-bit PNG mm by default).
       b. Build a binary mask from the YOLO-seg polygon.
       c. Fit a road plane to NON-mask pixels in the same image (RANSAC).
       d. ``depth_m`` (GT) = p95 of |signed distance to plane| inside the mask.
       e. ``area_m2`` (GT) = sum of per-pixel BEV area
          ``(z/fx) * (z/fy) / cos(theta)`` inside the mask, where ``z`` is the
          GT depth at the pixel and theta is the angle between the camera
          optical axis and the plane normal.
    3. Severity is bucketed via ``configs/default.yaml`` thresholds.
    4. Emit:
        ``data/pothrgbd/processed/<split>/<id>.jpg``       resized RGB
        ``data/pothrgbd/masks/<split>/<id>_<n>.png``       binary mask
        ``data/pothrgbd/calibration.yaml``                 D415 intrinsics
        ``data/pothrgbd/test_gt.json``                     ready for evaluate.py

Example::

    python scripts/import_pothrgbd.py \\
        --src C:\\datasets\\pothrgbd \\
        --out-root data\\pothrgbd \\
        --target-size 1280 720
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="root of the unzipped PothRGBD download")
    p.add_argument("--out-root", type=Path, default=Path("data/pothrgbd"))
    p.add_argument("--depth-scale", type=float, default=0.001,
                   help="multiplier from depth-image units to metres (D415 PNG = mm = 0.001)")
    p.add_argument("--target-size", type=int, nargs=2, default=None, metavar=("W", "H"),
                   help="resize RGB+masks to this WxH; default keeps native 640x480")
    p.add_argument("--severity-config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--ransac-threshold-m", type=float, default=0.02)
    p.add_argument("--min-inlier-ratio", type=float, default=0.5,
                   help="loosened from 0.7: PothRGBD is close-up so fewer non-mask pixels")
    p.add_argument("--max-iters", type=int, default=2000)
    return p.parse_args()


def _resize_pair(rgb, depth, masks, target):
    import cv2

    if target is None:
        return rgb, depth, masks
    tw, th = target
    if rgb.shape[1] == tw and rgb.shape[0] == th:
        return rgb, depth, masks
    rgb_r = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    depth_r = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_NEAREST)
    masks_r = [cv2.resize(m.astype("uint8"), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]
    return rgb_r, depth_r, masks_r


def _scale_intrinsics(K, src_size, dst_size):
    import numpy as np

    sw, sh = src_size
    dw, dh = dst_size
    sx = dw / sw
    sy = dh / sh
    K2 = np.asarray(K, dtype=np.float64).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import cv2
    import numpy as np
    import yaml

    from src.classifier import make_severity_classifier
    from src.data.pothrgbd import (
        discover_pothrgbd,
        iter_loaded,
        load_depth_metres,
    )
    from src.geometry.plane_fitting import (
        compute_depth_offset,
        depth_to_pointcloud,
        fit_road_plane,
    )

    cfg = yaml.safe_load(args.severity_config.read_text())
    sev_cfg = cfg["severity"]
    classify = make_severity_classifier(sev_cfg["thresholds"], signal=sev_cfg.get("signal", "max"))

    # Load D415 default intrinsics; we'll scale them if --target-size is set.
    calib_src = Path("data/calibration/pothrgbd_d415.yaml")
    calib = yaml.safe_load(calib_src.read_text())
    K_native = np.asarray(calib["K"], dtype=np.float64)
    src_size = (calib["resolution"]["width"], calib["resolution"]["height"])

    out_calib = args.out_root / "calibration.yaml"
    out_calib.parent.mkdir(parents=True, exist_ok=True)
    if args.target_size:
        K = _scale_intrinsics(K_native, src_size, tuple(args.target_size))
        calib_out = dict(calib)
        calib_out["resolution"] = {"width": args.target_size[0], "height": args.target_size[1]}
        calib_out["K"] = K.tolist()
        out_calib.write_text(yaml.safe_dump(calib_out, sort_keys=False))
    else:
        K = K_native
        shutil.copy2(calib_src, out_calib)

    splits = discover_pothrgbd(args.src)
    if not any(splits.values()):
        raise SystemExit(f"no usable RGB+depth+label triplets under {args.src}")

    test_gt_records: list[dict] = []
    written_total = 0
    img_idx = 0

    for split, samples in splits.items():
        if not samples:
            continue
        rgb_dir = args.out_root / "processed" / split
        mask_dir = args.out_root / "masks" / split
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for s in iter_loaded(samples):
            rgb = cv2.imread(str(s.rgb_path))
            try:
                depth = load_depth_metres(s.depth_path, scale=args.depth_scale)
            except FileNotFoundError as e:
                logging.warning("skipping %s: %s", s.rgb_path, e)
                continue
            if rgb is None or depth is None:
                continue
            if depth.shape[:2] != rgb.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            rgb, depth, masks = _resize_pair(rgb, depth, s.masks, args.target_size)
            if not masks:
                continue
            h, w = depth.shape[:2]

            # Fit road plane (with masks excluded).
            pc = depth_to_pointcloud(depth, K)
            plane = fit_road_plane(
                pc, (h, w),
                exclude_masks=masks,
                threshold_m=args.ransac_threshold_m,
                max_iters=args.max_iters,
                min_inlier_ratio=args.min_inlier_ratio,
                min_points=500,
                up_axis=(0.0, -1.0, 0.0),
                up_cos_threshold=0.5,            # PothRGBD is close-up so the plane normal can tilt
            )
            if plane is None:
                logging.info("plane fit failed for %s — skipping (%d masks)", s.rgb_path.name, len(masks))
                continue

            # Cache + write the resized RGB.
            img_idx += 1
            out_rgb_name = f"{img_idx:07d}.jpg"
            cv2.imwrite(str(rgb_dir / out_rgb_name), rgb)

            for mi, mask in enumerate(masks):
                if not mask.any():
                    continue
                # Per-instance mask PNG (255 inside, 0 outside).
                mask_name = f"{img_idx:07d}_{mi}.png"
                cv2.imwrite(str(mask_dir / mask_name), (mask.astype(np.uint8) * 255))

                # GT depth_m = p95 |signed dist to plane| inside the mask.
                stats = compute_depth_offset(plane, pc, mask)
                gt_depth_m = float(stats.p95_m)

                # GT area_m2 = sum of per-pixel BEV area inside the mask.
                # Each pixel covers (z/fx) * (z/fy) m^2 on the image plane;
                # divide by cos(angle between optical axis and plane normal)
                # to project onto the road plane.
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                z = depth[mask]
                z = z[(z > 0) & np.isfinite(z)]
                if z.size == 0:
                    continue
                axis_dot = abs(float(plane.normal[2]))   # |n . z_axis|
                cos_angle = max(axis_dot, 0.05)
                area_m2 = float((z * z / (fx * fy)).sum() / cos_angle)
                if not np.isfinite(area_m2) or area_m2 <= 0.0:
                    continue

                severity = classify(gt_depth_m, area_m2)

                test_gt_records.append({
                    "image": str((rgb_dir / out_rgb_name).as_posix()),
                    "mask_path": str((mask_dir / mask_name).as_posix()),
                    "depth_m": round(gt_depth_m, 4),
                    "area_m2": round(area_m2, 4),
                    "severity": severity,
                })
                written_total += 1

            if img_idx % 50 == 0:
                logging.info("processed %d images, %d annotations so far", img_idx, written_total)

    out_json = args.out_root / "test_gt.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(test_gt_records, indent=2))
    logging.info("wrote %s with %d annotations across %d images", out_json, written_total, img_idx)
    logging.info("calibration written to %s", out_calib)
    logging.info(
        "next: python scripts/evaluate.py --gt %s --output experiments\\results\\paper\\evaluate.csv --latex",
        out_json.as_posix(),
    )


if __name__ == "__main__":
    main()
