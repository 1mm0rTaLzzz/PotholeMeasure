"""Convert RDD2022 VOC annotations into a COCO pothole dataset.

Steps:
    1. Scan the RDD2022 tree for VOC XMLs with D40 (pothole) objects.
    2. Stratified 70/15/15 split by pothole-count bucket.
    3. Resize images to target resolution (default 1280x720) and scale boxes.
    4. Optionally run SAM2 on each bbox to emit polygon segmentation.
    5. Emit ``train.json`` / ``val.json`` / ``test.json`` in COCO format.

Example:
    python scripts/prepare_data.py \\
        --rdd-root /data/RDD2022 \\
        --out-root data \\
        --generate-masks
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a script: `python scripts/prepare_data.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.coco import build_coco, write_coco
from src.data.splits import box_count_bucket, stratified_split
from src.data.voc import scan_rdd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rdd-root", type=Path, required=True, help="root of the RDD2022 dataset")
    p.add_argument("--out-root", type=Path, default=Path("data"), help="output root (annotations + processed images)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generate-masks", action="store_true", help="run SAM2 to produce polygon masks")
    p.add_argument("--sam2-model", default="facebook/sam2-hiera-large")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    samples = scan_rdd(args.rdd_root)
    logging.info("scanned %d samples with D40 objects", len(samples))
    if not samples:
        raise SystemExit("no D40 samples found — check --rdd-root layout")

    train, val, test = stratified_split(samples, key=box_count_bucket, seed=args.seed)
    logging.info("split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))

    mask_generator = None
    if args.generate_masks:
        from src.data.sam2_masks import SAM2MaskGenerator

        mask_generator = SAM2MaskGenerator(model_id=args.sam2_model, device=args.device)
        logging.info("SAM2 mask generator loaded: %s", args.sam2_model)

    ann_dir = args.out_root / "annotations"
    processed_dir = args.out_root / "processed"
    for name, subset in (("train", train), ("val", val), ("test", test)):
        coco = build_coco(
            subset,
            (args.width, args.height),
            processed_dir / name,
            mask_generator=mask_generator,
            progress_desc=f"build {name}",
        )
        write_coco(coco, ann_dir / f"{name}.json")
        logging.info(
            "wrote %s: %d images, %d annotations",
            ann_dir / f"{name}.json",
            len(coco["images"]),
            len(coco["annotations"]),
        )


if __name__ == "__main__":
    main()
