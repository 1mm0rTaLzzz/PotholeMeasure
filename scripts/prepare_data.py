"""Convert an RDD2022-style dataset into the project's COCO pothole layout.

Two input formats are supported:

``--format voc``  (default)
    Raw RDD2022 VOC layout — the script scans for ``annotations/**/*.xml``,
    filters ``<name>`` == ``--target-class`` (D40 by default), then does a
    stratified 70/15/15 split by pothole-count bucket.

``--format yolo``
    Pre-split YOLO layout::

        <rdd_root>/{train,val,test}/images/*.jpg
        <rdd_root>/{train,val,test}/labels/*.txt  # YOLO: cls cx cy w h

    Splits are taken as-is; each label line is filtered by ``--class-id``
    (default ``3`` = D40 in the common RDD ordering D00/D10/D20/D40).

Either way the output is::

    <out-root>/annotations/{train,val,test}.json   # COCO
    <out-root>/processed/{train,val,test}/*.jpg    # resized images

Example (VOC)::

    python scripts/prepare_data.py --rdd-root /data/RDD2022 --out-root data --generate-masks

Example (YOLO pre-split)::

    python scripts/prepare_data.py \\
        --rdd-root /data/rdd2022 --format yolo --class-id 3 \\
        --out-root data --generate-masks
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
from src.data.voc import POTHOLE_CLASS, scan_rdd
from src.data.yolo_rdd import scan_yolo_rdd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rdd-root", type=Path, required=True, help="root of the dataset")
    p.add_argument("--out-root", type=Path, default=Path("data"), help="output root (annotations + processed images)")
    p.add_argument("--format", choices=("voc", "yolo"), default="voc", help="input annotation format")
    p.add_argument("--target-class", default=POTHOLE_CLASS, help="[voc] VOC <name> to keep (default: D40)")
    p.add_argument("--class-id", type=int, default=3, help="[yolo] class id to keep (default: 3 = D40)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generate-masks", action="store_true", help="run SAM2 to produce polygon masks")
    p.add_argument("--sam2-model", default="facebook/sam2-hiera-large")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _collect_splits(args: argparse.Namespace) -> dict[str, list]:
    if args.format == "voc":
        samples = scan_rdd(args.rdd_root, target_class=args.target_class)
        logging.info("scanned %d samples with %r objects", len(samples), args.target_class)
        if not samples:
            raise SystemExit(
                f"no {args.target_class!r} samples found — check --rdd-root layout "
                f"(see the class-distribution log above; override with --target-class if needed)"
            )
        train, val, test = stratified_split(samples, key=box_count_bucket, seed=args.seed)
        return {"train": train, "val": val, "test": test}

    # --format yolo
    splits = scan_yolo_rdd(args.rdd_root, target_class_id=args.class_id)
    total = sum(len(v) for v in splits.values())
    if total == 0:
        raise SystemExit(
            f"no samples for class-id={args.class_id} in {args.rdd_root}/(train|val|test)/labels "
            f"— check the YOLO class-distribution log above and pass --class-id accordingly"
        )
    return splits


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    splits = _collect_splits(args)
    logging.info("split sizes: %s", {k: len(v) for k, v in splits.items()})

    mask_generator = None
    if args.generate_masks:
        from src.data.sam2_masks import SAM2MaskGenerator

        mask_generator = SAM2MaskGenerator(model_id=args.sam2_model, device=args.device)
        logging.info("SAM2 mask generator loaded: %s", args.sam2_model)

    ann_dir = args.out_root / "annotations"
    processed_dir = args.out_root / "processed"
    for name, subset in splits.items():
        if not subset:
            logging.warning("split %s is empty — skipping", name)
            continue
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
