"""Convert a YOLO-seg test split into a COCO instance-segmentation JSON.

Uses existing project utilities:
  - src.data.pothrgbd.parse_yolo_seg_label  (YOLO-seg txt → bool masks)
  - src.data.coco.mask_to_polygons          (mask → COCO polygon)
  - src.data.coco.write_coco                (dict → JSON)

Example::

    python scripts/yolo_seg_to_coco.py \
        --images-dir data/pothrgbd_yolo/test/images \
        --labels-dir data/pothrgbd_yolo/test/labels \
        --out data/annotations/test.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--class-name", default="pothole")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import cv2

    from src.data.coco import mask_to_polygons, write_coco
    from src.data.pothrgbd import parse_yolo_seg_label

    category = {"id": 1, "name": args.class_name, "supercategory": "road_damage"}

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    img_paths = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    if not img_paths:
        raise SystemExit(f"no images found in {args.images_dir}")

    for img_id, img_path in enumerate(img_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning("cannot read %s — skipped", img_path)
            continue
        h, w = img.shape[:2]

        label_path = args.labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            logging.warning("no label for %s — skipped", img_path.name)
            continue

        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        masks = parse_yolo_seg_label(label_path, w, h)
        for mask in masks:
            if not mask.any():
                continue
            polys = mask_to_polygons(mask)
            if not polys:
                continue

            ys, xs = mask.nonzero()
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bbox = [x0, y0, x1 - x0, y1 - y0]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category["id"],
                "segmentation": polys,
                "bbox": bbox,
                "area": float(int(mask.sum())),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": f"PothRGBD test split — {args.class_name}"},
        "images": images,
        "annotations": annotations,
        "categories": [category],
    }
    write_coco(coco, args.out)
    logging.info(
        "wrote %s: %d images, %d annotations",
        args.out, len(images), len(annotations),
    )


if __name__ == "__main__":
    main()
