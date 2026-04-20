"""Fine-tune YOLOv8-seg on the prepared pothole COCO dataset.

Steps:
    1. Convert COCO (data/annotations/*.json + data/processed/<split>) to the
       Ultralytics YOLO-seg layout under data/yolo/.
    2. Emit a data.yaml pointing to it.
    3. Run model.train(...) with TensorBoard logging.
    4. Copy the best checkpoint into experiments/checkpoints/seg/best.pt.

Example:
    python scripts/train_segmentation.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data.yolo import coco_to_yolo, write_data_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--yolo-root", type=Path, default=Path("data/yolo"))
    p.add_argument("--skip-convert", action="store_true", help="reuse existing YOLO layout")
    p.add_argument("--project", type=Path, default=Path("experiments/runs/seg"))
    p.add_argument("--name", default="train")
    return p.parse_args()


def convert_all_splits(data_root: Path, yolo_root: Path) -> Path:
    ann_dir = data_root / "annotations"
    processed = data_root / "processed"
    for split in ("train", "val", "test"):
        coco_json = ann_dir / f"{split}.json"
        if not coco_json.exists():
            logging.warning("missing %s — skipping split", coco_json)
            continue
        n = coco_to_yolo(
            coco_json=coco_json,
            images_dir=processed / split,
            out_images_dir=yolo_root / "images" / split,
            out_labels_dir=yolo_root / "labels" / split,
        )
        logging.info("split %s: wrote %d label lines", split, n)

    data_yaml = yolo_root / "data.yaml"
    write_data_yaml(data_yaml, yolo_root)
    return data_yaml


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = yaml.safe_load(Path(args.config).read_text())
    seg_cfg = config["segmentation"]
    train_cfg = seg_cfg.get("train", {})

    if args.skip_convert and (args.yolo_root / "data.yaml").exists():
        data_yaml = args.yolo_root / "data.yaml"
    else:
        data_yaml = convert_all_splits(args.data_root, args.yolo_root)

    from ultralytics import YOLO

    model = YOLO(seg_cfg["model"])
    results = model.train(
        data=str(data_yaml),
        epochs=train_cfg.get("epochs", 100),
        batch=train_cfg.get("batch_size", 16),
        imgsz=train_cfg.get("imgsz", 1280),
        lr0=train_cfg.get("lr0", 0.001),
        device=train_cfg.get("device", 0),
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )

    # Copy best weights to the canonical location referenced by the pipeline.
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dst = Path(seg_cfg["finetuned_weights"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        logging.info("copied best weights to %s", dst)
    else:
        logging.warning("best.pt not found in %s", results.save_dir)


if __name__ == "__main__":
    main()
