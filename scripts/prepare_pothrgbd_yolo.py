"""Prepare PothRGBD for YOLOv8-seg training and RGB-D test evaluation.

Input layouts are auto-detected by ``src.data.pothrgbd``. The output is both:

* an Ultralytics YOLO-seg dataset:
  ``<out>/{train,val,test}/{images,labels}`` plus ``<out>/data.yaml``;
* a split-preserving RGB-D layout with ``depths/`` next to each split, so
  ``scripts/import_pothrgbd.py --src <out> --splits test`` can build a
  leakage-free GT file for the paper table.
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="root of the unzipped PothRGBD download")
    p.add_argument("--out-root", type=Path, default=Path("data/pothrgbd_yolo"))
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean", action="store_true", help="remove --out-root before writing")
    return p.parse_args()


def _normalise_label(src: Path, dst: Path) -> None:
    lines: list[str] = []
    for raw in src.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 7:
            continue
        parts[0] = "0"
        lines.append(" ".join(parts))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def _write_data_yaml(out_root: Path) -> None:
    content = (
        f"path: {out_root.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: pothole\n"
    )
    (out_root / "data.yaml").write_text(content)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.data.pothrgbd import discover_pothrgbd

    splits_in = discover_pothrgbd(args.src)
    samples = [s for split_samples in splits_in.values() for s in split_samples]
    if not samples:
        raise SystemExit(f"no PothRGBD RGB+depth+label samples found under {args.src}")

    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)

    rng = random.Random(args.seed)
    samples = sorted(samples, key=lambda s: s.rgb_path.name)
    rng.shuffle(samples)

    n = len(samples)
    n_train = int(round(n * args.train_frac))
    n_val = int(round(n * args.val_frac))
    split_map = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split, subset in split_map.items():
        img_dir = args.out_root / split / "images"
        label_dir = args.out_root / split / "labels"
        depth_dir = args.out_root / split / "depths"
        for d in (img_dir, label_dir, depth_dir):
            d.mkdir(parents=True, exist_ok=True)
        for sample in subset:
            shutil.copy2(sample.rgb_path, img_dir / sample.rgb_path.name)
            _normalise_label(sample.label_path, label_dir / f"{sample.rgb_path.stem}.txt")
            shutil.copy2(sample.depth_path, depth_dir / sample.depth_path.name)
        logging.info("split %s: %d samples", split, len(subset))

    _write_data_yaml(args.out_root)
    logging.info("wrote %s", args.out_root / "data.yaml")


if __name__ == "__main__":
    main()
