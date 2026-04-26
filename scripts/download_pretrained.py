"""Download a pre-trained pothole-segmentation checkpoint from HuggingFace.

Why: training YOLOv8-seg from COCO weights on a small RDD2022 subset is slow
and converges to a mediocre mAP. Starting from a checkpoint that has already
seen thousands of pothole crops gets you to 0.9+ mAP in a fraction of the
epochs (or sometimes obviates fine-tuning entirely).

Recommended models (Ultralytics-compatible single-class ``pothole`` weights):

    keremberke/yolov8n-pothole-segmentation   # ~6 MB, fastest
    keremberke/yolov8s-pothole-segmentation   # ~22 MB, default — best quality / speed
    keremberke/yolov8m-pothole-segmentation   # ~52 MB, slowest but most accurate

Example::

    python scripts/download_pretrained.py \\
        --hf-repo keremberke/yolov8s-pothole-segmentation

Then plug the resulting path into ``configs/default.yaml`` under
``segmentation.finetuned_weights``.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--hf-repo",
        default="keremberke/yolov8s-pothole-segmentation",
        help="HuggingFace repo id containing best.pt (default: keremberke yolov8s)",
    )
    p.add_argument(
        "--filename",
        default="best.pt",
        help="weight file inside the repo (most ultralytics-format repos use best.pt)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/checkpoints/seg"),
        help="destination directory",
    )
    p.add_argument(
        "--out-name",
        default=None,
        help="output filename (default: <repo_owner>_<repo_name>.pt)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required: `pip install huggingface_hub` "
            "(it's already part of transformers' deps in requirements.txt)."
        ) from e

    logging.info("downloading %s/%s ...", args.hf_repo, args.filename)
    cached = hf_hub_download(repo_id=args.hf_repo, filename=args.filename)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_name:
        out_name = args.out_name
    else:
        owner, name = args.hf_repo.split("/", 1)
        out_name = f"{owner}_{name}.pt"
    dest = args.out_dir / out_name
    shutil.copy2(cached, dest)
    logging.info("wrote %s (%d bytes)", dest, dest.stat().st_size)
    print(str(dest))


if __name__ == "__main__":
    main()
