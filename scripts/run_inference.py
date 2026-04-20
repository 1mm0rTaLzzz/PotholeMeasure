"""Run the full pothole pipeline on one image or a folder of images.

Example:
    python scripts/run_inference.py \\
        --config configs/default.yaml \\
        --input path/to/image_or_folder \\
        --output experiments/results
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--input", type=Path, required=True, help="image or directory")
    p.add_argument("--output", type=Path, default=Path("experiments/results"))
    p.add_argument("--no-viz", action="store_true", help="skip visualisation PNG output")
    return p.parse_args()


def iter_inputs(path: Path):
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise SystemExit(f"input not found: {path}")
    for p in sorted(path.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXT:
            yield p


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import cv2

    from src.pipeline import PotholePipeline
    from src.visualize import draw_results

    pipeline = PotholePipeline.from_config(args.config)

    args.output.mkdir(parents=True, exist_ok=True)
    viz_dir = args.output / "viz"
    json_dir = args.output / "json"
    viz_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)

    summary = []
    for img_path in iter_inputs(args.input):
        image = cv2.imread(str(img_path))
        if image is None:
            logging.warning("could not read %s", img_path)
            continue
        frame = pipeline.process(image, image_path=img_path)
        payload = frame.to_json_dict()
        (json_dir / f"{img_path.stem}.json").write_text(json.dumps(payload, indent=2))

        if not args.no_viz:
            vis = draw_results(image, frame.potholes)
            cv2.imwrite(str(viz_dir / f"{img_path.stem}.jpg"), vis)

        summary.append({"image": str(img_path), "n_potholes": len(frame.potholes)})
        logging.info("%s → %d potholes", img_path.name, len(frame.potholes))

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("wrote summary with %d entries to %s", len(summary), args.output / "summary.json")


if __name__ == "__main__":
    main()
