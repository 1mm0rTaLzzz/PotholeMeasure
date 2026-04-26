"""Convert a COCO segmentation JSON into the Ultralytics YOLO-seg layout.

YOLO-seg expects per image a .txt label file with one line per instance:

    <class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...

All coordinates are normalised to [0, 1] by image width/height.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def _flatten_polygon(coords: list[float], width: int, height: int) -> list[float]:
    out: list[float] = []
    for i, v in enumerate(coords):
        norm = v / width if i % 2 == 0 else v / height
        out.append(max(0.0, min(1.0, norm)))
    return out


def _materialise_image(src: Path, dst: Path, prefer_symlink: bool) -> None:
    """Place ``src`` at ``dst``. Tries symlink (cheap) then falls back to copy.

    Windows refuses symlink creation without Developer Mode / admin
    (``WinError 1314``); silently fall back to ``shutil.copy2`` there.
    A broken symlink left over from a previous failed run is replaced.
    """
    if dst.is_symlink() and not dst.exists():
        dst.unlink()
    if dst.exists():
        return
    if prefer_symlink:
        try:
            dst.symlink_to(src.resolve())
            return
        except (OSError, NotImplementedError):
            pass  # fall through to copy
    shutil.copy2(src, dst)


def coco_to_yolo(
    coco_json: Path,
    images_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    *,
    class_id: int = 0,
    symlink_images: bool = True,
) -> int:
    """Emit YOLO-seg labels + image folder. Returns number of annotations written."""
    coco = json.loads(Path(coco_json).read_text())
    images = {img["id"]: img for img in coco["images"]}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    for img_id, img in images.items():
        src = images_dir / img["file_name"]
        if not src.exists():
            continue
        dst = out_images_dir / img["file_name"]
        _materialise_image(src, dst, prefer_symlink=symlink_images)

        label_path = out_labels_dir / (Path(img["file_name"]).stem + ".txt")
        lines: list[str] = []
        for ann in anns_by_img.get(img_id, []):
            segmentation = ann.get("segmentation") or []
            if not segmentation:
                continue
            # Use the largest polygon (COCO can hold multiple per ann).
            polygon = max(segmentation, key=len)
            if len(polygon) < 6:
                continue
            flat = _flatten_polygon(polygon, img["width"], img["height"])
            values = " ".join(f"{v:.6f}" for v in flat)
            lines.append(f"{class_id} {values}")
        label_path.write_text("\n".join(lines))
        total_lines += len(lines)

    return total_lines


def write_data_yaml(
    out_path: Path,
    yolo_root: Path,
    class_name: str = "pothole",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"path: {yolo_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
