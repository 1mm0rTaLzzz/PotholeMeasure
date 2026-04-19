"""COCO JSON assembly for the pothole dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .voc import Box, Sample

POTHOLE_CATEGORY = {"id": 1, "name": "pothole", "supercategory": "road_damage"}


def resize_image_and_boxes(image, boxes: Sequence[Box], target_w: int, target_h: int):
    import cv2

    src_h, src_w = image.shape[:2]
    sx = target_w / src_w
    sy = target_h / src_h
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    scaled = [Box(b.xmin * sx, b.ymin * sy, b.xmax * sx, b.ymax * sy) for b in boxes]
    return resized, scaled


def mask_to_polygons(mask, min_area: float = 4.0) -> list[list[float]]:
    """Convert binary mask to COCO polygon ``segmentation``."""
    import cv2
    import numpy as np

    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    polygons: list[list[float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        pts = cnt.reshape(-1, 2)
        if len(pts) < 3:
            continue
        polygons.append(pts.flatten().astype(float).tolist())
    return polygons


def build_coco(
    samples: Iterable[Sample],
    target_size: tuple[int, int],
    out_images_dir: Path,
    mask_generator=None,
    progress_desc: str = "build",
) -> dict:
    import cv2
    from tqdm import tqdm

    out_images_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    target_w, target_h = target_size

    for img_id, sample in enumerate(tqdm(list(samples), desc=progress_desc), start=1):
        image = cv2.imread(str(sample.image_path))
        if image is None:
            continue
        resized, boxes = resize_image_and_boxes(image, sample.boxes, target_w, target_h)
        out_name = f"{img_id:07d}.jpg"
        cv2.imwrite(str(out_images_dir / out_name), resized)

        images.append(
            {
                "id": img_id,
                "file_name": out_name,
                "width": target_w,
                "height": target_h,
            }
        )

        masks = None
        if mask_generator is not None and boxes:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            masks = mask_generator(rgb, boxes)

        for i, box in enumerate(boxes):
            w = box.width
            h = box.height
            area = float(w * h)
            segmentation: list = []
            if masks is not None and i < len(masks):
                polys = mask_to_polygons(masks[i])
                if polys:
                    segmentation = polys
                    area = float(masks[i].sum())
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": POTHOLE_CATEGORY["id"],
                    "bbox": [box.xmin, box.ymin, w, h],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )
            ann_id += 1

    return {
        "info": {"description": "RDD2022 potholes (D40)"},
        "images": images,
        "annotations": annotations,
        "categories": [POTHOLE_CATEGORY],
    }


def write_coco(coco: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco))
