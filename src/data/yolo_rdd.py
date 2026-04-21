"""Reader for RDD2022 in YOLO layout with pre-made splits.

Expected structure::

    <rdd_root>/
      train/{images,labels}/*
      val/{images,labels}/*
      test/{images,labels}/*

Each label file is YOLO-format: one object per line,
``<class_id> <x_center> <y_center> <w> <h>`` (all normalized to [0, 1]).
"""
from __future__ import annotations

import logging
from pathlib import Path

from .voc import Box, Sample


SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def _image_for_label(label_path: Path) -> Path | None:
    stem = label_path.stem
    images_dir = label_path.parent.parent / "images"
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _image_size(path: Path) -> tuple[int, int] | None:
    """(width, height) via cv2; returns None if the file can't be decoded."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None
    h, w = image.shape[:2]
    return w, h


def _parse_label_file(
    path: Path,
    image_w: int,
    image_h: int,
    target_class_id: int,
) -> tuple[list[Box], dict[int, int]]:
    boxes: list[Box] = []
    seen: dict[int, int] = {}
    for raw in path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        seen[cls_id] = seen.get(cls_id, 0) + 1
        if cls_id != target_class_id:
            continue
        x0 = (cx - bw / 2.0) * image_w
        y0 = (cy - bh / 2.0) * image_h
        x1 = (cx + bw / 2.0) * image_w
        y1 = (cy + bh / 2.0) * image_h
        boxes.append(Box(xmin=x0, ymin=y0, xmax=x1, ymax=y1))
    return boxes, seen


def scan_yolo_split(
    split_dir: Path,
    target_class_id: int,
    class_counter: dict[int, int] | None = None,
) -> list[Sample]:
    logger = logging.getLogger(__name__)
    labels_dir = split_dir / "labels"
    if not labels_dir.is_dir():
        return []

    samples: list[Sample] = []
    n_labels = 0
    n_missing_image = 0
    n_decode_err = 0
    local_counter: dict[int, int] = {}

    for label_path in sorted(labels_dir.glob("*.txt")):
        n_labels += 1
        image_path = _image_for_label(label_path)
        if image_path is None:
            n_missing_image += 1
            continue
        size = _image_size(image_path)
        if size is None:
            n_decode_err += 1
            continue
        w, h = size
        boxes, seen = _parse_label_file(label_path, w, h, target_class_id)
        for k, v in seen.items():
            local_counter[k] = local_counter.get(k, 0) + v
        if not boxes:
            continue
        samples.append(Sample(image_path=image_path, width=w, height=h, boxes=boxes))

    if class_counter is not None:
        for k, v in local_counter.items():
            class_counter[k] = class_counter.get(k, 0) + v

    logger.info(
        "YOLO split %s: labels=%d target_cls=%d samples=%d missing_img=%d decode_err=%d classes_seen=%s",
        split_dir.name, n_labels, target_class_id, len(samples),
        n_missing_image, n_decode_err, sorted(local_counter.items()),
    )
    return samples


def scan_yolo_rdd(
    rdd_root: Path,
    target_class_id: int,
) -> dict[str, list[Sample]]:
    """Read pre-split YOLO labels. Returns ``{'train': [...], 'val': [...], 'test': [...]}``.

    Missing splits are returned as empty lists (e.g. dataset without ``test/``).
    """
    logger = logging.getLogger(__name__)
    class_counter: dict[int, int] = {}
    out: dict[str, list[Sample]] = {}
    for name in SPLIT_NAMES:
        out[name] = scan_yolo_split(rdd_root / name, target_class_id, class_counter)
    total = sum(len(v) for v in out.values())
    logger.info(
        "YOLO scan total: target_cls=%d samples=%d per_split=%s full_class_distribution=%s",
        target_class_id, total, {k: len(v) for k, v in out.items()}, sorted(class_counter.items()),
    )
    return out
