"""Tests for COCO → YOLO-seg conversion."""
from __future__ import annotations

import json
from pathlib import Path

from src.data.yolo import coco_to_yolo, write_data_yaml


def _make_fake_coco(tmp_path: Path) -> tuple[Path, Path]:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (images_dir / name).write_bytes(b"\xff\xd8\xff\xd9")

    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 200},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 200},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0,
                "segmentation": [[10, 20, 40, 20, 40, 60, 10, 60]],
            },
            {
                "id": 2, "image_id": 1, "category_id": 1,
                "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0,
                "segmentation": [],  # no polygon → should be skipped
            },
            {
                "id": 3, "image_id": 2, "category_id": 1,
                "bbox": [0, 0, 100, 200], "area": 20000, "iscrowd": 0,
                "segmentation": [[0, 0, 100, 0, 100, 200, 0, 200]],
            },
        ],
        "categories": [{"id": 1, "name": "pothole"}],
    }
    coco_path = tmp_path / "train.json"
    coco_path.write_text(json.dumps(coco))
    return coco_path, images_dir


def test_coco_to_yolo_writes_normalised_polygons(tmp_path: Path) -> None:
    coco_path, images_dir = _make_fake_coco(tmp_path)
    out_images = tmp_path / "yolo/images/train"
    out_labels = tmp_path / "yolo/labels/train"

    n_lines = coco_to_yolo(
        coco_json=coco_path,
        images_dir=images_dir,
        out_images_dir=out_images,
        out_labels_dir=out_labels,
        class_id=0,
        symlink_images=False,
    )

    assert n_lines == 2
    assert (out_images / "a.jpg").exists()
    assert (out_images / "b.jpg").exists()

    a_label = (out_labels / "a.txt").read_text().strip().splitlines()
    assert len(a_label) == 1
    parts = a_label[0].split()
    assert parts[0] == "0"
    coords = [float(x) for x in parts[1:]]
    # Polygon was (10,20)-(40,20)-(40,60)-(10,60) normalised by (100, 200).
    assert coords == [0.10, 0.10, 0.40, 0.10, 0.40, 0.30, 0.10, 0.30]

    b_label = (out_labels / "b.txt").read_text().strip().splitlines()
    b_coords = [float(x) for x in b_label[0].split()[1:]]
    # Normalised full-image rectangle → all values 0 or 1.
    assert all(v in (0.0, 1.0) for v in b_coords)


def test_coco_to_yolo_clamps_out_of_bounds(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "c.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    coco = {
        "images": [{"id": 1, "file_name": "c.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [-10, -10, 200, 200], "area": 1, "iscrowd": 0,
                "segmentation": [[-10, -10, 150, -10, 150, 150, -10, 150]],
            }
        ],
        "categories": [{"id": 1, "name": "pothole"}],
    }
    coco_path = tmp_path / "train.json"
    coco_path.write_text(json.dumps(coco))

    coco_to_yolo(coco_path, images_dir, tmp_path / "img", tmp_path / "lab", symlink_images=False)
    coords = [float(x) for x in (tmp_path / "lab/c.txt").read_text().split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in coords)


def test_write_data_yaml(tmp_path: Path) -> None:
    yolo_root = tmp_path / "yolo"
    yolo_root.mkdir()
    write_data_yaml(yolo_root / "data.yaml", yolo_root)
    content = (yolo_root / "data.yaml").read_text()
    assert "train: images/train" in content
    assert "0: pothole" in content
    assert str(yolo_root.resolve()) in content
