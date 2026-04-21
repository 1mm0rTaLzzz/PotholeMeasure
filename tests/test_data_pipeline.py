"""Tests for the Phase 2 data-prep pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.splits import box_count_bucket, stratified_split
from src.data.voc import POTHOLE_CLASS, Sample, parse_xml, scan_rdd
from src.data.yolo_rdd import scan_yolo_rdd

VOC_TEMPLATE = """<annotation>
  <folder>images</folder>
  <filename>{stem}.jpg</filename>
  <size>
    <width>{w}</width>
    <height>{h}</height>
    <depth>3</depth>
  </size>
  {objects}
</annotation>
"""

OBJ_TEMPLATE = """  <object>
    <name>{cls}</name>
    <bndbox>
      <xmin>{x0}</xmin><ymin>{y0}</ymin>
      <xmax>{x1}</xmax><ymax>{y1}</ymax>
    </bndbox>
  </object>"""


def _write_sample(rdd_root: Path, country: str, stem: str, objs: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    ann_dir = rdd_root / country / "train" / "annotations" / "xmls"
    img_dir = rdd_root / country / "train" / "images"
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    objects_xml = "\n".join(
        OBJ_TEMPLATE.format(cls=cls, x0=b[0], y0=b[1], x1=b[2], y1=b[3])
        for cls, b in objs
    )
    (ann_dir / f"{stem}.xml").write_text(
        VOC_TEMPLATE.format(stem=stem, w=640, h=480, objects=objects_xml)
    )
    # Minimal JPEG placeholder (content doesn't matter — scan never opens it).
    (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


def test_parse_xml_filters_pothole_class(tmp_path: Path) -> None:
    _write_sample(tmp_path, "Japan", "img_0001", [
        ("D00", (10, 10, 50, 50)),
        (POTHOLE_CLASS, (100, 100, 200, 200)),
        (POTHOLE_CLASS, (300, 300, 400, 400)),
    ])
    xml_path = tmp_path / "Japan/train/annotations/xmls/img_0001.xml"
    w, h, boxes, seen = parse_xml(xml_path)
    assert (w, h) == (640, 480)
    assert len(boxes) == 2
    assert boxes[0].xmin == 100
    assert boxes[1].xmax == 400
    assert seen == ["D00", POTHOLE_CLASS, POTHOLE_CLASS]


def test_scan_rdd_finds_flat_annotations_layout(tmp_path: Path) -> None:
    # Alt layout: annotations/*.xml (no xmls/ subdir) + sibling images/.
    ann_dir = tmp_path / "Japan" / "train" / "annotations"
    img_dir = tmp_path / "Japan" / "train" / "images"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    xml = VOC_TEMPLATE.format(
        stem="img_flat", w=640, h=480,
        objects=OBJ_TEMPLATE.format(cls=POTHOLE_CLASS, x0=10, y0=10, x1=40, y1=40),
    )
    (ann_dir / "img_flat.xml").write_text(xml)
    (img_dir / "img_flat.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    samples = scan_rdd(tmp_path)
    assert len(samples) == 1
    assert samples[0].image_path.name == "img_flat.jpg"


def test_scan_rdd_skips_images_without_potholes(tmp_path: Path) -> None:
    _write_sample(tmp_path, "Japan", "pos", [(POTHOLE_CLASS, (10, 10, 20, 20))])
    _write_sample(tmp_path, "India", "neg", [("D00", (0, 0, 10, 10))])
    _write_sample(tmp_path, "India", "pos2", [
        (POTHOLE_CLASS, (0, 0, 10, 10)),
        (POTHOLE_CLASS, (20, 20, 30, 30)),
        (POTHOLE_CLASS, (40, 40, 50, 50)),
    ])
    samples = scan_rdd(tmp_path)
    assert len(samples) == 2
    by_stem = {s.image_path.stem: s for s in samples}
    assert set(by_stem) == {"pos", "pos2"}
    assert len(by_stem["pos2"].boxes) == 3


def test_stratified_split_ratios_and_determinism() -> None:
    samples = [
        Sample(image_path=Path(f"img_{i}.jpg"), width=640, height=480,
               boxes=[object()] * ((i % 5) + 1))
        for i in range(200)
    ]
    train1, val1, test1 = stratified_split(samples, key=box_count_bucket, seed=7)
    train2, val2, test2 = stratified_split(samples, key=box_count_bucket, seed=7)
    assert [s.image_path for s in train1] == [s.image_path for s in train2]
    assert len(train1) + len(val1) + len(test1) == 200
    assert 0.65 * 200 <= len(train1) <= 0.75 * 200
    assert 0.10 * 200 <= len(val1) <= 0.20 * 200


def test_stratified_split_rejects_bad_ratios() -> None:
    with pytest.raises(ValueError):
        stratified_split([1, 2, 3], key=lambda x: x, ratios=(0.5, 0.3, 0.3))


def _write_yolo_sample(
    split_dir: Path,
    stem: str,
    lines: list[tuple[int, float, float, float, float]],
    image_size: tuple[int, int] = (640, 480),
) -> None:
    import cv2
    import numpy as np

    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    cv2.imwrite(str(split_dir / "images" / f"{stem}.jpg"), img)
    (split_dir / "labels" / f"{stem}.txt").write_text(
        "\n".join(f"{c} {cx} {cy} {bw} {bh}" for c, cx, cy, bw, bh in lines)
    )


def test_scan_yolo_rdd_filters_by_class_id(tmp_path: Path) -> None:
    # train: one image with class 3 + class 0, another with only class 0
    _write_yolo_sample(tmp_path / "train", "a", [(0, 0.5, 0.5, 0.1, 0.1), (3, 0.25, 0.5, 0.2, 0.4)])
    _write_yolo_sample(tmp_path / "train", "b", [(0, 0.5, 0.5, 0.2, 0.2)])
    # val: one image with class 3
    _write_yolo_sample(tmp_path / "val", "c", [(3, 0.6, 0.7, 0.1, 0.1)])

    splits = scan_yolo_rdd(tmp_path, target_class_id=3)
    assert [s.image_path.stem for s in splits["train"]] == ["a"]
    assert [s.image_path.stem for s in splits["val"]] == ["c"]
    assert splits["test"] == []

    # Denormalised bbox sanity: cx=0.25, bw=0.2, image_w=640 → x0=96, x1=224.
    box = splits["train"][0].boxes[0]
    assert abs(box.xmin - 96) < 1e-6 and abs(box.xmax - 224) < 1e-6
