"""Tests for the Phase 2 data-prep pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.splits import box_count_bucket, stratified_split
from src.data.voc import POTHOLE_CLASS, Sample, parse_xml, scan_rdd

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
    w, h, boxes = parse_xml(xml_path)
    assert (w, h) == (640, 480)
    assert len(boxes) == 2
    assert boxes[0].xmin == 100
    assert boxes[1].xmax == 400


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
