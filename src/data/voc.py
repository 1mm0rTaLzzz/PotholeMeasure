"""VOC XML parsing for RDD2022.

RDD2022 layout:
    <rdd_root>/<country>/train/annotations/xmls/*.xml
    <rdd_root>/<country>/train/images/*.jpg
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


POTHOLE_CLASS = "D40"


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass
class Sample:
    image_path: Path
    width: int
    height: int
    boxes: list[Box]


def parse_xml(xml_path: Path, target_class: str = POTHOLE_CLASS) -> tuple[int, int, list[Box]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))
    boxes: list[Box] = []
    for obj in root.findall("object"):
        if obj.findtext("name") != target_class:
            continue
        bnd = obj.find("bndbox")
        boxes.append(Box(
            xmin=float(bnd.findtext("xmin")),
            ymin=float(bnd.findtext("ymin")),
            xmax=float(bnd.findtext("xmax")),
            ymax=float(bnd.findtext("ymax")),
        ))
    return w, h, boxes


def find_image_for_xml(xml_path: Path) -> Path | None:
    stem = xml_path.stem
    # annotations/xmls/<stem>.xml  →  images/<stem>.jpg
    candidate_root = xml_path.parents[1].parent / "images"
    for ext in (".jpg", ".jpeg", ".png"):
        p = candidate_root / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def scan_rdd(rdd_root: Path, target_class: str = POTHOLE_CLASS) -> list[Sample]:
    samples: list[Sample] = []
    xml_paths = sorted(rdd_root.rglob("annotations/xmls/*.xml"))
    for xml_path in xml_paths:
        image_path = find_image_for_xml(xml_path)
        if image_path is None:
            continue
        try:
            w, h, boxes = parse_xml(xml_path, target_class=target_class)
        except (ET.ParseError, ValueError, TypeError):
            continue
        if not boxes:
            continue
        samples.append(Sample(image_path=image_path, width=w, height=h, boxes=boxes))
    return samples
