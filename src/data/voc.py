"""VOC XML parsing for RDD2022.

Supported layouts (the scanner tries both):
    <rdd_root>/**/annotations/xmls/*.xml  +  <rdd_root>/**/images/*.jpg
    <rdd_root>/**/annotations/*.xml        +  <rdd_root>/**/images/*.jpg
"""
from __future__ import annotations

import logging
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


def parse_xml(xml_path: Path, target_class: str = POTHOLE_CLASS) -> tuple[int, int, list[Box], list[str]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))
    boxes: list[Box] = []
    seen_classes: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or ""
        seen_classes.append(name)
        if name != target_class:
            continue
        bnd = obj.find("bndbox")
        boxes.append(Box(
            xmin=float(bnd.findtext("xmin")),
            ymin=float(bnd.findtext("ymin")),
            xmax=float(bnd.findtext("xmax")),
            ymax=float(bnd.findtext("ymax")),
        ))
    return w, h, boxes, seen_classes


def find_image_for_xml(xml_path: Path) -> Path | None:
    """Locate the matching image regardless of whether XMLs live under
    ``annotations/xmls/`` or directly under ``annotations/``.
    """
    stem = xml_path.stem
    # Walk up until we find a sibling "images" directory.
    for parent in xml_path.parents:
        candidate = parent / "images"
        if candidate.is_dir():
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                p = candidate / f"{stem}{ext}"
                if p.exists():
                    return p
            break
    return None


def _collect_xml_paths(rdd_root: Path) -> list[Path]:
    patterns = ("annotations/xmls/*.xml", "annotations/*.xml")
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in patterns:
        for p in rdd_root.rglob(pattern):
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            out.append(p)
    return sorted(out)


def scan_rdd(rdd_root: Path, target_class: str = POTHOLE_CLASS) -> list[Sample]:
    logger = logging.getLogger(__name__)
    samples: list[Sample] = []
    xml_paths = _collect_xml_paths(rdd_root)

    n_xml = len(xml_paths)
    n_parse_err = 0
    n_no_image = 0
    n_target = 0
    class_counter: dict[str, int] = {}

    for xml_path in xml_paths:
        try:
            w, h, boxes, seen = parse_xml(xml_path, target_class=target_class)
        except (ET.ParseError, ValueError, TypeError):
            n_parse_err += 1
            continue
        for c in seen:
            class_counter[c] = class_counter.get(c, 0) + 1
        if not boxes:
            continue
        image_path = find_image_for_xml(xml_path)
        if image_path is None:
            n_no_image += 1
            continue
        n_target += 1
        samples.append(Sample(image_path=image_path, width=w, height=h, boxes=boxes))

    logger.info(
        "VOC scan: root=%s xmls=%d parse_err=%d target=%r samples_with_%s=%d missing_image=%d",
        rdd_root, n_xml, n_parse_err, target_class, target_class, n_target, n_no_image,
    )
    if class_counter:
        top = sorted(class_counter.items(), key=lambda kv: -kv[1])[:10]
        logger.info("class distribution (top 10 of %d): %s", len(class_counter), top)
    elif n_xml == 0:
        logger.warning(
            "no XMLs matched under %s — expected annotations/xmls/*.xml or annotations/*.xml",
            rdd_root,
        )
    return samples
