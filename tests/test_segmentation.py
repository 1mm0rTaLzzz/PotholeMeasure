"""Tests for the PotholeSegmentor wrapper (Ultralytics YOLO mocked)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


class _FakeTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)


class _FakeMasks:
    def __init__(self, data: np.ndarray):
        self.data = _FakeTensor(data)

    def __len__(self):
        return len(self.data._arr)


class _FakeResult:
    def __init__(self, masks: np.ndarray | None, boxes: np.ndarray, confs: np.ndarray, orig_shape=(720, 1280)):
        self.masks = None if masks is None else _FakeMasks(masks)
        self.boxes = _FakeBoxes(boxes, confs)
        self.orig_shape = orig_shape


class _FakeYOLO:
    last_weights: str | None = None
    next_result: _FakeResult | None = None

    def __init__(self, weights: str):
        _FakeYOLO.last_weights = weights

    def predict(self, **kwargs):
        if _FakeYOLO.next_result is None:
            return []
        return [_FakeYOLO.next_result]

    def train(self, **kwargs):
        return types.SimpleNamespace(save_dir=Path("/tmp/noop"))


@pytest.fixture
def fake_ultralytics(monkeypatch):
    module = types.ModuleType("ultralytics")
    module.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    _FakeYOLO.last_weights = None
    _FakeYOLO.next_result = None
    yield
    _FakeYOLO.next_result = None


def test_segmentor_loads_weights(fake_ultralytics) -> None:
    from src.models.segmentation import PotholeSegmentor

    seg = PotholeSegmentor(weights="yolov8m-seg.pt", device="cpu")
    assert _FakeYOLO.last_weights == "yolov8m-seg.pt"

    seg.load_finetuned("experiments/best.pt")
    assert _FakeYOLO.last_weights == "experiments/best.pt"


def test_segmentor_returns_empty_for_no_detections(fake_ultralytics) -> None:
    from src.models.segmentation import PotholeSegmentor

    seg = PotholeSegmentor(weights="yolov8m-seg.pt", device="cpu")
    _FakeYOLO.next_result = _FakeResult(
        masks=None, boxes=np.zeros((0, 4)), confs=np.zeros(0)
    )
    detections = seg.predict("img.jpg")
    assert detections == []


def test_segmentor_parses_masks_and_scores(fake_ultralytics) -> None:
    from src.models.segmentation import PotholeSegmentor

    seg = PotholeSegmentor(weights="yolov8m-seg.pt", device="cpu")
    masks = np.stack([
        np.zeros((720, 1280), dtype=np.uint8),
        np.ones((720, 1280), dtype=np.uint8),
    ])
    boxes = np.array([[10.0, 20.0, 50.0, 60.0], [100.0, 200.0, 300.0, 400.0]])
    confs = np.array([0.9, 0.6])
    _FakeYOLO.next_result = _FakeResult(masks=masks, boxes=boxes, confs=confs)

    detections = seg.predict("img.jpg")
    assert len(detections) == 2
    assert detections[0].score == pytest.approx(0.9)
    assert detections[0].bbox == (10.0, 20.0, 50.0, 60.0)
    assert detections[0].mask.dtype == np.bool_
    assert detections[0].mask.shape == (720, 1280)
    assert detections[1].mask.all()
