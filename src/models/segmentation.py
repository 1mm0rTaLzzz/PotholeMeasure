"""Instance-segmentation wrapper around Ultralytics YOLOv8-seg."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union


ImageLike = Union["np.ndarray", str, Path]  # type: ignore[name-defined]


@dataclass
class PotholeDetection:
    """A single pothole prediction."""
    mask: "np.ndarray"       # type: ignore[name-defined]  # (H, W) bool
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) pixels
    score: float


class PotholeSegmentor:
    """Thin wrapper around Ultralytics YOLO for pothole segmentation.

    Deliberately keeps Ultralytics imports lazy so the module is importable
    without the heavy dependency (useful for CLI --help, unit tests that mock
    the backend, etc.).
    """

    def __init__(
        self,
        weights: Union[str, Path] = "yolov8m-seg.pt",
        device: str = "cuda",
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 1280,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def load_finetuned(self, weights: Union[str, Path]) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))

    def predict(self, image: ImageLike) -> list[PotholeDetection]:
        import numpy as np

        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            return []

        masks = result.masks.data.cpu().numpy().astype(bool)       # (N, H, W)
        boxes = result.boxes.xyxy.cpu().numpy()                     # (N, 4)
        scores = result.boxes.conf.cpu().numpy()                    # (N,)

        # Resize masks back to original image size if ultralytics padded/rescaled.
        orig_h, orig_w = result.orig_shape
        if masks.shape[1:] != (orig_h, orig_w):
            import cv2

            resized = np.zeros((masks.shape[0], orig_h, orig_w), dtype=bool)
            for i, m in enumerate(masks):
                resized[i] = cv2.resize(
                    m.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            masks = resized

        return [
            PotholeDetection(mask=masks[i], bbox=tuple(float(x) for x in boxes[i]), score=float(scores[i]))
            for i in range(len(masks))
        ]

    def predict_batch(self, images: Iterable[ImageLike]) -> list[list[PotholeDetection]]:
        return [self.predict(img) for img in images]
