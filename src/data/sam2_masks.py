"""SAM2 pseudo-mask generator for bounding boxes.

SAM2 is a heavy optional dependency. Install with:

    pip install "git+https://github.com/facebookresearch/sam2.git"

The predictor is loaded lazily so the data pipeline can also run in
``bbox-only`` mode without SAM2 installed.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .voc import Box


class SAM2MaskGenerator:
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        device: str = "cuda",
    ):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)

    def __call__(self, image_rgb: np.ndarray, boxes: Iterable[Box]) -> np.ndarray:
        """Return a (N, H, W) bool array — one mask per input box."""
        box_arr = np.asarray(
            [[b.xmin, b.ymin, b.xmax, b.ymax] for b in boxes], dtype=np.float32
        )
        if box_arr.size == 0:
            h, w = image_rgb.shape[:2]
            return np.zeros((0, h, w), dtype=bool)

        self.predictor.set_image(image_rgb)
        masks, _, _ = self.predictor.predict(
            box=box_arr, multimask_output=False
        )
        if masks.ndim == 4:          # (N, 1, H, W) → (N, H, W)
            masks = masks[:, 0]
        return masks.astype(bool)
