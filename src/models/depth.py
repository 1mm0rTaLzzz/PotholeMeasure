"""Metric monocular depth estimation via Depth Anything V2.

Wraps the HuggingFace ``depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf``
model to produce a per-pixel depth map in metres. Heavy dependencies (torch,
transformers) are imported lazily so callers that just want ``--help`` or unit
tests don't pay the cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

ImageLike = Union["np.ndarray", "PIL.Image.Image", str, Path]  # type: ignore[name-defined]


@dataclass
class DepthConfig:
    model_id: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
    device: str = "cuda"
    min_depth_m: float = 0.5
    max_depth_m: float = 80.0
    use_half: bool = False          # fp16 on GPU for speed.


def to_pil(image: ImageLike):
    """Coerce the input into an RGB :class:`PIL.Image`."""
    from PIL import Image
    import numpy as np

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 array, got shape {image.shape}")
        # Assume OpenCV BGR input — flip to RGB for PIL.
        return Image.fromarray(image[:, :, ::-1])
    raise TypeError(f"unsupported image type: {type(image)!r}")


def clip_depth(depth, min_m: float, max_m: float):
    """In-place clip depth to [min_m, max_m] and cast to float32."""
    import numpy as np

    depth = np.asarray(depth, dtype=np.float32)
    np.clip(depth, min_m, max_m, out=depth)
    return depth


class MetricDepthEstimator:
    """Predicts a metric-depth map (float32, metres) the same size as the input."""

    def __init__(self, config: DepthConfig | None = None):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch

        self.config = config or DepthConfig()
        self._torch = torch

        self.processor = AutoImageProcessor.from_pretrained(self.config.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.config.model_id)
        self.model.to(self.config.device).eval()
        if self.config.use_half and self.config.device.startswith("cuda"):
            self.model.half()

    def predict(self, image: ImageLike) -> "np.ndarray":
        pil = to_pil(image)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        if self.config.use_half and self.config.device.startswith("cuda"):
            inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self.model(**inputs)

        predicted = outputs.predicted_depth
        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)          # (B, 1, H, W)
        resized = self._torch.nn.functional.interpolate(
            predicted.float(),
            size=pil.size[::-1],                        # (H, W)
            mode="bicubic",
            align_corners=False,
        )
        return clip_depth(
            resized.squeeze().cpu().numpy(),
            self.config.min_depth_m,
            self.config.max_depth_m,
        )

    def predict_batch(self, images) -> list["np.ndarray"]:
        return [self.predict(img) for img in images]


def load_from_yaml_config(cfg: dict) -> MetricDepthEstimator:
    """Build an estimator from the ``depth`` block of ``configs/default.yaml``."""
    d = cfg.get("depth", {})
    return MetricDepthEstimator(DepthConfig(
        model_id=d.get("model_id", DepthConfig.model_id),
        device=d.get("device", DepthConfig.device),
        min_depth_m=float(d.get("min_depth_m", DepthConfig.min_depth_m)),
        max_depth_m=float(d.get("max_depth_m", DepthConfig.max_depth_m)),
        use_half=bool(d.get("use_half", DepthConfig.use_half)),
    ))
