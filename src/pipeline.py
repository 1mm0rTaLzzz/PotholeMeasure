"""End-to-end pothole detection and measurement pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from src.geometry.homography import (
    compute_homography_from_calibration,
    load_calibration,
    mask_to_area_m2,
)
from src.geometry.plane_fitting import (
    DepthOffsetStats,
    Plane,
    compute_depth_offset,
    depth_to_pointcloud,
    fit_road_plane,
)


@dataclass
class PotholeResult:
    bbox: tuple[float, float, float, float]
    confidence: float
    depth_m: float
    area_m2: float
    mask_area_px: int
    severity: Optional[str] = None
    depth_stats: Optional[DepthOffsetStats] = None
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    def to_json_dict(self) -> dict:
        out = {
            "bbox": [float(x) for x in self.bbox],
            "confidence": float(self.confidence),
            "depth_m": float(self.depth_m),
            "area_m2": float(self.area_m2),
            "mask_area_px": int(self.mask_area_px),
            "severity": self.severity,
        }
        if self.depth_stats is not None:
            out["depth_stats"] = asdict(self.depth_stats)
        return out


@dataclass
class FrameResult:
    image_path: Optional[Path]
    potholes: list[PotholeResult]
    plane: Optional[Plane]
    plane_fit_failed: bool

    def to_json_dict(self) -> dict:
        return {
            "image": str(self.image_path) if self.image_path else None,
            "plane_fit_failed": self.plane_fit_failed,
            "plane": None
            if self.plane is None
            else {
                "normal": self.plane.normal.tolist(),
                "d": self.plane.d,
                "inlier_ratio": self.plane.inlier_ratio,
                "n_inliers": self.plane.n_inliers,
            },
            "potholes": [p.to_json_dict() for p in self.potholes],
        }


class PotholePipeline:
    """Composes segmentation → depth → plane → area into a single call."""

    def __init__(
        self,
        segmentor,
        depth_estimator,
        K: np.ndarray,
        H_img2world: np.ndarray,
        plane_cfg: Optional[dict] = None,
        classifier: Optional[Callable[[float, float], str]] = None,
    ):
        self.segmentor = segmentor
        self.depth_estimator = depth_estimator
        self.K = np.asarray(K, dtype=np.float64)
        self.H_img2world = np.asarray(H_img2world, dtype=np.float64)
        self.plane_cfg = plane_cfg or {}
        self.classifier = classifier

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "PotholePipeline":
        import yaml

        cfg = yaml.safe_load(Path(config_path).read_text())
        K, calib = load_calibration(cfg["paths"]["calibration"])
        H = compute_homography_from_calibration(
            K,
            camera_height_m=float(calib["mount"]["height_m"]),
            pitch_deg=float(calib["mount"]["pitch_deg"]),
        )

        from src.models.depth import load_from_yaml_config as _load_depth
        from src.models.segmentation import PotholeSegmentor

        seg_cfg = cfg["segmentation"]
        weights = seg_cfg.get("finetuned_weights") or seg_cfg["model"]
        segmentor = PotholeSegmentor(
            weights=weights,
            device=cfg.get("depth", {}).get("device", "cuda"),
            conf=float(seg_cfg.get("conf_threshold", 0.35)),
            iou=float(seg_cfg.get("iou_threshold", 0.5)),
            imgsz=int(cfg["image"]["target_width"]),
        )
        depth_estimator = _load_depth(cfg)

        classifier = None
        sev_cfg = cfg.get("severity")
        if sev_cfg:
            from src.classifier import make_severity_classifier

            classifier = make_severity_classifier(sev_cfg["thresholds"])

        return cls(
            segmentor=segmentor,
            depth_estimator=depth_estimator,
            K=K,
            H_img2world=H,
            plane_cfg=cfg.get("plane_fitting", {}),
            classifier=classifier,
        )

    def process(
        self,
        image,
        image_path: Optional[Union[str, Path]] = None,
    ) -> FrameResult:
        detections = self.segmentor.predict(image)
        masks = [d.mask for d in detections]

        depth_map = self.depth_estimator.predict(image)
        h, w = depth_map.shape
        pc = depth_to_pointcloud(depth_map, self.K)

        plane = fit_road_plane(
            pc,
            (h, w),
            exclude_masks=masks,
            threshold_m=float(self.plane_cfg.get("ransac_threshold_m", 0.02)),
            max_iters=int(self.plane_cfg.get("max_iterations", 1000)),
            min_inlier_ratio=float(self.plane_cfg.get("min_inlier_ratio", 0.7)),
            min_points=int(self.plane_cfg.get("min_points", 1000)),
            up_axis=tuple(self.plane_cfg.get("up_axis", (0.0, -1.0, 0.0))),
            up_cos_threshold=float(self.plane_cfg.get("up_cos_threshold", 0.9)),
        )

        potholes: list[PotholeResult] = []
        for det in detections:
            stats = compute_depth_offset(plane, pc, det.mask) if plane is not None else None
            depth_m = stats.p95_m if stats is not None else 0.0
            area_m2 = mask_to_area_m2(det.mask, self.H_img2world)
            severity = self.classifier(depth_m, area_m2) if self.classifier else None
            potholes.append(
                PotholeResult(
                    bbox=tuple(float(x) for x in det.bbox),
                    confidence=float(det.score),
                    depth_m=float(depth_m),
                    area_m2=float(area_m2),
                    mask_area_px=int(det.mask.sum()),
                    severity=severity,
                    depth_stats=stats,
                    mask=det.mask,
                )
            )

        return FrameResult(
            image_path=Path(image_path) if image_path else None,
            potholes=potholes,
            plane=plane,
            plane_fit_failed=plane is None,
        )
