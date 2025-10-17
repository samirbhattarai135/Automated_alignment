"""High-level orchestration for the projector mapping workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from loguru import logger

from projector_mapper.calibration import CalibrationManager
from projector_mapper.config import PipelineConfig
from projector_mapper.detection import (
    Detection,
    DetectionBackend,
    DetectionResult,
    FalEvfSamDetector,
    RoboflowDetector,
)
from projector_mapper.feedback import HomographyRefiner, OverlayTracker, TrackingResult
from projector_mapper.masking import MaskBuilder, MaskBuildResult
from projector_mapper.projection import ProjectorCalibration, ProjectorMapper


@dataclass(slots=True)
class PipelineOutput:
    detections: DetectionResult
    masks: MaskBuildResult
    overlays: Dict[str, np.ndarray]
    tracking: Optional[TrackingResult]


class AlignmentPipeline:
    """Manages detection, masking, and projection alignment for live systems."""

    def __init__(
        self,
        config: PipelineConfig,
        detection_backend: DetectionBackend,
        mask_builder: MaskBuilder,
        projector_mapper: ProjectorMapper,
        overlay_tracker: OverlayTracker,
        homography_refiner: HomographyRefiner,
    ) -> None:
        self._config = config
        self._detector = detection_backend
        self._mask_builder = mask_builder
        self._projector_mapper = projector_mapper
        self._overlay_tracker = overlay_tracker
        self._homography_refiner = homography_refiner
        self._last_projector_polygons: Dict[str, List[np.ndarray]] = {}

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "AlignmentPipeline":
        detection_backend = cls._build_detector(config)
        mask_builder = MaskBuilder(
            dilate_kernel=config.pipeline.masking.dilate_kernel,
            blur_kernel=config.pipeline.masking.blur_kernel,
        )
        calibrations = cls._load_projector_calibrations(config)
        projector_mapper = ProjectorMapper(calibrations)
        overlay_tracker = OverlayTracker(
            projectors=config.hardware.projectors,
            min_contour_area=config.pipeline.alignment.min_overlay_contour_area,
        )
        homography_refiner = HomographyRefiner(
            mapper=projector_mapper,
            feedback_gain=config.pipeline.alignment.feedback_gain,
            max_polygon_points=config.pipeline.alignment.max_polygon_points,
        )
        logger.info("Alignment pipeline initialized with %d projector(s)", len(calibrations))
        return cls(
            config=config,
            detection_backend=detection_backend,
            mask_builder=mask_builder,
            projector_mapper=projector_mapper,
            overlay_tracker=overlay_tracker,
            homography_refiner=homography_refiner,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> PipelineOutput:
        logger.debug("Processing frame of shape %s", frame_bgr.shape)
        detections = self._detector.detect(frame_bgr)
        masks = self._mask_builder.build(frame_bgr.shape[:2], detections.detections)
        overlays = self._render_overlays(masks, detections.detections)
        tracking = self._overlay_tracker.track(frame_bgr)
        # Homography feedback disabled - only running computer vision detection
        # self._apply_feedback(tracking)
        return PipelineOutput(detections=detections, masks=masks, overlays=overlays, tracking=tracking)

    def _render_overlays(
        self,
        mask_result: MaskBuildResult,
        detections: Sequence[Detection],
    ) -> Dict[str, np.ndarray]:
        overlays: Dict[str, np.ndarray] = {}
        projector_polygons: Dict[str, List[np.ndarray]] = {}
        for projector_id in self._projector_mapper.projector_ids():
            overlay = self._projector_mapper.render_overlay(mask_result.composite_mask, projector_id)
            overlays[projector_id] = overlay
            warped_polys: List[np.ndarray] = []
            for detection in detections:
                polygon = getattr(detection, "polygon", None)
                if polygon is None:
                    continue
                warped_polys.append(self._projector_mapper.project_polygon(polygon, projector_id))
            projector_polygons[projector_id] = warped_polys
        self._last_projector_polygons = projector_polygons
        return overlays

    def _apply_feedback(self, tracking: TrackingResult) -> None:
        for projector_id, camera_polygons in tracking.polygons.items():
            projector_polygons = self._last_projector_polygons.get(projector_id, [])
            if not camera_polygons or not projector_polygons:
                continue
            self._homography_refiner.refine(
                projector_id=projector_id,
                camera_polygons=camera_polygons,
                projector_polygons=projector_polygons,
            )

    @staticmethod
    def _build_detector(config: PipelineConfig) -> DetectionBackend:
        detection_cfg = config.pipeline.detection
        backend = detection_cfg.backend.lower()
        if backend == "roboflow":
            return RoboflowDetector(
                endpoint=detection_cfg.api.endpoint,
                api_key_env=detection_cfg.api.api_key_env,
                classes=detection_cfg.classes,
                confidence_threshold=detection_cfg.confidence_threshold,
            )
        if backend == "fal_evfsam":
            if detection_cfg.evfsam is None:
                raise ValueError("EVF-SAM configuration is required for fal_evfsam backend")
            evf = detection_cfg.evfsam
            return FalEvfSamDetector(
                classes=detection_cfg.classes,
                confidence_threshold=detection_cfg.confidence_threshold,
                model_id=evf.model_id,
                api_key_env=evf.api_key_env,
                prompt_map=evf.prompt_map,
                mask_only=evf.mask_only,
                fill_holes=evf.fill_holes,
                revert_mask=evf.revert_mask,
                poll_interval_s=evf.poll_interval_s,
                timeout_s=evf.timeout_s,
            )
        raise ValueError(f"Unsupported detection backend: {backend}")

    @staticmethod
    def _load_projector_calibrations(config: PipelineConfig) -> Iterable[ProjectorCalibration]:
        manager = CalibrationManager(config.calibration, config.hardware.projectors)
        calibrations = []
        for projector in config.hardware.projectors:
            try:
                homography = manager.load_homography(projector.id)
            except FileNotFoundError:
                logger.warning(
                    "Homography for %s not found; starting from identity and relying on live refinement",
                    projector.id,
                )
                homography = np.eye(3, dtype=np.float32)
            calibrations.append(
                ProjectorCalibration(
                    id=projector.id,
                    homography=homography,
                    resolution=(projector.width, projector.height),
                    color_bgr=tuple(int(value) for value in projector.overlay_color_bgr),
                )
            )
        return calibrations
