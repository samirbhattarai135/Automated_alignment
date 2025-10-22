"""High-level orchestration for the projector mapping workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
from loguru import logger

from projector_mapper.alignment import ICPAligner, AlignmentResult  # ADD THIS
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
    alignment_results: Optional[Dict[str, AlignmentResult]] = None  # ADD THIS


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
        icp_aligner: ICPAligner,  # ADD THIS
    ) -> None:
        self._config = config
        self._detector = detection_backend
        self._mask_builder = mask_builder
        self._projector_mapper = projector_mapper
        self._overlay_tracker = overlay_tracker
        self._homography_refiner = homography_refiner
        self._icp_aligner = icp_aligner  # ADD THIS
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
        
        # Initialize ICP aligner with relaxed parameters
        icp_aligner = ICPAligner(
            max_iterations=50,          # More iterations for better convergence
            tolerance=0.5,              # Tighter tolerance for better alignment
            min_correspondences=4,       # Minimum for homography
            subsample_rate=2,           # More points for better coverage
            outlier_threshold=100.0,    # Very tolerant to handle full house
        )
        
        logger.info("Alignment pipeline initialized with %d projector(s)", len(calibrations))
        return cls(
            config=config,
            detection_backend=detection_backend,
            mask_builder=mask_builder,
            projector_mapper=projector_mapper,
            overlay_tracker=overlay_tracker,
            homography_refiner=homography_refiner,
            icp_aligner=icp_aligner,  # ADD THIS
        )

    def process_frame(self, frame_bgr: np.ndarray) -> PipelineOutput:
        """Process frame with detection, masking, and ICP alignment."""
        logger.debug("Processing frame of shape %s", frame_bgr.shape)
        
        # 1) Detect features
        detections = self._detector.detect(frame_bgr)
        
        # 2) Build masks in camera space
        masks = self._mask_builder.build(frame_bgr.shape[:2], detections.detections)
        
        # 3) Extract detected contours for ICP
        detected_contours = [det.polygon for det in detections.detections]
        
        # 4) Run ICP alignment and warp to projector space
        overlays: Dict[str, np.ndarray] = {}
        alignment_results: Dict[str, AlignmentResult] = {}
        
        for projector_id in self._projector_mapper.projector_ids():
            # Get current homography (camera → projector)
            current_H = self._projector_mapper.get_homography(projector_id)
            
            # Get projector resolution
            calib = self._projector_mapper._calibrations[projector_id]
            projector_shape = calib.resolution
            
            # Run ICP to find camera→camera alignment (starting from identity)
            # ICP computes: How should we transform the mask to match detections?
            alignment_result = self._icp_aligner.align(
                camera_frame=frame_bgr,
                overlay_mask=masks.composite_mask,
                detected_contours=detected_contours,
                initial_homography=None,  # Start from identity - align in camera space
            )
            
            # TEMPORARILY DISABLED: ICP refinement causing scale issues
            # Update homography if ICP converged
            if False and alignment_result.converged and alignment_result.correspondences >= 5:
                # Compose: new_H = calibration_H @ icp_H
                # This applies ICP correction before warping to projector
                refined_H = current_H @ alignment_result.homography
                
                self._projector_mapper.update_homography(
                    projector_id,
                    refined_H,
                    gain=0.5  # More aggressive update for better alignment
                )
                logger.debug(
                    f"ICP updated {projector_id}: error={alignment_result.error:.2f}px, "
                    f"correspondences={alignment_result.correspondences}"
                )
            
            # Render overlay using refined homography
            overlay = self._projector_mapper.render_overlay(masks.composite_mask, projector_id)
            
            overlays[projector_id] = overlay
            alignment_results[projector_id] = alignment_result
        
        # 5) Track overlays (optional feedback)
        tracking = self._overlay_tracker.track(frame_bgr)
        
        return PipelineOutput(
            detections=detections,
            masks=masks,
            overlays=overlays,
            tracking=tracking,
            alignment_results=alignment_results,  # ADD THIS
        )

    def _render_overlays(
        self,
        mask_result: MaskBuildResult,
        detections: Sequence[Detection],
    ) -> Dict[str, np.ndarray]:
        """Legacy method - now integrated into process_frame."""
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
        """Legacy feedback method - now using ICP."""
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
                logger.info(f"✓ Loaded calibrated homography for {projector.id}")
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
