"""Estimate updated camera-to-projector homographies from feedback observations."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

from projector_mapper.projection.mapper import ProjectorMapper


class HomographyRefiner:
    """Refines projector homographies using observed overlay polygons."""

    def __init__(
        self,
        mapper: ProjectorMapper,
        feedback_gain: float,
        max_polygon_points: int,
    ) -> None:
        self._mapper = mapper
        self._gain = feedback_gain
        self._max_points = max(4, max_polygon_points)

    def refine(
        self,
        projector_id: str,
        camera_polygons: Sequence[np.ndarray],
        projector_polygons: Sequence[np.ndarray],
    ) -> None:
        if not camera_polygons or not projector_polygons:
            return

        cam_sorted = self._sort_polygons(camera_polygons)
        proj_sorted = self._sort_polygons(projector_polygons)
        paired = zip(cam_sorted, proj_sorted)

        src_points: List[np.ndarray] = []
        dst_points: List[np.ndarray] = []
        for cam_poly, proj_poly in paired:
            cam_sample = self._sample_polygon(cam_poly)
            proj_sample = self._sample_polygon(proj_poly)
            count = min(len(cam_sample), len(proj_sample))
            if count < 4:
                continue
            src_points.append(cam_sample[:count])
            dst_points.append(proj_sample[:count])

        if not src_points:
            return

        src = np.concatenate(src_points, axis=0)
        dst = np.concatenate(dst_points, axis=0)
        homography, status = cv2.findHomography(src, dst, method=cv2.RANSAC)
        if homography is None or status is None or status.sum() < 4:
            logger.debug("Homography refinement skipped for %s due to insufficient inliers", projector_id)
            return

        self._mapper.update_homography(projector_id, homography, gain=self._gain)
        logger.debug("Updated homography for %s using %d correspondences", projector_id, int(status.sum()))

    def _sample_polygon(self, polygon: np.ndarray) -> np.ndarray:
        contour = polygon.reshape(-1, 1, 2).astype(np.float32)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(simplified) > self._max_points:
            step = len(simplified) // self._max_points
            simplified = simplified[::step]
        if simplified.shape[0] < 4:
            simplified = polygon.reshape(-1, 2)
        return simplified.astype(np.float32)

    def _sort_polygons(self, polygons: Iterable[np.ndarray]) -> List[np.ndarray]:
        return sorted(
            polygons,
            key=lambda poly: cv2.contourArea(poly.reshape(-1, 1, 2).astype(np.float32)),
            reverse=True,
        )
