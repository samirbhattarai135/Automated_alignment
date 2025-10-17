"""Handles warping masks into projector frames and managing feedback alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import cv2
import numpy as np
from loguru import logger

from projector_mapper.geometry import warp_polygon


@dataclass(slots=True)
class ProjectorCalibration:
    id: str
    homography: np.ndarray
    resolution: tuple[int, int]
    color_bgr: tuple[int, int, int]


class ProjectorMapper:
    """Projects masks through calibrated homographies into projector image space."""

    def __init__(self, calibrations: Iterable[ProjectorCalibration]) -> None:
        self._calibrations: Dict[str, ProjectorCalibration] = {
            calib.id: calib for calib in calibrations
        }
        if not self._calibrations:
            raise ValueError("ProjectorMapper requires at least one calibration")

    def warp_mask(self, mask: np.ndarray, projector_id: str) -> np.ndarray:
        calibration = self._calibrations[projector_id]
        width, height = calibration.resolution
        warped = cv2.warpPerspective(
            mask,
            calibration.homography,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
        )
        return warped

    def render_overlay(
        self,
        mask: np.ndarray,
        projector_id: str,
    ) -> np.ndarray:
        warped_mask = self.warp_mask(mask, projector_id)
        calibration = self._calibrations[projector_id]
        color_frame = np.zeros((*warped_mask.shape, 3), dtype=np.uint8)
        for channel, value in enumerate(calibration.color_bgr):
            color_frame[:, :, channel] = np.where(warped_mask > 0, value, 0)
        return color_frame

    def project_polygon(self, polygon: np.ndarray, projector_id: str) -> np.ndarray:
        calibration = self._calibrations[projector_id]
        return warp_polygon(polygon, calibration.homography)

    def update_homography(self, projector_id: str, new_homography: np.ndarray, gain: float) -> None:
        calibration = self._calibrations[projector_id]
        if gain >= 1.0:
            blended = new_homography
        else:
            blended = calibration.homography + gain * (new_homography - calibration.homography)
        if blended[2, 2] != 0:
            blended = blended / blended[2, 2]
        calibration.homography = blended
        logger.debug("Homography for %s updated", projector_id)

    def projector_ids(self) -> List[str]:
        return list(self._calibrations.keys())

    def get_homography(self, projector_id: str) -> np.ndarray:
        return self._calibrations[projector_id].homography.copy()
