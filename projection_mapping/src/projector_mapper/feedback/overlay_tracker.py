"""Detects projected overlay regions in the camera feed by color thresholding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from projector_mapper.config import ProjectorConfig


@dataclass(slots=True)
class TrackingResult:
    masks: Dict[str, np.ndarray]
    polygons: Dict[str, List[np.ndarray]]


class OverlayTracker:
    """Segment projector overlays in the camera frame for closed-loop alignment."""

    def __init__(
        self,
        projectors: Iterable[ProjectorConfig],
        min_contour_area: int,
    ) -> None:
        self._ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for projector in projectors:
            lower = np.array(projector.overlay_color_lower_hsv, dtype=np.uint8)
            upper = np.array(projector.overlay_color_upper_hsv, dtype=np.uint8)
            self._ranges[projector.id] = (lower, upper)
        self._min_area = float(min_contour_area)

    def track(self, frame_bgr: np.ndarray) -> TrackingResult:
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        masks: Dict[str, np.ndarray] = {}
        polygons: Dict[str, List[np.ndarray]] = {}

        for projector_id, (lower, upper) in self._ranges.items():
            mask = cv2.inRange(frame_hsv, lower, upper)
            mask = cv2.medianBlur(mask, 5)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered: List[np.ndarray] = []
            cleaned = np.zeros_like(mask)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self._min_area:
                    continue
                cv2.drawContours(cleaned, [contour], -1, color=255, thickness=-1)
                filtered.append(contour.reshape(-1, 2).astype(np.float32))
            masks[projector_id] = cleaned
            polygons[projector_id] = filtered

        return TrackingResult(masks=masks, polygons=polygons)
