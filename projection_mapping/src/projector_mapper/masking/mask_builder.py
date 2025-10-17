"""Utilities to construct binary masks from detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from projector_mapper.detection import Detection


@dataclass(slots=True)
class MaskBuildResult:
    composite_mask: np.ndarray
    per_label_masks: Dict[str, np.ndarray]


class MaskBuilder:
    """Transforms detection polygons into projector-ready binary masks."""

    def __init__(self, dilate_kernel: int, blur_kernel: int) -> None:
        self._dilate_kernel = max(1, dilate_kernel)
        self._blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def build(self, frame_shape: Tuple[int, int], detections: Iterable[Detection]) -> MaskBuildResult:
        height, width = frame_shape
        composite = np.zeros((height, width), dtype=np.uint8)
        per_label: Dict[str, np.ndarray] = {}

        for detection in detections:
            mask = per_label.setdefault(detection.label, np.zeros_like(composite))
            polygon = detection.polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [polygon], 255)
            composite = cv2.bitwise_or(composite, mask)

        if self._dilate_kernel > 1:
            kernel = np.ones((self._dilate_kernel, self._dilate_kernel), np.uint8)
            composite = cv2.dilate(composite, kernel, iterations=1)
            for label, mask in per_label.items():
                per_label[label] = cv2.dilate(mask, kernel, iterations=1)

        composite = cv2.GaussianBlur(composite, (self._blur_kernel, self._blur_kernel), sigmaX=0)
        for label, mask in per_label.items():
            per_label[label] = cv2.GaussianBlur(mask, (self._blur_kernel, self._blur_kernel), sigmaX=0)

        return MaskBuildResult(composite_mask=composite, per_label_masks=per_label)
