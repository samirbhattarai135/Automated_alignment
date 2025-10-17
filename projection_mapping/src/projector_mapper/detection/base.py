"""Shared interfaces for detection backends."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class Detection:
    """Represents a single semantic object detected on the facade."""

    label: str
    polygon: np.ndarray  # shape (N, 2)
    confidence: float

    def bounding_box(self) -> np.ndarray:
        """Return axis-aligned bounding box [x_min, y_min, x_max, y_max]."""
        x_coords = self.polygon[:, 0]
        y_coords = self.polygon[:, 1]
        return np.array([x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()], dtype=np.float32)


@dataclass(slots=True)
class DetectionResult:
    detections: List[Detection]
    latency_ms: float
    raw_response: Mapping[str, object] | None = None

    @property
    def has_detections(self) -> bool:
        return bool(self.detections)


class DetectionBackend(abc.ABC):
    """Base class for all detection providers."""

    def __init__(self, classes: Sequence[str], confidence_threshold: float) -> None:
        self._classes = set(classes)
        self._confidence_threshold = confidence_threshold

    @property
    def supported_classes(self) -> Iterable[str]:
        return self._classes

    @abc.abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Detect facade features in an OpenCV BGR frame."""

    def _filter_by_confidence(self, detections: Iterable[Detection]) -> List[Detection]:
        return [det for det in detections if det.confidence >= self._confidence_threshold]

    def _start_timer(self) -> float:
        return time.perf_counter()

    def _elapsed_ms(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1_000
