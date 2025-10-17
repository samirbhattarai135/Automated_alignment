"""Detection backends for extracting facade features."""

from .base import Detection, DetectionBackend, DetectionResult  # noqa: F401
from .fal_evfsam import FalEvfSamDetector  # noqa: F401
from .roboflow import RoboflowDetector  # noqa: F401
