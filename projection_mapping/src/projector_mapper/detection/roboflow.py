"""Roboflow detection backend using hosted inference API."""

from __future__ import annotations

import os
from typing import List, Mapping

import cv2
import numpy as np
import requests
from loguru import logger

from .base import Detection, DetectionBackend, DetectionResult


class RoboflowDetector(DetectionBackend):
    """Calls a Roboflow Hosted Inference endpoint for object detection."""

    def __init__(self, endpoint: str, api_key_env: str, classes: List[str], confidence_threshold: float) -> None:
        super().__init__(classes=classes, confidence_threshold=confidence_threshold)
        self._endpoint = endpoint.rstrip("/")
        self._api_key_env = api_key_env

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        start_time = self._start_timer()
        api_key = os.getenv(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{self._api_key_env}' not populated with Roboflow API key"
            )

        ok, buffer = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("Failed to encode frame for Roboflow request")

        try:
            response = requests.post(
                self._endpoint,
                params={"api_key": api_key},
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network errors hard to test deterministically
            raise RuntimeError(f"Roboflow request failed: {exc}") from exc

        payload: Mapping[str, object] = response.json()
        detections = self._parse_predictions(payload)
        filtered = self._filter_by_confidence(detections)
        latency_ms = self._elapsed_ms(start_time)
        logger.debug(
            "Roboflow returned %d detections in %.2f ms", len(filtered), latency_ms
        )
        return DetectionResult(detections=filtered, latency_ms=latency_ms, raw_response=payload)

    def _parse_predictions(self, payload: Mapping[str, object]) -> List[Detection]:
        predictions = payload.get("predictions", [])
        detections: List[Detection] = []
        for pred in predictions:
            label = str(pred.get("class", "unknown"))
            if self._classes and label not in self._classes:
                continue
            confidence = float(pred.get("confidence", 0.0))
            polygon = self._prediction_to_polygon(pred)
            detections.append(Detection(label=label, polygon=polygon, confidence=confidence))
        return detections

    def _prediction_to_polygon(self, pred: Mapping[str, object]) -> np.ndarray:
        if "points" in pred:
            points = pred["points"]
            return np.array([(float(pt["x"]), float(pt["y"])) for pt in points], dtype=np.float32)

        x = float(pred.get("x", 0.0))
        y = float(pred.get("y", 0.0))
        width = float(pred.get("width", 0.0))
        height = float(pred.get("height", 0.0))
        half_w = width / 2.0
        half_h = height / 2.0
        # Construct rectangle polygon in clockwise order.
        return np.array(
            [
                (x - half_w, y - half_h),
                (x + half_w, y - half_h),
                (x + half_w, y + half_h),
                (x - half_w, y + half_h),
            ],
            dtype=np.float32,
        )
