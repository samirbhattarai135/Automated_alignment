"""Detection backend integrating FAL.ai EVF-SAM masks."""

from __future__ import annotations

import base64
import os
import time
from typing import Dict, Iterable, List

import cv2
import numpy as np
from loguru import logger

try:
    import fal_client
    HAS_FAL_CLIENT = True
except ImportError:
    HAS_FAL_CLIENT = False
    import requests

from .base import Detection, DetectionBackend, DetectionResult


class FalEvfSamDetector(DetectionBackend):
    """Generates masks using FAL.ai EVF-SAM for configured façade classes."""

    _BASE_URL = "https://fal.ai/models/"

    def __init__(
        self,
        classes: Iterable[str],
        confidence_threshold: float,
        model_id: str,
        api_key_env: str,
        prompt_map: Dict[str, str],
        mask_only: bool,
        fill_holes: bool,
        revert_mask: bool,
        poll_interval_s: float,
        timeout_s: float,
    ) -> None:
        super().__init__(classes=classes, confidence_threshold=confidence_threshold)
        self._model_id = model_id.strip("/")
        self._api_key_env = api_key_env
        self._prompt_map = dict(prompt_map)
        self._mask_only = mask_only
        self._fill_holes = fill_holes
        self._revert_mask = revert_mask
        self._poll_interval_s = poll_interval_s
        self._timeout_s = timeout_s

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        start = self._start_timer()
        api_key = os.getenv(self._api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable '{self._api_key_env}' not populated with FAL.ai API key")

        # Set API key for fal_client
        if HAS_FAL_CLIENT:
            os.environ["FAL_KEY"] = api_key

        ok, buffer = cv2.imencode(".png", frame_bgr)
        if not ok:
            raise RuntimeError("Failed to encode frame for FAL EVF-SAM request")
        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        detections: List[Detection] = []
        for label in self.supported_classes:
            prompt = self._prompt_map.get(label, label)
            mask = self._request_mask(api_key, image_b64, prompt)
            if mask is None:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.shape[0] < 3:
                    continue
                polygon = contour.reshape(-1, 2).astype(np.float32)
                detections.append(Detection(label=label, polygon=polygon, confidence=1.0))

        filtered = self._filter_by_confidence(detections)
        latency_ms = self._elapsed_ms(start)
        logger.debug("FAL EVF-SAM produced %d detections in %.2f ms", len(filtered), latency_ms)
        return DetectionResult(detections=filtered, latency_ms=latency_ms)

    def _request_mask(self, api_key: str, image_b64: str, prompt: str) -> np.ndarray | None:
        if HAS_FAL_CLIENT:
            return self._request_mask_with_client(image_b64, prompt)
        else:
            return self._request_mask_with_requests(api_key, image_b64, prompt)

    def _request_mask_with_client(self, image_b64: str, prompt: str) -> np.ndarray | None:
        """Use official fal-client library according to FAL.ai documentation."""
        try:
            # According to FAL.ai docs, use subscribe with image_url and prompt
            result = fal_client.subscribe(
                self._model_id,  # "fal-ai/evf-sam"
                arguments={
                    "prompt": prompt,
                    "image_url": f"data:image/png;base64,{image_b64}",
                    "mask_only": self._mask_only,
                    "fill_holes": self._fill_holes,
                    "revert_mask": self._revert_mask,
                },
                with_logs=False,
            )
            
            # Response format: {"image": {"url": "...", "content_type": "image/png", ...}}
            if isinstance(result, dict) and "image" in result:
                image_data = result["image"]
                if isinstance(image_data, dict) and "url" in image_data:
                    image_url = image_data["url"]
                    return self._download_mask(image_url)
            
            logger.warning(f"FAL client returned unexpected format for prompt '{prompt}': {result}")
            return None
        except Exception as e:
            logger.warning(f"FAL client error for prompt '{prompt}': {e}")
            return None

    def _request_mask_with_requests(self, api_key: str, image_b64: str, prompt: str) -> np.ndarray | None:
        """Fallback to REST API if fal-client not available."""
        import requests
        
        request_id = self._submit_job(api_key, image_b64, prompt)
        if not request_id:
            return None
        expiry = time.time() + self._timeout_s
        # FAL.ai uses /queue endpoint for status checks
        status_url = f"{self._BASE_URL}/queue/{self._model_id}/{request_id}"
        headers = self._headers(api_key)
        while time.time() < expiry:
            response = requests.get(status_url, headers=headers, timeout=15)
            response.raise_for_status()
            payload = response.json()
            state = payload.get("status")
            if state == "COMPLETED":
                output = payload.get("output", {})
                # Try to get image URL from various possible fields
                image_url = output.get("image_url") or output.get("mask_url")
                if isinstance(output, dict) and "image" in output:
                    image_data = output["image"]
                    if isinstance(image_data, dict):
                        image_url = image_data.get("url")
                if not image_url:
                    logger.warning("FAL EVF-SAM completed but no image URL found: %s", payload)
                    return None
                return self._download_mask(image_url)
            if state in {"FAILED", "CANCELLED"}:
                logger.warning("FAL EVF-SAM job failed with status %s", state)
                return None
            time.sleep(self._poll_interval_s)
        logger.warning("FAL EVF-SAM job timed out for prompt '%s'", prompt)
        return None

    def _submit_job(self, api_key: str, image_b64: str, prompt: str) -> str | None:
        """Submit job to FAL.ai queue endpoint according to API documentation."""
        # FAL.ai queue endpoint format
        url = f"{self._BASE_URL}/queue/{self._model_id}"
        headers = self._headers(api_key)
        
        # Payload format according to FAL.ai docs
        payload = {
            "prompt": prompt,
            "image_url": f"data:image/png;base64,{image_b64}",
            "mask_only": self._mask_only,
            "fill_holes": self._fill_holes,
            "revert_mask": self._revert_mask,
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        body = response.json()
        request_id = body.get("request_id")
        if not request_id:
            logger.warning("FAL EVF-SAM response missing request_id: %s", body)
            return None
        return str(request_id)

    def _download_mask(self, url: str) -> np.ndarray | None:
        """Download mask image from URL and convert to binary mask."""
        import requests
        
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            logger.warning("Failed to download FAL mask from %s: %s", url, response.status_code)
            return None
        buffer = np.frombuffer(response.content, dtype=np.uint8)
        mask = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def _headers(api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
