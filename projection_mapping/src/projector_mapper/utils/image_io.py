"""Image IO helper routines."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger


def save_frame(frame, directory: Path, stem: Optional[str] = None) -> Path:
    """Persist an OpenCV BGR frame to disk."""
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    stem = stem or "frame"
    path = directory / f"{stem}_{timestamp}.png"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to write frame to {path}")
    logger.debug("Saved frame to %s", path)
    return path
