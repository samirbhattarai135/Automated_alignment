"""Homography utilities for mapping camera coordinates to projector planes."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute a projective transform mapping src_points to dst_points.

    Args:
        src_points: Array of shape (N, 2) in camera coordinates.
        dst_points: Array of shape (N, 2) in projector coordinates.

    Returns:
        3x3 homography matrix.
    """
    if src_points.shape[0] < 4 or dst_points.shape[0] < 4:
        raise ValueError("Need at least four point correspondences to compute homography")
    homography, status = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC)
    if homography is None:
        raise RuntimeError("cv2.findHomography failed to compute a valid transform")
    if status is not None and status.sum() < 4:
        raise RuntimeError("Insufficient inliers for a stable homography")
    return homography


def warp_polygon(points: Iterable[Iterable[float]], homography: np.ndarray) -> np.ndarray:
    """Apply homography to polygon vertices."""
    pts = np.array(list(points), dtype=np.float32)
    pts = pts.reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, homography)
    return warped.reshape(-1, 2)
