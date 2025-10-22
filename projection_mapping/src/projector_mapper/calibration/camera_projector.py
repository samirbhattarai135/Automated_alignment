"""Camera-projector calibration routines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from loguru import logger

from projector_mapper.config import CalibrationConfig, ProjectorConfig


@dataclass(slots=True)
class CameraIntrinsics:
    matrix: np.ndarray
    distortion: np.ndarray


@dataclass(slots=True)
class ProjectorHomography:
    projector_id: str
    homography: np.ndarray


class CalibrationManager:
    """Computes and persists calibration between a fixed camera and multiple projectors."""

    def __init__(self, config: CalibrationConfig, projectors: Iterable[ProjectorConfig]) -> None:
        self._config = config
        self._projectors = list(projectors)

    def estimate_camera_intrinsics(self, image_paths: Iterable[Path]) -> CameraIntrinsics:
        """Estimate camera intrinsics from checkerboard images."""
        obj_points: List[np.ndarray] = []
        img_points: List[np.ndarray] = []
        pattern_size = tuple(self._config.pattern.inner_corners)
        square_size = self._config.pattern.square_size_m

        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size

        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                logger.warning("Skipping unreadable calibration image: %s", path)
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if not ok:
                logger.warning("Checkerboard not detected in %s", path)
                continue
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(refined)

        if not obj_points:
            raise RuntimeError("No valid checkerboard detections for intrinsics calibration")

        ret, matrix, distortion, _, _ = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        if not ret:
            raise RuntimeError("cv2.calibrateCamera failed to converge")

        logger.info("Estimated camera intrinsics with reprojection error %.4f", ret)
        self._save_camera_intrinsics(matrix, distortion)
        return CameraIntrinsics(matrix=matrix, distortion=distortion)

    def compute_projector_homography(
        self,
        camera_points: np.ndarray,
        projector_points: np.ndarray,
        projector_id: str,
    ) -> ProjectorHomography:
        if camera_points.shape != projector_points.shape:
            raise ValueError("Camera and projector points must share shape")
        homography, status = cv2.findHomography(camera_points, projector_points, method=cv2.RANSAC)
        if homography is None or status is None:
            raise RuntimeError("Failed to compute projector homography")
        inlier_ratio = float(status.sum()) / float(status.size)
        logger.info(
            "Homography for %s computed with %.2f%% inliers",
            projector_id,
            inlier_ratio * 100.0,
        )
        self._save_projector_homography(projector_id, homography)
        return ProjectorHomography(projector_id=projector_id, homography=homography)

    def load_homography(self, projector_id: str) -> np.ndarray:
        path = Path("/Users/samir/Projects/Illumibot_alignment/Automated_alignment/projection_mapping/data/calibration/projector_1_homography.npy")
        if not path.exists():
            raise FileNotFoundError(f"Homography file missing for projector {projector_id}: {path}")
        return np.load(path)

    def _save_camera_intrinsics(self, matrix: np.ndarray, distortion: np.ndarray) -> None:
        payload: Dict[str, List[float]] = {
            "camera_matrix": matrix.tolist(),
            "distortion_coefficients": distortion.tolist(),
        }
        with open(self._config.camera_intrinsics_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Camera intrinsics saved to %s", self._config.camera_intrinsics_path)

    def _save_projector_homography(self, projector_id: str, homography: np.ndarray) -> None:
        path = self._config.homographies_dir / f"{projector_id}.npy"
        np.save(path, homography)
        logger.info("Homography for %s saved to %s", projector_id, path)
