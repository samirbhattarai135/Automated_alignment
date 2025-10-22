"""ICP-based alignment for projector overlay."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from loguru import logger
from scipy.spatial import KDTree


@dataclass(slots=True)
class AlignmentResult:
    """Result of ICP alignment process."""
    
    homography: np.ndarray  # 3x3 transformation matrix
    aligned_overlay: np.ndarray  # Warped overlay image
    error: float  # Mean alignment error in pixels
    iterations: int  # Number of ICP iterations performed
    converged: bool  # Whether ICP converged
    correspondences: int  # Number of point correspondences found


class ICPAligner:
    """Aligns projected overlay with real objects using Iterative Closest Point."""
    
    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 0.5,  # pixels
        min_correspondences: int = 4,
        subsample_rate: int = 5,  # Use every Nth contour point
        outlier_threshold: float = 50.0,  # pixels - reject correspondences beyond this
    ):
        """
        Args:
            max_iterations: Maximum ICP iterations
            tolerance: Stop when error change is below this (pixels)
            min_correspondences: Minimum points needed for homography
            subsample_rate: Sample every Nth point from contours (reduces computation)
            outlier_threshold: Reject point pairs with distance > threshold
        """
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._min_correspondences = min_correspondences
        self._subsample_rate = subsample_rate
        self._outlier_threshold = outlier_threshold
    
    def align(
        self,
        camera_frame: np.ndarray,
        overlay_mask: np.ndarray,
        detected_contours: List[np.ndarray],
        initial_homography: Optional[np.ndarray] = None,
    ) -> AlignmentResult:
        """
        Align overlay mask to detected objects using ICP.
        
        Args:
            camera_frame: BGR camera image (for visualization/debugging)
            overlay_mask: Binary mask of projected overlay (what we want to align)
            detected_contours: List of contours from object detection (ground truth)
            initial_homography: Starting transform (identity if None)
        
        Returns:
            AlignmentResult with aligned overlay and transformation
        """
        start_time = time.perf_counter()
        
        # Extract contours from overlay mask
        overlay_contours = self._extract_contours(overlay_mask)
        
        if not overlay_contours or not detected_contours:
            logger.warning("No contours found for alignment")
            return self._create_fallback_result(overlay_mask, initial_homography)
        
        # Convert contours to point clouds
        overlay_points = self._contours_to_points(overlay_contours)
        detected_points = self._contours_to_points(detected_contours)
        
        if len(overlay_points) < self._min_correspondences or len(detected_points) < self._min_correspondences:
            logger.warning(f"Insufficient points: overlay={len(overlay_points)}, detected={len(detected_points)}")
            return self._create_fallback_result(overlay_mask, initial_homography)
        
        # Initialize transformation
        if initial_homography is None:
            current_H = np.eye(3, dtype=np.float32)
        else:
            current_H = initial_homography.copy()
        
        # Run ICP iterations
        prev_error = float('inf')
        converged = False
        iteration = 0
        
        for iteration in range(self._max_iterations):
            # Transform overlay points with current homography
            transformed_points = self._transform_points(overlay_points, current_H)
            
            # Find nearest neighbors (correspondences)
            correspondences, distances = self._find_correspondences(
                transformed_points, detected_points
            )
            
            # Filter outliers
            inlier_mask = distances < self._outlier_threshold
            inlier_source = overlay_points[inlier_mask]
            inlier_target = detected_points[correspondences[inlier_mask]]
            
            num_inliers = np.sum(inlier_mask)
            
            if num_inliers < self._min_correspondences:
                logger.warning(f"ICP iteration {iteration}: only {num_inliers} inliers")
                break
            
            # Compute mean error
            mean_error = np.mean(distances[inlier_mask])
            
            # Check convergence
            error_change = abs(prev_error - mean_error)
            if error_change < self._tolerance:
                logger.debug(f"ICP converged at iteration {iteration} (error={mean_error:.2f}px)")
                converged = True
                break
            
            # Estimate new homography from correspondences
            try:
                H_update, _ = cv2.findHomography(
                    inlier_source.astype(np.float32),
                    inlier_target.astype(np.float32),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0,
                )
                
                if H_update is None:
                    # This is normal - just means not enough points for this iteration
                    logger.debug(f"ICP iteration {iteration}: homography estimation failed (not enough inliers)")
                    break
                
                # Compose transformations
                current_H = H_update @ current_H
                
            except Exception as e:
                logger.warning(f"ICP iteration {iteration} failed: {e}")
                break
            
            prev_error = mean_error
        
        # Apply final transformation to overlay
        h, w = camera_frame.shape[:2]
        aligned_overlay = cv2.warpPerspective(
            overlay_mask,
            current_H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"ICP alignment completed in {elapsed_ms:.1f}ms: error={prev_error:.2f}px, iterations={iteration+1}")
        
        return AlignmentResult(
            homography=current_H,
            aligned_overlay=aligned_overlay,
            error=prev_error,
            iterations=iteration + 1,
            converged=converged,
            correspondences=num_inliers if num_inliers else 0,
        )
    
    def _extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extract contours from binary mask."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        return [cnt.reshape(-1, 2) for cnt in contours if len(cnt) >= 4]
    
    def _contours_to_points(self, contours: List[np.ndarray]) -> np.ndarray:
        """Convert list of contours to single point cloud with subsampling."""
        all_points = []
        for contour in contours:
            # Subsample to reduce computation
            sampled = contour[::self._subsample_rate]
            all_points.append(sampled)
        
        if not all_points:
            return np.empty((0, 2), dtype=np.float32)
        
        return np.vstack(all_points).astype(np.float32)
    
    def _transform_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Apply homography to points."""
        if len(points) == 0:
            return points
        
        # Convert to homogeneous coordinates
        ones = np.ones((len(points), 1), dtype=np.float32)
        points_h = np.hstack([points, ones])
        
        # Apply transformation
        transformed_h = (H @ points_h.T).T
        
        # Convert back to Cartesian
        transformed = transformed_h[:, :2] / transformed_h[:, 2:3]
        
        return transformed.astype(np.float32)
    
    def _find_correspondences(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest neighbor correspondences using KD-tree.
        
        Returns:
            indices: For each source point, index of nearest target point
            distances: Distance to nearest neighbor
        """
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)
        return indices, distances
    
    def _create_fallback_result(
        self,
        overlay_mask: np.ndarray,
        initial_homography: Optional[np.ndarray],
    ) -> AlignmentResult:
        """Create fallback result when alignment fails."""
        H = initial_homography if initial_homography is not None else np.eye(3, dtype=np.float32)
        
        return AlignmentResult(
            homography=H,
            aligned_overlay=overlay_mask,
            error=float('inf'),
            iterations=0,
            converged=False,
            correspondences=0,
        )