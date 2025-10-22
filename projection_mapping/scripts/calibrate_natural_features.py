"""
Markerless camera-projector calibration using natural scene features.

This method:
1. Projects simple shapes (dots, rectangles) at known projector coordinates
2. Detects those shapes in the camera feed
3. Also uses detected building features (windows/doors) as correspondences
4. Computes homography from projector → camera mapping
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

from projector_mapper.config import load_config
from projector_mapper.detection import FalEvfSamDetector


class NaturalFeatureCalibrator:
    """Calibrate using natural features + projected markers."""
    
    def __init__(
        self,
        camera_resolution: Tuple[int, int],
        projector_resolution: Tuple[int, int],
    ):
        self.camera_res = camera_resolution
        self.projector_res = projector_resolution
        
        # Correspondence storage
        self.projector_points: List[np.ndarray] = []
        self.camera_points: List[np.ndarray] = []
    
    def generate_dot_pattern(
        self,
        num_dots: int = 12
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Generate a grid of colored dots to project.
        
        Returns:
            (pattern_image, dot_positions_in_projector_coords)
        """
        width, height = self.projector_res
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create grid of dots
        dots = []
        margin = 100
        cols = int(np.sqrt(num_dots * width / height))
        rows = int(np.ceil(num_dots / cols))
        
        x_spacing = (width - 2 * margin) // (cols - 1)
        y_spacing = (height - 2 * margin) // (rows - 1)
        
        for i in range(rows):
            for j in range(cols):
                x = margin + j * x_spacing
                y = margin + i * y_spacing
                
                # Draw colored circle
                color = (0, 255, 0)  # Green for easy detection
                cv2.circle(pattern, (x, y), 30, color, -1)
                cv2.circle(pattern, (x, y), 32, (255, 255, 255), 2)  # White outline
                
                dots.append((x, y))
        
        return pattern, dots
    
    def detect_dots_in_camera(
        self,
        camera_frame: np.ndarray,
        expected_num: int = 12
    ) -> List[Tuple[int, int]]:
        """
        Detect projected green dots in camera image.
        
        Returns:
            List of (x, y) positions in camera coordinates
        """
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        
        # Detect green color (adjust ranges if needed)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours (blobs)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract centers of circular blobs
        dots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Too small
                continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            dots.append((cx, cy))
        
        logger.debug(f"Detected {len(dots)} dots in camera")
        
        return dots
    
    def match_dots(
        self,
        projector_dots: List[Tuple[int, int]],
        camera_dots: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match detected dots between projector and camera coordinates.
        
        Uses spatial consistency - dots should maintain relative positions.
        """
        if len(camera_dots) < 4:
            raise ValueError(f"Need at least 4 detected dots, got {len(camera_dots)}")
        
        # Sort both by position (left-to-right, top-to-bottom)
        projector_sorted = sorted(projector_dots, key=lambda p: (p[1], p[0]))
        camera_sorted = sorted(camera_dots, key=lambda p: (p[1], p[0]))
        
        # Take the minimum number available
        n_matches = min(len(projector_sorted), len(camera_sorted))
        
        proj_pts = np.array(projector_sorted[:n_matches], dtype=np.float32)
        cam_pts = np.array(camera_sorted[:n_matches], dtype=np.float32)
        
        logger.info(f"Matched {n_matches} dot correspondences")
        
        return proj_pts, cam_pts
    
    def capture_dot_correspondences(
        self,
        camera_frame: np.ndarray,
        projector_dots: List[Tuple[int, int]]
    ) -> bool:
        """Capture correspondences from projected dots."""
        camera_dots = self.detect_dots_in_camera(camera_frame)
        
        if len(camera_dots) < 4:
            logger.warning(f"Only detected {len(camera_dots)} dots, need at least 4")
            return False
        
        proj_pts, cam_pts = self.match_dots(projector_dots, camera_dots)
        
        self.projector_points.append(proj_pts)
        self.camera_points.append(cam_pts)
        
        return True
    
    def add_feature_correspondences(
        self,
        detector: FalEvfSamDetector,
        camera_frame: np.ndarray,
        projected_overlay: np.ndarray
    ) -> int:
        """
        Add correspondences from naturally detected features (windows/doors).
        
        This enhances calibration by using real scene geometry.
        """
        # Detect features in camera view
        camera_detections = detector.detect(camera_frame)
        
        # Detect features in projected overlay
        overlay_gray = cv2.cvtColor(projected_overlay, cv2.COLOR_BGR2GRAY)
        _, overlay_mask = cv2.threshold(overlay_gray, 1, 255, cv2.THRESH_BINARY)
        overlay_contours, _ = cv2.findContours(
            overlay_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not camera_detections.detections or not overlay_contours:
            logger.warning("No features detected for correspondence")
            return 0
        
        # Extract corner points from detections
        camera_pts = []
        for det in camera_detections.detections:
            # Use polygon corners
            for point in det.polygon[::2]:  # Subsample to avoid too many points
                camera_pts.append(point)
        
        # Extract corner points from overlay
        projector_pts = []
        for contour in overlay_contours:
            # Approximate polygon and get corners
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            for point in approx[::2]:
                projector_pts.append(point[0])
        
        if len(camera_pts) < 4 or len(projector_pts) < 4:
            logger.warning("Insufficient feature points")
            return 0
        
        # Align counts
        n = min(len(camera_pts), len(projector_pts))
        camera_pts = np.array(camera_pts[:n], dtype=np.float32)
        projector_pts = np.array(projector_pts[:n], dtype=np.float32)
        
        self.camera_points.append(camera_pts)
        self.projector_points.append(projector_pts)
        
        logger.info(f"Added {n} feature correspondences from natural features")
        return n
    
    def compute_homography(self) -> Tuple[np.ndarray, float]:
        """Compute homography using RANSAC on all correspondences."""
        if not self.camera_points:
            raise ValueError("No correspondences captured")
        
        # Concatenate all correspondences
        projector_pts = np.vstack(self.projector_points).astype(np.float32)
        camera_pts = np.vstack(self.camera_points).astype(np.float32)
        
        logger.info(f"Computing homography from {len(projector_pts)} correspondences")
        
        # RANSAC homography
        H, mask = cv2.findHomography(
            projector_pts,
            camera_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        
        if H is None:
            raise RuntimeError("Failed to compute homography")
        
        # Calculate metrics
        inliers = mask.sum()
        inlier_ratio = inliers / len(mask)
        
        projected = cv2.perspectiveTransform(
            projector_pts.reshape(-1, 1, 2),
            H
        ).reshape(-1, 2)
        
        errors = np.linalg.norm(camera_pts - projected, axis=1)
        mean_error = errors[mask.ravel() == 1].mean()
        
        logger.info(f"Homography computed:")
        logger.info(f"  Inliers: {inliers}/{len(mask)} ({inlier_ratio*100:.1f}%)")
        logger.info(f"  Mean error: {mean_error:.2f} pixels")
        
        return H, mean_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Markerless camera-projector calibration using natural features"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to pipeline configuration"
    )
    parser.add_argument(
        "--projector-id",
        type=str,
        default="projector_1",
        help="ID of projector to calibrate"
    )
    parser.add_argument(
        "--use-features",
        action="store_true",
        help="Also use detected building features for calibration"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    
    # Load environment for API keys
    from run_alignment_loop import load_env_file
    load_env_file()
    
    # Get projector config
    projector_cfg = cfg.projector_by_id(args.projector_id)
    
    # Initialize calibrator
    calibrator = NaturalFeatureCalibrator(
        camera_resolution=(cfg.hardware.camera.width, cfg.hardware.camera.height),
        projector_resolution=(projector_cfg.width, projector_cfg.height),
    )
    
    # Initialize camera
    camera_cfg = cfg.hardware.camera
    capture = cv2.VideoCapture(camera_cfg.device_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)
    
    if not capture.isOpened():
        raise RuntimeError("Failed to open camera")
    
    # Initialize detector if using features
    detector = None
    if args.use_features:
        from projector_mapper.pipeline import AlignmentPipeline
        detector = AlignmentPipeline._build_detector(cfg.pipeline)
        logger.info("Feature detection enabled - will use building features")
    
    # Generate dot pattern
    dot_pattern, projector_dots = calibrator.generate_dot_pattern(num_dots=12)
    
    # Create projector window
    window_name = f"Calibration_{args.projector_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 1920, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, dot_pattern)
    cv2.waitKey(1000)
    
    logger.info("=" * 60)
    logger.info("MARKERLESS CALIBRATION - Natural Features + Dots")
    logger.info("=" * 60)
    logger.info(f"Projecting {len(projector_dots)} green dots")
    logger.info("")
    logger.info("Instructions:")
    logger.info("1. Adjust camera to see the projected dots on the surface")
    logger.info("2. Press SPACE to capture correspondences")
    logger.info("3. Capture 3-5 times from different angles")
    logger.info("4. Press F to use building features (if --use-features)")
    logger.info("5. Press Q to compute calibration")
    logger.info("=" * 60)
    
    captured = 0
    feature_used = False
    
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                continue
            
            # Visualize detected dots
            display = frame.copy()
            dots_detected = calibrator.detect_dots_in_camera(frame)
            
            for x, y in dots_detected:
                cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            
            cv2.putText(
                display,
                f"Captured: {captured} | Dots: {len(dots_detected)} | SPACE=capture F=features Q=finish",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            if len(dots_detected) >= 4:
                cv2.putText(
                    display,
                    "Ready! Press SPACE to capture",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
            else:
                cv2.putText(
                    display,
                    f"Need {4 - len(dots_detected)} more dots visible",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            
            cv2.imshow("Camera Feed", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(" "):  # SPACE
                if calibrator.capture_dot_correspondences(frame, projector_dots):
                    captured += 1
                    logger.info(f"✓ Captured correspondence {captured}")
                else:
                    logger.warning("✗ Failed to capture")
            
            elif key == ord("f") and args.use_features:  # F - use features
                logger.info("Detecting natural features...")
                # Project current overlay for feature matching
                n_features = calibrator.add_feature_correspondences(
                    detector,
                    frame,
                    dot_pattern
                )
                if n_features > 0:
                    feature_used = True
                    logger.info(f"✓ Added {n_features} feature correspondences")
            
            elif key in (ord("q"), 27):  # Q or ESC
                if captured >= 3 or feature_used:
                    break
                else:
                    logger.warning("Need at least 3 captures or feature detection")
        
        # Compute homography
        logger.info("")
        logger.info("Computing homography...")
        homography, error = calibrator.compute_homography()
        
        logger.info("")
        logger.info("Homography matrix:")
        logger.info(str(homography))
        logger.info(f"Reprojection error: {error:.2f} pixels")
        
        # Save
        output_path = cfg.calibration.homographies_dir / f"{args.projector_id}.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, homography)
        logger.info(f"✓ Saved to {output_path}")
        
        # Visualize
        ret, frame = capture.read()
        if ret:
            vis = frame.copy()
            
            # Draw projected grid overlay
            w, h = calibrator.projector_res
            for x in range(0, w, 100):
                line_proj = np.array([[[x, 0]], [[x, h]]], dtype=np.float32)
                line_cam = cv2.perspectiveTransform(line_proj, homography)
                pt1 = tuple(line_cam[0, 0].astype(int))
                pt2 = tuple(line_cam[1, 0].astype(int))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
            
            for y in range(0, h, 100):
                line_proj = np.array([[[0, y]], [[w, y]]], dtype=np.float32)
                line_cam = cv2.perspectiveTransform(line_proj, homography)
                pt1 = tuple(line_cam[0, 0].astype(int))
                pt2 = tuple(line_cam[1, 0].astype(int))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
            
            cv2.imshow("Calibration Result", vis)
            
            vis_path = output_path.parent / f"{args.projector_id}_calibration_vis.png"
            cv2.imwrite(str(vis_path), vis)
            logger.info(f"✓ Saved visualization to {vis_path}")
            
            cv2.waitKey(0)
    
    finally:
        capture.release()
        cv2.destroyAllWindows()
        logger.info("Calibration complete!")


if __name__ == "__main__":
    main()