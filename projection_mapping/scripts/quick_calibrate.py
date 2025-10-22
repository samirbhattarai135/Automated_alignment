"""
Quick interactive calibration - click corresponding points in camera and projector.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from projector_mapper.config import load_config


class PointSelector:
    """Interactive point selection for calibration."""
    
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.points = []
        self.current_image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            # Draw point
            cv2.circle(self.current_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                self.current_image,
                f"{len(self.points)}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            cv2.imshow(self.window_name, self.current_image)
            logger.info(f"Point {len(self.points)}: ({x}, {y})")
    
    def select_points(self, image: np.ndarray, num_points: int = 4) -> np.ndarray:
        """
        Display image and let user click points.
        
        Args:
            image: Image to display
            num_points: Number of points to collect
            
        Returns:
            Array of points shape (N, 2)
        """
        self.points = []
        self.current_image = image.copy()
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.current_image)
        
        logger.info(f"Click {num_points} points. Press ENTER when done, ESC to cancel.")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                if len(self.points) >= 4:
                    break
                else:
                    logger.warning(f"Need at least 4 points, have {len(self.points)}")
            elif key == 27:  # ESC
                return None
                
        cv2.destroyWindow(self.window_name)
        return np.array(self.points, dtype=np.float32)


def calibrate_manual():
    """Manual calibration by clicking corresponding points."""
    parser = argparse.ArgumentParser(description="Quick manual calibration")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--projector", type=str, default="projector_1", help="Projector ID")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get camera
    camera_id = config.hardware.camera.device_id
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.hardware.camera.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.hardware.camera.resolution[1])
    
    # Get projector config
    projector_config = None
    for proj in config.hardware.projectors:
        if proj.id == args.projector:
            projector_config = proj
            break
    
    if not projector_config:
        logger.error(f"Projector {args.projector} not found in config")
        return
    
    projector_w, projector_h = projector_config.resolution
    
    logger.info("=" * 60)
    logger.info("QUICK CALIBRATION")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Step 1: Capture camera frame")
    logger.info("  Press SPACE to capture, ESC to quit")
    logger.info("")
    
    # Capture frame
    camera_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read camera")
            return
            
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # Space
            camera_frame = frame.copy()
            break
        elif key == 27:  # ESC
            return
    
    cv2.destroyWindow("Camera")
    
    logger.info("")
    logger.info("Step 2: Select points in CAMERA view")
    logger.info("  Click at least 4 distinct points (corners work best)")
    logger.info("  Press ENTER when done")
    logger.info("")
    
    # Select camera points
    camera_selector = PointSelector("Camera Points")
    camera_points = camera_selector.select_points(camera_frame, num_points=4)
    
    if camera_points is None:
        logger.info("Calibration cancelled")
        return
    
    logger.info("")
    logger.info("Step 3: Select SAME points in PROJECTOR space")
    logger.info("  Create a reference image showing your projector view")
    logger.info("  Click the same points in the same order")
    logger.info("")
    
    # Create projector reference image
    projector_ref = np.zeros((projector_h, projector_w, 3), dtype=np.uint8)
    
    # Draw grid for reference
    for i in range(0, projector_w, 100):
        cv2.line(projector_ref, (i, 0), (i, projector_h), (50, 50, 50), 1)
    for i in range(0, projector_h, 100):
        cv2.line(projector_ref, (0, i), (projector_w, i), (50, 50, 50), 1)
    
    # Draw coordinate text
    cv2.putText(projector_ref, f"{projector_w}x{projector_h}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Overlay camera frame (resized) for reference
    scale = min(projector_w / camera_frame.shape[1], projector_h / camera_frame.shape[0]) * 0.5
    ref_w = int(camera_frame.shape[1] * scale)
    ref_h = int(camera_frame.shape[0] * scale)
    camera_small = cv2.resize(camera_frame, (ref_w, ref_h))
    
    x_offset = (projector_w - ref_w) // 2
    y_offset = (projector_h - ref_h) // 2
    
    # Blend camera frame into projector space
    roi = projector_ref[y_offset:y_offset+ref_h, x_offset:x_offset+ref_w]
    blended = cv2.addWeighted(roi, 0.3, camera_small, 0.7, 0)
    projector_ref[y_offset:y_offset+ref_h, x_offset:x_offset+ref_w] = blended
    
    # Draw camera points on reference (scaled and offset)
    for i, (cx, cy) in enumerate(camera_points):
        px = int(cx * scale) + x_offset
        py = int(cy * scale) + y_offset
        cv2.circle(projector_ref, (px, py), 8, (0, 255, 255), 2)
        cv2.putText(projector_ref, f"{i+1}", (px+12, py-12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.putText(projector_ref, "Click corresponding points in same order",
               (20, projector_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Select projector points
    projector_selector = PointSelector("Projector Points")
    projector_points = projector_selector.select_points(projector_ref, num_points=len(camera_points))
    
    if projector_points is None:
        logger.info("Calibration cancelled")
        return
    
    # Compute homography
    logger.info("")
    logger.info("Computing homography...")
    logger.info(f"Camera points: {len(camera_points)}")
    logger.info(f"Projector points: {len(projector_points)}")
    
    H, mask = cv2.findHomography(camera_points, projector_points, cv2.RANSAC, 5.0)
    
    if H is None:
        logger.error("Failed to compute homography!")
        return
    
    inliers = np.sum(mask)
    logger.info(f"Homography computed with {inliers}/{len(mask)} inliers")
    
    # Save homography
    output_dir = Path("data/calibration/homographies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{args.projector}.npy"
    np.save(output_path, H)
    logger.info(f"Saved homography to {output_path}")
    
    # Visualize result
    logger.info("")
    logger.info("Visualizing calibration...")
    
    # Warp camera frame to projector space
    warped = cv2.warpPerspective(camera_frame, H, (projector_w, projector_h))
    
    # Show result
    cv2.imshow("Calibration Result", warped)
    logger.info("Press any key to finish")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cap.release()
    logger.info("Calibration complete!")


if __name__ == "__main__":
    calibrate_manual()
