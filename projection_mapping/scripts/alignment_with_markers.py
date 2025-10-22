"""
Interactive marker-based alignment system.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import yaml
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from projector_mapper.config import PipelineConfig
from projector_mapper.detection.fal_evfsam import FalEvfSamDetector
from projector_mapper.masking.mask_builder import MaskBuilder

logger = logging.getLogger(__name__)

# Marker colors (BGR) - all darker navy blue
MARKER_COLORS = [
    (80, 0, 0),      # Darker navy blue
    (80, 0, 0),      # Darker navy blue
    (80, 0, 0),      # Darker navy blue
    (80, 0, 0),      # Darker navy blue
    (80, 0, 0),      # Darker navy blue
    (80, 0, 0),      # Darker navy blue
    (80, 0, 0),      # Darker navy blue
    (80, 0, 0),      # Darker navy blue
]



class MarkerPoint:
    """Represents a calibration marker point."""
    
    def __init__(self, id: int, projector_pos: Tuple[int, int], color: Tuple[int, int, int]):
        self.id = id
        self.projector_pos = np.array(projector_pos, dtype=np.float32)
        self.camera_pos: Optional[np.ndarray] = None
        self.color = color
        self.detected = False


class MarkerBasedAlignment:
    """Continuous alignment using projected markers and camera feedback."""
    
    def __init__(self, config: PipelineConfig, projector_id: str = "projector_1"):
        self.config = config
        self.projector_id = projector_id
        
        # Get projector config
        self.proj_config = config.projector_by_id(projector_id)
        self.proj_width, self.proj_height = self.proj_config.resolution
        
        # Initialize camera directly with OpenCV
        self.camera = cv2.VideoCapture(config.hardware.camera.device_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.hardware.camera.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.hardware.camera.height)
        
        # Marker settings
        self.markers: List[MarkerPoint] = []
        self.reference_image: Optional[np.ndarray] = None
        self.homography: Optional[np.ndarray] = None
        
        # Detection settings
        self.marker_radius = 30  # pixels
        self.marker_text_size = 1.5
        
        # Paths
        self.data_dir = Path("data/marker_calibration")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Detector (initialized later when needed)
        self.detector = None
        self.mask_builder = None  # Changed from mask_generator to mask_builder
        
        logger.info(f"Marker-based alignment initialized for {projector_id}")
    
    def read_camera(self) -> Tuple[bool, np.ndarray]:
        """Read a frame from the camera."""
        return self.camera.read()
    
    def release_camera(self):
        """Release the camera."""
        self.camera.release()
    
    def capture_reference_image(self) -> np.ndarray:
        """Capture a reference image from the camera showing the house."""
        logger.info("Capturing reference image...")
        
        cv2.namedWindow("Reference Capture", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reference Capture", 1280, 720)
        
        print("\n" + "="*60)
        print("REFERENCE IMAGE CAPTURE")
        print("="*60)
        print("Position your camera to see the entire house.")
        print("Press SPACE to capture the reference image")
        print("Press ESC to cancel")
        print("="*60 + "\n")
        
        while True:
            ret, frame = self.read_camera()
            if not ret:
                continue
            
            # Display with instructions
            display = frame.copy()
            cv2.putText(display, "Press SPACE to capture", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Press ESC to cancel", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Reference Capture", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                self.reference_image = frame.copy()
                # Save reference image
                ref_path = self.data_dir / f"{self.projector_id}_reference.jpg"
                cv2.imwrite(str(ref_path), self.reference_image)
                logger.info(f"Reference image saved to {ref_path}")
                break
            elif key == 27:  # ESC
                raise KeyboardInterrupt("Reference capture cancelled")
        
        cv2.destroyWindow("Reference Capture")
        return self.reference_image
    
    def select_marker_points(self, num_points: int = 8) -> List[MarkerPoint]:
        """Interactive point selection on reference image."""
        logger.info(f"Starting point selection for {num_points} markers...")
        
        points_selected = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points_selected) < num_points:
                point_id = len(points_selected)
                marker = MarkerPoint(
                    id=point_id,
                    projector_pos=(x, y),
                    color=MARKER_COLORS[point_id % len(MARKER_COLORS)]
                )
                points_selected.append(marker)
                logger.info(f"Point {point_id} selected at ({x}, {y})")
        
        cv2.namedWindow("Select Marker Points", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Marker Points", 1280, 720)
        cv2.setMouseCallback("Select Marker Points", mouse_callback)
        
        print("\n" + "="*60)
        print("MARKER POINT SELECTION")
        print("="*60)
        print(f"Click {num_points} points on key features:")
        print("  - Corners of buildings")
        print("  - Window corners")
        print("  - Door frames")
        print("  - Any distinctive architectural features")
        print("\nSpread points across the entire house area")
        print("Press ENTER when all points are selected")
        print("Press ESC to cancel")
        print("="*60 + "\n")
        
        while True:
            display = self.reference_image.copy()
            
            # Draw selected points
            for marker in points_selected:
                x, y = int(marker.projector_pos[0]), int(marker.projector_pos[1])
                cv2.circle(display, (x, y), 10, marker.color, -1)
                cv2.circle(display, (x, y), 12, (255, 255, 255), 2)
                cv2.putText(display, str(marker.id), (x + 15, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, marker.color, 2)
            
            # Instructions
            remaining = num_points - len(points_selected)
            if remaining > 0:
                text = f"Click {remaining} more point(s)"
                color = (0, 165, 255)
            else:
                text = "Press ENTER to continue"
                color = (0, 255, 0)
            
            cv2.putText(display, text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow("Select Marker Points", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(points_selected) == num_points:  # ENTER
                break
            elif key == 27:  # ESC
                raise KeyboardInterrupt("Point selection cancelled")
        
        cv2.destroyWindow("Select Marker Points")
        self.markers = points_selected
        
        # Save marker positions
        marker_data = {
            "projector_id": self.projector_id,
            "num_markers": len(self.markers),
            "markers": [
                {
                    "id": m.id,
                    "projector_pos": m.projector_pos.tolist(),
                    "color": m.color
                }
                for m in self.markers
            ]
        }
        marker_path = self.data_dir / f"{self.projector_id}_markers.yaml"
        with open(marker_path, 'w') as f:
            yaml.dump(marker_data, f)
        logger.info(f"Marker configuration saved to {marker_path}")
        
        return self.markers
    
    def create_marker_overlay(self) -> np.ndarray:
        """Create projection overlay with numbered colored markers."""
        overlay = np.zeros((self.proj_height, self.proj_width, 3), dtype=np.uint8)
        
        for marker in self.markers:
            x, y = int(marker.projector_pos[0]), int(marker.projector_pos[1])
            
            # Draw marker circle
            cv2.circle(overlay, (x, y), self.marker_radius, marker.color, -1)
            cv2.circle(overlay, (x, y), self.marker_radius + 3, (255, 255, 255), 3)
            
            # Draw ID number
            text = str(marker.id)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       self.marker_text_size, 3)[0]
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2
            
            # White background for text
            cv2.putText(overlay, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.marker_text_size,
                       (255, 255, 255), 6)
            # Colored text
            cv2.putText(overlay, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.marker_text_size,
                       marker.color, 3)
        
        return overlay
    
    def detect_markers_in_camera(self, frame: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Detect projected markers in camera frame.
        Since all markers are the same color, detect all navy blue blobs
        and match them to markers by spatial proximity (assuming rough alignment).
        
        Returns:
            List of (marker_id, camera_position) tuples
        """
        detections = []
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Since all markers are the same color, detect all navy blue regions at once
        lower_bound, upper_bound = self._get_color_range(self.markers[0].color)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find all contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Get centroids of all detected blobs
        detected_positions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area threshold
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_positions.append(np.array([cx, cy], dtype=np.float32))
        
        # Reset all markers
        for marker in self.markers:
            marker.detected = False
            marker.camera_pos = None
        
        # Simple greedy matching: assign each detected position to nearest marker
        # Assumes camera and projector have similar framing
        used_positions = set()
        
        for marker in self.markers:
            if not detected_positions:
                break
            
            # Normalize projector position to camera frame (rough estimate)
            # Assume both views have similar aspect ratio and scale
            proj_x_norm = marker.projector_pos[0] / self.proj_width
            proj_y_norm = marker.projector_pos[1] / self.proj_height
            
            cam_h, cam_w = frame.shape[:2]
            expected_x = proj_x_norm * cam_w
            expected_y = proj_y_norm * cam_h
            expected_pos = np.array([expected_x, expected_y], dtype=np.float32)
            
            # Find nearest unused detected position
            best_dist = float('inf')
            best_idx = None
            for idx, pos in enumerate(detected_positions):
                if idx in used_positions:
                    continue
                dist = np.linalg.norm(pos - expected_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            if best_idx is not None and best_dist < 200:  # Max distance threshold (pixels)
                marker.camera_pos = detected_positions[best_idx]
                marker.detected = True
                used_positions.add(best_idx)
                detections.append((marker.id, marker.camera_pos))
        
        return detections
    
    def _get_color_range(self, bgr_color: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get HSV range for detecting a BGR color."""
        # Convert BGR to HSV
        bgr = np.uint8([[bgr_color]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        # Define range with wider tolerances for projected markers
        # Convert to int first to avoid overflow, then create uint8 arrays
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        
        # Wider tolerances: H±40, S±100, V±100 (to handle projection lighting variations)
        lower = np.array([max(0, h - 40), max(0, s - 100), max(0, v - 100)], dtype=np.uint8)
        upper = np.array([min(179, h + 40), min(255, s + 100), min(255, v + 100)], dtype=np.uint8)
       
        return lower, upper
    
    def generate_reference_mask(self) -> np.ndarray:
        """Generate mask from the reference image using detection."""
        logger.info("Generating mask from reference image...")
        
        # Detect house features in reference image
        detections_result = self.detector.detect(self.reference_image)
        
        if not detections_result.detections:
            logger.warning("No detections found in reference image")
            return np.zeros(self.reference_image.shape[:2], dtype=np.uint8)
        
        # Build masks using MaskBuilder (same as run_alignment_loop.py)
        mask_result = self.mask_builder.build(
            self.reference_image.shape[:2], 
            detections_result.detections
        )
        
        logger.info(f"Reference mask created: {len(detections_result.detections)} detections, "
                   f"{np.count_nonzero(mask_result.composite_mask)} pixels")
        
        return mask_result.composite_mask
    
    def compute_homography(self) -> Optional[np.ndarray]:
        """Compute homography from detected markers."""
        # Get detected marker correspondences
        projector_points = []
        camera_points = []
        
        for marker in self.markers:
            if marker.detected and marker.camera_pos is not None:
                projector_points.append(marker.projector_pos)
                camera_points.append(marker.camera_pos)
        
        if len(projector_points) < 4:
            logger.warning(f"Not enough markers detected: {len(projector_points)}/8")
            return None
        
        # Compute homography: camera -> projector
        projector_points = np.array(projector_points, dtype=np.float32)
        camera_points = np.array(camera_points, dtype=np.float32)
        
        H, mask = cv2.findHomography(camera_points, projector_points, cv2.RANSAC, 5.0)
        
        if H is not None:
            inliers = np.sum(mask)
            logger.info(f"Homography computed with {inliers}/{len(projector_points)} inliers")
            self.homography = H
        
        return H
    
    def run_alignment_loop(self):
        """Main alignment loop with marker tracking and overlay projection."""
        logger.info("Starting marker-based alignment loop...")
        
        # Initialize detector and mask builder lazily
        self.detector = FalEvfSamDetector(
            classes=self.config.pipeline.detection.classes,
            confidence_threshold=self.config.pipeline.detection.confidence_threshold,
            model_id=self.config.pipeline.detection.evfsam.model_id,
            api_key_env=self.config.pipeline.detection.evfsam.api_key_env,
            prompt_map=self.config.pipeline.detection.evfsam.prompt_map,
            mask_only=self.config.pipeline.detection.evfsam.mask_only,
            fill_holes=self.config.pipeline.detection.evfsam.fill_holes,
            revert_mask=self.config.pipeline.detection.evfsam.revert_mask,
            poll_interval_s=self.config.pipeline.detection.evfsam.poll_interval_s,
            timeout_s=self.config.pipeline.detection.evfsam.timeout_s,
        )
        # Fixed: Pass dilate_kernel and blur_kernel separately
        self.mask_builder = MaskBuilder(
            dilate_kernel=self.config.pipeline.masking.dilate_kernel,
            blur_kernel=self.config.pipeline.masking.blur_kernel
        )
        
        # Generate mask from reference image (only once)
        logger.info("Generating mask from reference image...")
        reference_mask = self.generate_reference_mask()
        
        # Create windows
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 1280, 720)
        
        cv2.namedWindow(f"Overlay_{self.projector_id}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Overlay_{self.projector_id}", self.proj_width, self.proj_height)
        
        # Initial marker overlay
        marker_overlay = self.create_marker_overlay()
        current_overlay = marker_overlay.copy()
        mode = "markers"  # "markers" or "mask"
        
        print("\n" + "="*60)
        print("ALIGNMENT LOOP RUNNING")
        print("="*60)
        print("Controls:")
        print("  M - Toggle between MARKERS and MASK mode")
        print("  S - Save current homography")
        print("  R - Reset to marker mode")
        print("  F - Toggle fullscreen (projector window)")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        frame_count = 0
        detection_interval = 30  # Detect house every 30 frames
        fullscreen = False
        
        while True:
            ret, frame = self.read_camera()
            if not ret:
                continue
            
            frame_count += 1
            
            # Detect markers in camera view
            detections = self.detect_markers_in_camera(frame)
            
            # Log detection status periodically
            if frame_count % 30 == 0:
                detected_ids = [marker.id for marker in self.markers if marker.detected]
                logger.info(f"Marker detection: {len(detections)}/8 markers detected - IDs: {detected_ids}")
            
            # Compute/update homography if enough markers detected
            if len(detections) >= 4:
                self.compute_homography()
            elif frame_count % 30 == 0:
                logger.warning(f"Not enough markers for homography: {len(detections)}/4 needed")
            
            # Visualize on camera view
            vis_frame = frame.copy()
            for marker in self.markers:
                if marker.detected and marker.camera_pos is not None:
                    x, y = int(marker.camera_pos[0]), int(marker.camera_pos[1])
                    cv2.circle(vis_frame, (x, y), 8, marker.color, -1)
                    cv2.circle(vis_frame, (x, y), 10, (255, 255, 255), 2)
                    cv2.putText(vis_frame, str(marker.id), (x + 12, y + 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, marker.color, 2)
            
            # Status text
            detected_count = sum(1 for m in self.markers if m.detected)
            status_color = (0, 255, 0) if detected_count >= 4 else (0, 0, 255)
            cv2.putText(vis_frame, f"Markers: {detected_count}/8", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(vis_frame, f"Mode: {mode.upper()}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.imshow("Camera", vis_frame)
            
            # Generate overlay based on mode
            if mode == "markers":
                current_overlay = marker_overlay
            else:  # mask mode
                # Use pre-generated reference mask and warp to projector space
                if self.homography is not None:
                    # Warp reference mask to projector space using homography
                    mask_warped = cv2.warpPerspective(
                        reference_mask,
                        self.homography,
                        (self.proj_width, self.proj_height)
                    )
                    
                    # Create colored overlay
                    overlay_new = np.zeros((self.proj_height, self.proj_width, 3), dtype=np.uint8)
                    overlay_new[mask_warped > 0] = self.proj_config.overlay_color_bgr
                    current_overlay = overlay_new
                else:
                    # No homography yet, keep showing markers
                    logger.warning("No homography available for mask warping")
            
            cv2.imshow(f"Overlay_{self.projector_id}", current_overlay)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                mode = "mask" if mode == "markers" else "markers"
                logger.info(f"Switched to {mode} mode")
            elif key == ord('s'):
                if self.homography is not None:
                    # Save homography
                    save_path = Path(self.config.calibration.homographies_dir) / f"{self.projector_id}.npy"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(save_path, self.homography)
                    logger.info(f"✓ Homography saved to {save_path}")
                    print(f"\n✓ Homography saved to {save_path}\n")
            elif key == ord('r'):
                mode = "markers"
                current_overlay = marker_overlay
                logger.info("Reset to marker mode")
            elif key == ord('f'):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(f"Overlay_{self.projector_id}", 
                                         cv2.WND_PROP_FULLSCREEN, 
                                         cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(f"Overlay_{self.projector_id}", 
                                         cv2.WND_PROP_FULLSCREEN, 
                                         cv2.WINDOW_NORMAL)
        
        cv2.destroyAllWindows()
        self.release_camera()


def load_env_file(env_path: Path = None) -> None:
    """Load environment variables from .env file."""
    if env_path is None:
        # Look for .env in parent directory
        env_path = Path(__file__).parent.parent.parent / ".env"
    
    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}")
        return
    
    logger.info(f"Loading environment from {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
            # Log with masked value
            masked = value[:10] + "..." if len(value) > 10 else value
            logger.debug(f"Set {key}={masked}")


def main():
    """Run marker-based alignment workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Marker-based projection alignment")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--projector-id", type=str, default="projector_1",
                       help="Projector ID to calibrate")
    parser.add_argument("--num-markers", type=int, default=8,
                       help="Number of calibration markers")
    
    args = parser.parse_args()
    
    # Load environment variables FIRST
    load_env_file()
    
    # Load configuration using the proper loader
    from projector_mapper.config import load_config
    config = load_config(args.config)
    
    # Initialize alignment system
    aligner = MarkerBasedAlignment(config, args.projector_id)
    
    try:
        # Step 1: Capture reference image
        aligner.capture_reference_image()
        
        # Step 2: Select marker points
        aligner.select_marker_points(num_points=args.num_markers)
        
        # Step 3: Run alignment loop
        aligner.run_alignment_loop()
        
    except KeyboardInterrupt:
        print("\nAlignment cancelled by user")
    except Exception as e:
        logger.error(f"Error during alignment: {e}", exc_info=True)
        raise
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    main()