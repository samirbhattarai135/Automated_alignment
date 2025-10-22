"""Project aligned masks using saved homography calibration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from projector_mapper.config import load_config
from projector_mapper.detection.fal_evfsam import FalEvfSamDetector
from projector_mapper.masking.mask_builder import MaskBuilder


def load_env_file(env_path: Path = None) -> None:
    """Load environment variables from .env file."""
    if env_path is None:
        current = Path(__file__).resolve().parent
        for _ in range(5):
            env_file = current / ".env"
            if env_file.exists():
                env_path = env_file
                break
            current = current.parent
    
    if env_path and env_path.exists():
        logger.info(f"Loading environment from {env_path}")
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key and value:
                        os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project aligned masks using saved homography")
    parser.add_argument("--config", type=Path, required=True, help="Path to configuration YAML")
    parser.add_argument("--projector-id", type=str, default="projector_1", help="Projector ID")
    parser.add_argument("--homography", type=Path, help="Path to saved homography (default: auto-detect from config)")
    parser.add_argument("--overlay-color", type=str, default="0,255,0", help="Overlay color as B,G,R (default: green)")
    parser.add_argument("--detection-interval", type=int, default=30, help="Detect objects every N frames (default: 30)")
    return parser.parse_args()


class HomographyProjector:
    """Projects aligned masks using a pre-calibrated homography."""
    
    def __init__(
        self,
        config,
        projector_id: str,
        homography_path: Path,
        overlay_color: tuple[int, int, int],
        detection_interval: int = 30,
    ):
        self.config = config
        self.projector_id = projector_id
        self.overlay_color = overlay_color
        self.detection_interval = detection_interval
        
        # Load homography
        logger.info(f"Loading homography from {homography_path}")
        self.homography = np.load(homography_path)
        logger.info(f"Homography loaded: shape {self.homography.shape}")
        
        # Get projector config
        self.proj_config = None
        for proj in config.hardware.projectors:
            if proj.id == projector_id:
                self.proj_config = proj
                break
        
        if self.proj_config is None:
            raise ValueError(f"Projector {projector_id} not found in config")
        
        logger.info(f"Projector: {self.proj_config.width}x{self.proj_config.height}")
        
        # Initialize detector and mask builder
        detection_cfg = config.pipeline.detection
        self.detector = FalEvfSamDetector(
            classes=detection_cfg.classes,
            confidence_threshold=detection_cfg.confidence_threshold,
            model_id=detection_cfg.evfsam.model_id,
            api_key_env=detection_cfg.evfsam.api_key_env,
            prompt_map=detection_cfg.evfsam.prompt_map,
            mask_only=detection_cfg.evfsam.mask_only,
            fill_holes=detection_cfg.evfsam.fill_holes,
            revert_mask=detection_cfg.evfsam.revert_mask,
            poll_interval_s=detection_cfg.evfsam.poll_interval_s,
            timeout_s=detection_cfg.evfsam.timeout_s,
        )
        
        masking_cfg = config.pipeline.masking
        self.mask_builder = MaskBuilder(
            dilate_kernel=masking_cfg.dilate_kernel,
            blur_kernel=masking_cfg.blur_kernel
        )
        
        # Get camera config
        self.camera_config = config.hardware.camera
        
    def run(self):
        """Run the projection loop."""
        # Open camera
        capture = cv2.VideoCapture(self.camera_config.device_id)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.height)
        capture.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
        
        if not capture.isOpened():
            raise RuntimeError("Failed to open camera")
        
        logger.info(f"Camera opened: {self.camera_config.width}x{self.camera_config.height} @ {self.camera_config.fps}fps")
        
        # Create windows
        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"Projector_{self.projector_id}", cv2.WINDOW_NORMAL)
        
        # Move projector window to extended display
        cv2.moveWindow(f"Projector_{self.projector_id}", 1920, 0)
        cv2.waitKey(100)
        
        logger.info(f"""
============================================================
PROJECTION RUNNING
============================================================
Using saved homography: {self.homography_path}
Overlay color: {self.overlay_color}
Detection interval: every {self.detection_interval} frames

Controls:
  F - Toggle fullscreen (projector window)
  D - Force detection now
  Q - Quit
============================================================
""")
        
        frame_count = 0
        current_mask = None
        fullscreen = False
        
        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    logger.warning("Camera frame unavailable, retrying...")
                    continue
                
                frame_count += 1
                
                # Detect objects periodically
                if frame_count % self.detection_interval == 0 or current_mask is None:
                    logger.info(f"Detecting objects in frame {frame_count}...")
                    detections = self.detector.detect(frame)
                    logger.info(f"Found {len(detections.detections)} objects")
                    
                    if detections.detections:
                        # Build mask
                        mask_result = self.mask_builder.build(
                            (frame.shape[0], frame.shape[1]),
                            detections.detections
                        )
                        current_mask = mask_result.composite_mask
                        logger.info(f"Mask created: {np.sum(current_mask > 0)} pixels")
                    else:
                        logger.warning("No objects detected")
                        current_mask = None
                
                # Create projector overlay
                proj_overlay = np.zeros(
                    (self.proj_config.height, self.proj_config.width, 3),
                    dtype=np.uint8
                )
                
                if current_mask is not None:
                    # Warp mask to projector space using homography
                    mask_warped = cv2.warpPerspective(
                        current_mask,
                        self.homography,
                        (self.proj_config.width, self.proj_config.height)
                    )
                    
                    # Apply color to warped mask
                    overlay_color_bgr = np.array(self.overlay_color, dtype=np.uint8)
                    proj_overlay[mask_warped > 0] = overlay_color_bgr
                
                # Display camera view with detection count
                camera_display = frame.copy()
                status_text = f"Frame: {frame_count} | Mask: {'Active' if current_mask is not None else 'None'}"
                cv2.putText(
                    camera_display,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow("Camera View", camera_display)
                cv2.imshow(f"Projector_{self.projector_id}", proj_overlay)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # Q or ESC
                    logger.info("Quit requested")
                    break
                elif key == ord('f'):  # Toggle fullscreen
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(
                            f"Projector_{self.projector_id}",
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN
                        )
                        logger.info("Fullscreen enabled")
                    else:
                        cv2.setWindowProperty(
                            f"Projector_{self.projector_id}",
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_NORMAL
                        )
                        logger.info("Fullscreen disabled")
                elif key == ord('d'):  # Force detection
                    logger.info("Force detection triggered")
                    frame_count = 0  # Will trigger detection on next frame
                    
        finally:
            capture.release()
            cv2.destroyAllWindows()
            logger.info("Projection stopped")


def main():
    load_env_file()
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine homography path
    if args.homography:
        homography_path = args.homography
    else:
        homography_path = Path(config.calibration.homographies_dir) / f"{args.projector_id}.npy"
    
    if not homography_path.exists():
        logger.error(f"Homography not found: {homography_path}")
        logger.error(f"Please run marker-based calibration first or specify --homography path")
        return
    
    # Parse overlay color
    try:
        color_parts = [int(x.strip()) for x in args.overlay_color.split(',')]
        if len(color_parts) != 3:
            raise ValueError("Color must have 3 components")
        overlay_color = tuple(color_parts)
    except Exception as e:
        logger.error(f"Invalid overlay color format: {args.overlay_color}")
        logger.error("Use format: B,G,R (e.g., '0,255,0' for green)")
        return
    
    # Create projector and run
    projector = HomographyProjector(
        config,
        args.projector_id,
        homography_path,
        overlay_color,
        args.detection_interval,
    )
    
    # Store homography path in projector for logging
    projector.homography_path = homography_path
    
    projector.run()


if __name__ == "__main__":
    main()
