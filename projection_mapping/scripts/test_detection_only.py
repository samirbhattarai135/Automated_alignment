"""Test object detection and mask display without homography alignment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Test object detection with projector display")
    parser.add_argument("--config", type=Path, required=True, help="Path to configuration YAML")
    parser.add_argument("--projector-id", type=str, default="projector_1", help="Projector ID")
    parser.add_argument("--detection-interval", type=int, default=30, help="Detect objects every N frames (default: 30)")
    parser.add_argument("--projector-color", type=str, default="0,0,255", help="Projector overlay color as B,G,R (default: red)")
    return parser.parse_args()


def main():
    load_env_file()
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get projector config
    projector_config = None
    for proj in config.hardware.projectors:
        if proj.id == args.projector_id:
            projector_config = proj
            break
    
    if projector_config is None:
        logger.error(f"Projector {args.projector_id} not found in config")
        return
    
    logger.info(f"Projector: {projector_config.width}x{projector_config.height}")
    
    # Parse projector color
    try:
        color_parts = [int(x.strip()) for x in args.projector_color.split(',')]
        if len(color_parts) != 3:
            raise ValueError("Color must have 3 components")
        projector_color = tuple(color_parts)
    except Exception as e:
        logger.error(f"Invalid projector color format: {args.projector_color}")
        logger.error("Use format: B,G,R (e.g., '0,0,255' for red)")
        return
    
    # Initialize detector and mask builder
    detection_cfg = config.pipeline.detection
    detector = FalEvfSamDetector(
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
    mask_builder = MaskBuilder(
        dilate_kernel=masking_cfg.dilate_kernel,
        blur_kernel=masking_cfg.blur_kernel
    )
    
    # Get camera config
    camera_config = config.hardware.camera
    
    # Open camera
    capture = cv2.VideoCapture(camera_config.device_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
    capture.set(cv2.CAP_PROP_FPS, camera_config.fps)
    
    if not capture.isOpened():
        raise RuntimeError("Failed to open camera")
    
    logger.info(f"Camera opened: {camera_config.width}x{camera_config.height} @ {camera_config.fps}fps")
    
    # Create windows
    cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Green Overlay", cv2.WINDOW_NORMAL)
    cv2.namedWindow(f"Projector_{args.projector_id}", cv2.WINDOW_NORMAL)
    
    # Move projector window to extended display and set fullscreen immediately
    cv2.moveWindow(f"Projector_{args.projector_id}", 1920, 0)
    cv2.waitKey(100)
    cv2.setWindowProperty(f"Projector_{args.projector_id}", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(100)

    logger.info(f"""
============================================================
DETECTION TEST WITH PROJECTOR
============================================================
Detection interval: every {args.detection_interval} frames
Projector color: {projector_color} (Red)
Classes: {detection_cfg.classes}

Controls:
  F - Toggle fullscreen (projector)
  D - Force detection now
  Q - Quit
============================================================
""")
    
    frame_count = 0
    current_mask = None
    cached_mask = None
    mask_cached = False
    fullscreen = False
    
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                logger.warning("Camera frame unavailable, retrying...")
                continue
            
            frame_count += 1
            
            # Detect objects periodically, unless we already have a cached mask
            if (not mask_cached and (frame_count % args.detection_interval == 0 or current_mask is None)):
                logger.info(f"Detecting objects in frame {frame_count}...")
                detections = detector.detect(frame)
                logger.info(f"Found {len(detections.detections)} objects")

                if detections.detections:
                    # Build mask
                    mask_result = mask_builder.build(
                        (frame.shape[0], frame.shape[1]),
                        detections.detections
                    )
                    current_mask = mask_result.composite_mask
                    logger.info(f"Mask created: {np.sum(current_mask > 0)} pixels")

                    # Cache first non-empty mask and stop further detections
                    if not mask_cached and np.sum(current_mask > 0) > 0:
                        cached_mask = current_mask.copy()
                        mask_cached = True
                        logger.info("Cached first non-empty mask; subsequent frames will reuse it until forced re-detection.")

                    # Log detection details
                    for i, det in enumerate(detections.detections):
                        logger.info(f"  Detection {i}: class={det.label}")
                else:
                    logger.warning("No objects detected")
                    current_mask = None
            
            # Create camera display (no green overlay)
            camera_display = frame.copy()

            # Decide which mask to use: cached_mask takes priority when available
            mask_to_use = cached_mask if mask_cached and cached_mask is not None else current_mask

            # Create green overlay window: only green where mask is
            green_overlay = np.zeros_like(frame)
            if mask_to_use is not None:
                green_bgr = np.array([0, 255, 0], dtype=np.uint8)
                green_overlay[mask_to_use > 0] = green_bgr

            # Compute detection count safely
            detections_count = 0
            if 'detections' in locals() and detections is not None:
                detections_count = len(detections.detections)

            # Add status text to camera view
            status_text = f"Frame: {frame_count} | Detections: {detections_count if mask_to_use is not None else 0}"
            cv2.putText(
                camera_display,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Create projector overlay with red mask
            projector_overlay = np.zeros(
                (projector_config.height, projector_config.width, 3),
                dtype=np.uint8
            )
            
            if mask_to_use is not None:
                # Resize mask to projector resolution (simple scaling, no homography)
                mask_resized = cv2.resize(
                    current_mask,
                    (projector_config.width, projector_config.height),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Apply red color to mask
                projector_color_bgr = np.array(projector_color, dtype=np.uint8)
                projector_overlay[mask_resized > 0] = projector_color_bgr
            
            # Display all windows
            cv2.imshow("Camera View", camera_display)
            cv2.imshow("Green Overlay", green_overlay)
            cv2.imshow(f"Projector_{args.projector_id}", projector_overlay)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # Q or ESC
                logger.info("Quit requested")
                break
            elif key == ord('f'):  # Toggle fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.moveWindow(f"Projector_{args.projector_id}", 1920, 0)
                    cv2.setWindowProperty(
                        f"Projector_{args.projector_id}",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN
                    )
                    logger.info("Fullscreen enabled")
                else:
                    cv2.setWindowProperty(
                        f"Projector_{args.projector_id}",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL
                    )
                    logger.info("Fullscreen disabled")
            elif key == ord('d'):  # Force detection
                logger.info("Force detection triggered: clearing cached mask and forcing a new detection")
                # Clear cache and force immediate detection
                mask_cached = False
                cached_mask = None
                current_mask = None
                frame_count = 0  # Will trigger detection on next frame
                
    finally:
        capture.release()
        cv2.destroyAllWindows()
        logger.info("Detection test stopped")


if __name__ == "__main__":
    main()
