"""Run the full alignment pipeline with live camera feedback."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from loguru import logger

from projector_mapper.alignment import AlignmentResult
from projector_mapper.config import load_config
from projector_mapper.feedback import TrackingResult
from projector_mapper.pipeline import AlignmentPipeline


def load_env_file(env_path: Path = None) -> None:
    """Load environment variables from .env file."""
    if env_path is None:
        # Look for .env in parent directories
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
                        logger.debug(f"Set {key}={value[:20]}..." if len(value) > 20 else f"Set {key}={value}")
    else:
        logger.warning("No .env file found - API keys must be set manually")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live alignment loop for projector mapping")
    parser.add_argument("--config", type=Path, required=True, help="Path to pipeline YAML configuration")
    parser.add_argument("--headless", action="store_true", help="Disable on-screen debug windows")
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    cfg = load_config(args.config)
    pipeline = AlignmentPipeline.from_config(cfg)

    camera_cfg = cfg.hardware.camera
    capture = cv2.VideoCapture(camera_cfg.device_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)
    capture.set(cv2.CAP_PROP_FPS, camera_cfg.fps)

    if not capture.isOpened():
        raise RuntimeError("Failed to open camera device")

    logger.info("Starting live alignment loop with ICP refinement. Press q to exit, f to toggle fullscreen.")
    
    # Track fullscreen state for each window
    fullscreen_state: Dict[str, bool] = {}
    
    # Track if windows have been initialized
    windows_initialized = False
    
    # Store the last output for saving on exit
    last_output = None

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                logger.warning("Camera frame unavailable, retrying...")
                continue

            output = pipeline.process_frame(frame)
            last_output = output  # Keep track of the latest output
            
            # Debug: Check overlay content
            if output.overlays:
                for proj_id, overlay in output.overlays.items():
                    non_zero = cv2.countNonZero(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY))
                    if non_zero == 0:
                        logger.warning(f"Overlay for {proj_id} is completely black (no pixels)")
                    else:
                        logger.debug(f"Overlay for {proj_id}: {overlay.shape}, {non_zero} non-zero pixels")
            
            if not args.headless:
                _render_debug_windows(
                    frame,
                    output.overlays,
                    output.tracking,
                    output.alignment_results,  # Pass alignment results
                    len(output.detections.detections) if output.detections else 0,
                    fullscreen_state,
                )
                
                # Set windows to fullscreen after they're created (on second iteration)
                if not windows_initialized and output.overlays:
                    windows_initialized = True
                    # Move projector_1 window to extended display before setting fullscreen
                    first_projector_id = next(iter(output.overlays.keys()))
                    window_name = f"Overlay_{first_projector_id}"
                    
                    # Move to extended display (assuming it's to the right of main display)
                    cv2.moveWindow(window_name, 1920, 0)  # Move to right display
                    cv2.waitKey(100)  # Give time for window to move
                    logger.info(f"Moved {window_name} to extended display at x=1920")
                    
                elif windows_initialized and fullscreen_state:
                    # Apply fullscreen ONLY to projector_1 (first projector)
                    first_projector_id = next(iter(output.overlays.keys()))
                    first_window = f"Overlay_{first_projector_id}"
                    
                    if first_window in fullscreen_state:
                        try:
                            cv2.setWindowProperty(first_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            logger.info(f"Set {first_window} to fullscreen")
                            del fullscreen_state[first_window]  # Remove after applying
                        except cv2.error as e:
                            logger.debug(f"Waiting for window {first_window} to be ready...")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("f"):
                # Toggle fullscreen for the overlay window
                if output.overlays:
                    first_projector_id = next(iter(output.overlays.keys()))
                    window_name = f"Overlay_{first_projector_id}"
                    
                    # Check current state (default to False if not in dict)
                    current_fullscreen = fullscreen_state.get(f"{window_name}_active", False)
                    
                    if current_fullscreen:
                        # Exit fullscreen
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        fullscreen_state[f"{window_name}_active"] = False
                        logger.info(f"Exited fullscreen for {window_name}")
                    else:
                        # Enter fullscreen
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        fullscreen_state[f"{window_name}_active"] = True
                        logger.info(f"Entered fullscreen for {window_name}")
    finally:
        # Save the last overlay before exiting
        if last_output and last_output.overlays:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for projector_id, overlay in last_output.overlays.items():
                output_path = output_dir / f"overlay_{projector_id}_{timestamp}.png"
                cv2.imwrite(str(output_path), overlay)
                logger.info(f"Saved overlay to {output_path}")
        
        capture.release()
        cv2.destroyAllWindows()
        logger.info("Alignment loop stopped")


def _render_debug_windows(
    frame: np.ndarray,
    overlays: Dict[str, np.ndarray],
    tracking: Optional[TrackingResult],
    alignment_results: Optional[Dict[str, AlignmentResult]],
    num_detections: int,
    fullscreen_state: Dict[str, bool],
) -> None:
    """Render debug windows with ICP alignment status."""
    # Add status overlay to camera frame
    status_frame = frame.copy()
    
    # Detection count
    cv2.putText(
        status_frame,
        f"Detections: {num_detections}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    # ICP alignment status
    if alignment_results:
        y_offset = 60
        for projector_id, result in alignment_results.items():
            if result.converged:
                status_text = f"{projector_id}: {result.error:.1f}px ({result.correspondences} pts)"
                color = (0, 255, 0) if result.error < 10 else (0, 255, 255)
            else:
                status_text = f"{projector_id}: No alignment"
                color = (0, 0, 255)
            
            cv2.putText(
                status_frame,
                status_text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            y_offset += 30
    
    # Controls
    cv2.putText(
        status_frame,
        "F=fullscreen Q=quit",
        (10, status_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    cv2.imshow("Camera", status_frame)
    
    # Show overlay windows for all projectors
    if overlays:
        for idx, (projector_id, overlay) in enumerate(overlays.items()):
            window_name = f"Overlay_{projector_id}"
            
            # Create window with explicit size
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Ensure the overlay has content
            if overlay is not None and overlay.size > 0:
                # Resize window to match overlay dimensions
                h, w = overlay.shape[:2]
                
                if idx == 0:
                    # First projector (projector_1): Fullscreen on extended display
                    cv2.resizeWindow(window_name, w, h)
                    cv2.imshow(window_name, overlay)
                    
                    # Mark for fullscreen
                    if window_name not in fullscreen_state:
                        fullscreen_state[window_name] = True
                else:
                    # Second projector (projector_2): Normal window on main display
                    # Show at half size for convenience
                    cv2.resizeWindow(window_name, w // 2, h // 2)
                    cv2.imshow(window_name, overlay)
                    
                    # Position on main display (top right corner)
                    cv2.moveWindow(window_name, 800, 100)
            else:
                # Show black screen if overlay is empty
                black = np.zeros((1080, 1920, 3), dtype=np.uint8)
                cv2.imshow(window_name, black)
                logger.warning(f"Overlay for {projector_id} is empty or invalid")


if __name__ == "__main__":
    main()
