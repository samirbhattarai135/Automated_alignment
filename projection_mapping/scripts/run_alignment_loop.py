"""Run the full alignment pipeline with live camera feedback."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
from loguru import logger

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

    logger.info("Starting live alignment loop. Press q to exit, f to toggle fullscreen.")
    
    # Track fullscreen state for each window
    fullscreen_state: Dict[str, bool] = {}
    
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
            if not args.headless:
                
                _render_debug_windows(frame, output.overlays, output.tracking)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("f"):
                # Toggle fullscreen for all overlay windows
                for projector_id in output.overlays.keys():
                    window_name = f"Overlay_{projector_id}"
                    is_fullscreen = fullscreen_state.get(window_name, False)
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        fullscreen_state[window_name] = False
                        logger.info(f"Exited fullscreen for {window_name}")
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        fullscreen_state[window_name] = True
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


def _render_debug_windows(frame, overlays: Dict[str, object], tracking: Optional[TrackingResult]) -> None:
    cv2.imshow("Camera", frame)
    
    # Only show the first projector's overlay and mask to reduce system load
    if overlays:
        first_projector_id = next(iter(overlays.keys()))
        overlay = overlays[first_projector_id]
        window_name = f"Overlay_{first_projector_id}"
        # Create window with NORMAL flag to allow fullscreen
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, overlay)
        
        if tracking:
            mask = tracking.masks.get(first_projector_id)
            if mask is not None:
                tracked_window = f"Tracked_{first_projector_id}"
                cv2.namedWindow(tracked_window, cv2.WINDOW_NORMAL)
                cv2.imshow(tracked_window, mask)

if __name__ == "__main__":
    main()
