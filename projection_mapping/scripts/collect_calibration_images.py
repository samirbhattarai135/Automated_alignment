"""Capture camera frames for checkerboard-based calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from loguru import logger

from projector_mapper.config import load_config
from projector_mapper.utils.image_io import save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect calibration images from the camera feed")
    parser.add_argument("--config", type=Path, required=True, help="Path to pipeline YAML configuration")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/calibration/raw"),
        help="Directory to store captured frames",
    )
    parser.add_argument("--count", type=int, default=50, help="Number of frames to capture")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    camera_cfg = cfg.hardware.camera
    capture = cv2.VideoCapture(camera_cfg.device_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)
    capture.set(cv2.CAP_PROP_FPS, camera_cfg.fps)

    if not capture.isOpened():
        raise RuntimeError("Failed to open camera device")

    logger.info("Starting calibration capture. Press space to save a frame, q to quit.")
    saved = 0
    try:
        while saved < args.count:
            ret, frame = capture.read()
            if not ret:
                logger.warning("Camera frame grab failed, retrying...")
                continue
            cv2.imshow("Calibration Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                save_frame(frame, args.output, stem="calibration")
                saved += 1
                logger.info("Captured %d/%d calibration frames", saved, args.count)
            elif key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        logger.info("Capture session complete with %d frames saved", saved)


if __name__ == "__main__":
    main()
