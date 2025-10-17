"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from projector_mapper.config import load_config


def test_load_config_creates_directories(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    data_root = tmp_path / "data"
    yaml.safe_dump(
        {
            "project": {"name": "test", "data_root": str(data_root)},
            "logging": {"level": "INFO", "output": "stdout"},
            "hardware": {
                "camera": {"device_id": 0, "resolution": [640, 480], "fps": 30},
                "projectors": [
                    {"id": "proj_1", "display_id": 1, "resolution": [800, 600]},
                ],
            },
            "pipeline": {
                "detection": {
                    "backend": "roboflow",
                    "classes": ["window"],
                    "confidence_threshold": 0.5,
                    "api": {"endpoint": "https://example.com", "api_key_env": "RF_API"},
                },
                "masking": {"dilate_kernel": 3, "blur_kernel": 5},
                "alignment": {
                    "max_iterations": 5,
                    "reprojection_tolerance_px": 3.0,
                    "feedback_gain": 0.5,
                },
            },
            "calibration": {
                "pattern": {"type": "checkerboard", "inner_corners": [9, 6], "square_size_m": 0.02},
                "camera_intrinsics_path": str(tmp_path / "calib" / "camera.json"),
                "projector_intrinsics_dir": str(tmp_path / "calib" / "projectors"),
                "homographies_dir": str(tmp_path / "calib" / "homographies"),
            },
            "storage": {
                "frames_dir": str(tmp_path / "frames"),
                "masks_dir": str(tmp_path / "masks"),
                "logs_dir": str(tmp_path / "logs"),
            },
        },
        config_path.open("w", encoding="utf-8"),
    )

    cfg = load_config(config_path)

    assert cfg.project.data_root.exists()
    assert cfg.storage.frames_dir.exists()
    assert cfg.calibration.homographies_dir.exists()
