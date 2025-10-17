# Projector Mapper

Computer vision pipeline for projector-based augmented reality on building facades. Supports detection of architectural features via remote AI APIs, mask generation, and closed-loop alignment of projected imagery to physical structures.

## Key Capabilities

- Camera-projector calibration for single or multiple projectors sharing a fixed camera reference.
- Pluggable detection backend supporting hosted AI APIs (e.g., Roboflow, Vertex AI) and on-device fallbacks.
- Optional FAL.ai EVF-SAM integration for prompt-driven semantic masks without maintaining a custom detector.
- Mask generation and geometric warping to map detected features onto projector coordinate systems.
- Closed-loop feedback using live camera feeds to iteratively refine alignment.
- Color-coded projector overlays tracked in the camera feed for automatic homography correction.
- Config-driven deployment with per-projector calibration metadata.

## Repository Layout

```
projection_mapping/
├── pyproject.toml
├── README.md
├── config/
│   └── default.yaml
├── scripts/
│   ├── collect_calibration_images.py
│   └── run_alignment_loop.py
├── src/
│   └── projector_mapper/
│       ├── __init__.py
│       ├── calibration/
│       │   ├── __init__.py
│       │   └── camera_projector.py
│       ├── config.py
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── roboflow.py
│       ├── geometry/
│       │   ├── __init__.py
│       │   └── homography.py
│       ├── masking/
│       │   ├── __init__.py
│       │   └── mask_builder.py
│       ├── pipeline.py
│       ├── projection/
│       │   ├── __init__.py
│       │   └── mapper.py
│       └── utils/
│           ├── __init__.py
│           └── image_io.py
└── tests/
    ├── __init__.py
    └── test_config.py
```

## Getting Started

1. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```
2. Duplicate `config/default.yaml` and update hardware-specific parameters such as camera resolution, projector IDs, and API credentials (exported as environment variables).
3. Run the calibration collection script to gather camera-projector correspondences:
   ```bash
   python scripts/collect_calibration_images.py --config config/your-config.yaml
   ```
4. After calibration data is processed, launch the live alignment loop:
   ```bash
   python scripts/run_alignment_loop.py --config config/your-config.yaml
   ```

## Detection Backends

`src/projector_mapper/detection` exposes an abstract interface for feature detectors. The `RoboflowDetector` implementation demonstrates integration with a hosted inference API. Implementations should convert API responses into `Detection` objects with bounding polygons and class labels.

Configure the active detector in the YAML file under `pipeline.detection`.

To use the FAL.ai EVF-SAM model instead of Roboflow, switch `pipeline.detection.backend` to `fal_evfsam`, export `FALAI_API_KEY`, and update `pipeline.detection.evfsam.prompt_map` so every façade class maps to its desired text prompt. The detector uploads each camera frame, requests an EVF-SAM mask per prompt, and converts the masks into polygons for downstream alignment.

## Calibration Workflow

- `collect_calibration_images.py` captures synchronized frames and stores them on disk.
- `camera_projector.py` computes homographies for each projector relative to the shared camera frame.
- Intrinsics and extrinsics are persisted in the `calibration` block of the config file for reuse at runtime.
- During operation the pipeline uses overlay color segmentation (configurable per projector) to continuously update the camera-to-projector homographies without manual recalibration.

## Live Alignment Loop

`scripts/run_alignment_loop.py` orchestrates the pipeline:

1. Acquire frame from camera.
2. Request detections via the configured backend.
3. Generate masks for each feature class of interest.
4. Warp masks into projector coordinates and composite into output frames.
5. Send frames to the projector(s) and collect updated camera frames for feedback.
6. Adjust warps based on observed misalignment using photometric error metrics.
7. Repeat with color-segmented overlay feedback to keep homographies fresh even as hardware shifts.

## Testing

Unit tests live under `tests/`. Run with:

```bash
pytest
```

## Next Steps

- Implement additional detection adapters (e.g., Google Vertex Vision, Azure Custom Vision).
- Add temporal smoothing on detections to reduce flicker.
- Integrate structured-light or AprilTag-based calibration for increased robustness.
- Extend to full 3D reconstruction if facade geometry is complex.
