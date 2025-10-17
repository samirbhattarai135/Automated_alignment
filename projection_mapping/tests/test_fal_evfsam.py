"""Integration test for FAL EVF-SAM detection backend.

Run with: pytest tests/test_fal_evfsam.py -v
Requires FALAI_API_KEY environment variable to be set.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from projector_mapper.detection import FalEvfSamDetector


@pytest.fixture
def api_key() -> str:
    """Retrieve FAL API key from environment."""
    key = os.getenv("FALAI_API_KEY")
    if not key:
        pytest.skip("FALAI_API_KEY not set; skipping EVF-SAM integration test")
    return key


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Generate a synthetic test frame (house-like shape)."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Gray background
    # Draw a simple "house" shape
    cv2.rectangle(frame, (150, 200), (490, 400), (100, 100, 100), -1)  # House body
    cv2.rectangle(frame, (220, 260), (300, 350), (50, 50, 200), -1)   # Door
    cv2.rectangle(frame, (340, 260), (420, 320), (200, 200, 50), -1)  # Window
    # Roof triangle
    roof_points = np.array([[150, 200], [320, 100], [490, 200]], np.int32)
    cv2.fillPoly(frame, [roof_points], (80, 80, 80))
    return frame


@pytest.fixture
def detector(api_key: str) -> FalEvfSamDetector:
    """Instantiate EVF-SAM detector with test configuration."""
    return FalEvfSamDetector(
        classes=["window", "door", "house"],
        confidence_threshold=0.5,
        model_id="fal-ai/evf-sam",
        api_key_env="FALAI_API_KEY",
        prompt_map={"window": "window", "door": "door", "house": "house"},
        mask_only=True,
        fill_holes=True,
        revert_mask=False,
        poll_interval_s=0.5,
        timeout_s=30.0,
    )


def test_evfsam_detect_basic(detector: FalEvfSamDetector, sample_frame: np.ndarray) -> None:
    """Test basic detection workflow with EVF-SAM."""
    result = detector.detect(sample_frame)
    
    assert result is not None
    assert result.latency_ms > 0
    assert isinstance(result.detections, list)
    
    print(f"\n✓ EVF-SAM returned {len(result.detections)} detection(s) in {result.latency_ms:.2f} ms")
    
    for detection in result.detections:
        assert detection.label in ["window", "door", "house"]
        assert detection.polygon.shape[0] >= 3
        assert detection.polygon.shape[1] == 2
        assert 0.0 <= detection.confidence <= 1.0
        print(f"  - {detection.label}: {detection.polygon.shape[0]} points, confidence={detection.confidence:.2f}")


def test_evfsam_specific_prompt(detector: FalEvfSamDetector, sample_frame: np.ndarray) -> None:
    """Test detection with a single specific prompt."""
    # Create a detector focused only on "house"
    house_detector = FalEvfSamDetector(
        classes=["house"],
        confidence_threshold=0.3,
        model_id="fal-ai/evf-sam",
        api_key_env="FALAI_API_KEY",
        prompt_map={"house": "house"},
        mask_only=True,
        fill_holes=True,
        revert_mask=False,
        poll_interval_s=0.5,
        timeout_s=30.0,
    )
    
    result = house_detector.detect(sample_frame)
    
    assert result is not None
    print(f"\n✓ House-only detection: {len(result.detections)} detection(s)")
    
    if result.detections:
        for det in result.detections:
            assert det.label == "house"
            bbox = det.bounding_box()
            print(f"  - Bounding box: ({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})")


def test_evfsam_no_detections(detector: FalEvfSamDetector) -> None:
    """Test behavior when no objects are detected (empty frame)."""
    blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Pure white
    
    result = detector.detect(blank_frame)
    
    assert result is not None
    assert isinstance(result.detections, list)
    print(f"\n✓ Blank frame returned {len(result.detections)} detection(s)")


def test_evfsam_saves_debug_masks(detector: FalEvfSamDetector, sample_frame: np.ndarray, tmp_path: Path) -> None:
    """Test detection and save resulting masks for visual inspection."""
    result = detector.detect(sample_frame)
    
    output_dir = Path("/Users/samir/Projects/Illumibot_alignment/Automated_alignment/projection_mapping/output/test_evfsam_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original frame
    original_path = output_dir / "original.png"
    cv2.imwrite(str(original_path), sample_frame)
    
    # Save detections as overlays
    overlay = sample_frame.copy()
    for i, detection in enumerate(result.detections):
        polygon = detection.polygon.astype(np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0) if detection.label == "house" else (255, 0, 0)
        cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=2)
        
        # Label the detection
        centroid = detection.polygon.mean(axis=0).astype(int)
        cv2.putText(
            overlay,
            detection.label,
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    
    overlay_path = output_dir / "detections_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    
    print(f"\n✓ Debug images saved to {output_dir}")
    print(f"  - {original_path}")
    print(f"  - {overlay_path}")
    
    assert original_path.exists()
    assert overlay_path.exists()


@pytest.mark.parametrize("prompt_text", ["window", "door", "roof"])
def test_evfsam_multiple_prompts(api_key: str, sample_frame: np.ndarray, prompt_text: str) -> None:
    """Test detection with different prompts."""
    detector = FalEvfSamDetector(
        classes=[prompt_text],
        confidence_threshold=0.3,
        model_id="fal-ai/evf-sam",
        api_key_env="FALAI_API_KEY",
        prompt_map={prompt_text: prompt_text},
        mask_only=True,
        fill_holes=True,
        revert_mask=False,
        poll_interval_s=0.5,
        timeout_s=30.0,
    )
    
    result = detector.detect(sample_frame)
    
    assert result is not None
    print(f"\n✓ Prompt '{prompt_text}': {len(result.detections)} detection(s)")


def test_evfsam_real_image(detector: FalEvfSamDetector, tmp_path: Path) -> None:
    """Test with real images from the data folder."""
    data_dir = Path(__file__).parent.parent / "data"
    
    # Test with both available images
    test_images = ["test_house.jpg", "real_test.jpg"]
    
    for image_name in test_images:
        image_path = data_dir / image_name
        
        if not image_path.exists():
            print(f"\n⚠ Skipping {image_name} - file not found")
            continue
            
        frame = cv2.imread(str(image_path))
        
        if frame is None:
            print(f"\n⚠ Failed to load {image_name}")
            continue
        
        print(f"\n✓ Testing with {image_name} (shape: {frame.shape})")
        result = detector.detect(frame)
        
        assert result is not None
        print(f"  - Found {len(result.detections)} detection(s) in {result.latency_ms:.2f} ms")
        
        for det in result.detections:
            print(f"    • {det.label}: {det.polygon.shape[0]} vertices, confidence={det.confidence:.2f}")
        
        # Save debug output
        output_dir = Path(__file__).parent.parent / "output" / "test_evfsam_debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        overlay = frame.copy()
        for detection in result.detections:
            polygon = detection.polygon.astype(np.int32).reshape((-1, 1, 2))
            color = (0, 255, 0) if detection.label == "window" else (255, 0, 0) if detection.label == "door" else (0, 0, 255)
            cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=3)
            
            # Label the detection
            centroid = detection.polygon.mean(axis=0).astype(int)
            cv2.putText(
                overlay,
                f"{detection.label} {detection.confidence:.2f}",
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        
        output_path = output_dir / f"detections_{image_name}"
        cv2.imwrite(str(output_path), overlay)
        print(f"  - Saved overlay to {output_path}")


if __name__ == "__main__":
    # Allow running directly for quick checks
    pytest.main([__file__, "-v", "-s"])
