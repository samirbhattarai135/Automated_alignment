"""Test EVF-SAM detection on the test_house.jpg image.

This script loads the test house image and runs FAL.ai EVF-SAM detection
to identify windows, doors, and other architectural features.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import cv2
import numpy as np
from projector_mapper.detection import FalEvfSamDetector


def main():
    # Paths
    image_path = project_root / "data" / "real_test.jpg"
    output_dir = project_root / "data" / "masks"
    output_dir.mkdir(exist_ok=True)
    
    print(f"🏠 Testing EVF-SAM on test house image")
    print(f"   Image: {image_path}")
    print(f"   Output: {output_dir}")
    
    # Check API key
    api_key = os.getenv("FALAI_API_KEY")
    if not api_key:
        print("\n❌ Error: FALAI_API_KEY not set in environment")
        print("   Set it in .env file or export it:")
        print("   export FALAI_API_KEY='your-key-here'")
        return 1
    
    # Load image
    if not image_path.exists():
        print(f"\n❌ Error: Image not found at {image_path}")
        return 1
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"\n❌ Error: Failed to load image")
        return 1
    
    print(f"\n✓ Image loaded successfully")
    print(f"   Shape: {frame.shape}")
    print(f"   Size: {frame.shape[1]}x{frame.shape[0]} pixels")
    
    # Initialize detector
    print(f"\n🔍 Initializing FAL.ai EVF-SAM detector...")
    detector = FalEvfSamDetector(
        classes=["window", "door"],
        confidence_threshold=0.3,
        model_id="fal-ai/evf-sam",
        api_key_env="FALAI_API_KEY",
        prompt_map={
            "window": "window",
            "door": "door",
        },
        mask_only=True,
        fill_holes=True,
        revert_mask=False,
        poll_interval_s=0.5,
        timeout_s=60.0,
    )
    
    print(f"✓ Detector initialized")
    print(f"   Classes: {detector.supported_classes}")
    
    # Run detection
    print(f"\n🔄 Running detection (this may take 20-60 seconds)...")
    print(f"   Processing {len(detector.supported_classes)} prompts...")
    
    result = detector.detect(frame)
    
    print(f"\n✅ Detection complete!")
    print(f"   Total detections: {len(result.detections)}")
    print(f"   Latency: {result.latency_ms:.2f} ms")
    
    # Group by label
    by_label = {}
    for det in result.detections:
        if det.label not in by_label:
            by_label[det.label] = []
        by_label[det.label].append(det)
    
    print(f"\n📊 Detection breakdown:")
    for label, dets in by_label.items():
        print(f"   {label}: {len(dets)}")
    
    # Visualize results
    print(f"\n🎨 Creating visualizations...")
    
    # 1. Draw all detections on original image
    vis_all = frame.copy()
    colors = {
        "window": (0, 255, 0),      # Green
        "door": (0, 0, 255),         # Red
    }
    
    for det in result.detections:
        color = colors.get(det.label, (255, 255, 255))
        pts = det.polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_all, [pts], True, color, 3)
        
        # Add label
        if len(det.polygon) > 0:
            cx = int(np.mean(det.polygon[:, 0]))
            cy = int(np.mean(det.polygon[:, 1]))
            cv2.putText(vis_all, det.label, (cx-30, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    output_vis = output_dir / "test_house_detections_all.jpg"
    cv2.imwrite(str(output_vis), vis_all)
    print(f"   ✓ Saved: {output_vis.name}")
    
    # 2. Create separate mask for each class
    for label in by_label.keys():
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for det in by_label[label]:
            pts = det.polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        
        output_mask = output_dir / f"test_house_mask_{label.replace(' ', '_')}.png"
        cv2.imwrite(str(output_mask), mask)
        print(f"   ✓ Saved: {output_mask.name}")
        
        # Create colored overlay
        overlay = frame.copy()
        color_mask = np.zeros_like(frame)
        color_mask[mask > 0] = colors.get(label, (255, 255, 255))
        overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
        
        output_overlay = output_dir / f"test_house_overlay_{label.replace(' ', '_')}.jpg"
        cv2.imwrite(str(output_overlay), overlay)
        print(f"   ✓ Saved: {output_overlay.name}")
    
    # 3. Create combined mask
    mask_combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    for det in result.detections:
        pts = det.polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask_combined, [pts], 255)
    
    output_combined = output_dir / "test_house_mask_combined.png"
    cv2.imwrite(str(output_combined), mask_combined)
    print(f"   ✓ Saved: {output_combined.name}")
    
    print(f"\n✨ Test complete!")
    print(f"\n📁 Output files saved to: {output_dir}")

    print(f"   Total features detected: {len(result.detections)}")
    for label, dets in sorted(by_label.items()):
        print(f"   - {label}: {len(dets)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
