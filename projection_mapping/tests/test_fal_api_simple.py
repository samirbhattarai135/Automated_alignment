"""Simple test to verify FAL.ai EVF-SAM API integration.

This demonstrates the correct usage according to FAL.ai official documentation:
https://fal.ai/models/fal-ai/evf-sam/api?platform=python
"""

import os
import cv2
import numpy as np
import fal_client


def test_fal_api_direct():
    """Test FAL.ai EVF-SAM API directly using official fal-client."""
    
    # Set API key from environment
    api_key = os.getenv("FALAI_API_KEY")
    if not api_key:
        print("⚠️  FALAI_API_KEY not set - skipping test")
        return
    
    os.environ["FAL_KEY"] = api_key
    
    # Create simple test image (house with window and door)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.rectangle(frame, (150, 200), (490, 400), (100, 100, 100), -1)  # House body
    cv2.rectangle(frame, (220, 260), (300, 350), (50, 50, 200), -1)   # Door
    cv2.rectangle(frame, (340, 260), (420, 320), (200, 200, 50), -1)  # Window
    
    # Encode image to base64
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Failed to encode image")
    
    import base64
    image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    
    print("\n🔍 Testing FAL.ai EVF-SAM API...")
    print(f"   Model: fal-ai/evf-sam")
    print(f"   Prompt: 'window'")
    
    # Call FAL.ai API using official method from documentation
    result = fal_client.subscribe(
        "fal-ai/evf-sam",
        arguments={
            "prompt": "window",
            "image_url": f"data:image/png;base64,{image_b64}",
            "mask_only": True,
            "fill_holes": True,
            "revert_mask": False,
        },
        with_logs=False,
    )
    
    print(f"\n✅ API Response received!")
    print(f"   Response keys: {list(result.keys())}")
    
    if "image" in result:
        image_data = result["image"]
        print(f"   Image URL: {image_data.get('url', 'N/A')[:80]}...")
        print(f"   Content type: {image_data.get('content_type', 'N/A')}")
        print(f"   File size: {image_data.get('file_size', 'N/A')} bytes")
        
        # Download and verify mask
        import requests
        mask_url = image_data["url"]
        response = requests.get(mask_url, timeout=30)
        
        if response.status_code == 200:
            buffer = np.frombuffer(response.content, dtype=np.uint8)
            mask = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                print(f"\n✅ Mask downloaded successfully!")
                print(f"   Mask shape: {mask.shape}")
                print(f"   Mask dtype: {mask.dtype}")
                print(f"   Non-zero pixels: {np.count_nonzero(mask)}")
            else:
                print(f"\n❌ Failed to decode mask image")
        else:
            print(f"\n❌ Failed to download mask: HTTP {response.status_code}")
    else:
        print(f"\n❌ Unexpected response format: {result}")
    
    print(f"\n✨ Test complete!")


if __name__ == "__main__":
    test_fal_api_direct()
