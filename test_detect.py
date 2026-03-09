"""Test face detection and embedding"""
import cv2
import numpy as np
from modules.face_detector import detect_and_crop
from modules.face_embedder import get_embedding

# Test with a sample image
print("Testing face detection and embedding pipeline...")

try:
    # You need to provide actual test image paths
    test_img = "uploads/test.jpg"  # Change this to your test image
    
    print(f"1. Detecting face in {test_img}...")
    crop, bbox = detect_and_crop(test_img, multi_face="largest")
    print(f"   ✓ Face detected at bbox: {bbox}")
    print(f"   ✓ Crop shape: {crop.shape}")
    
    print("2. Generating embedding...")
    embedding = get_embedding(crop)
    print(f"   ✓ Embedding shape: {embedding.shape}")
    print(f"   ✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    print("\n✓ All tests passed!")
    
except FileNotFoundError as e:
    print(f"\n✗ File error: {e}")
    print("   Please create a test image in uploads/test.jpg")
    
except ValueError as e:
    print(f"\n✗ Detection error: {e}")
    
except Exception as e:
    print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
