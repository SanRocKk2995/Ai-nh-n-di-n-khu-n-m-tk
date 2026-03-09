"""Quick test of buffalo_l embedding"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

print("Testing buffalo_l model...")

# Load model
app = FaceAnalysis(name="buffalo_l", root="~/.insightface")
app.prepare(ctx_id=-1, det_size=(640, 640))

# Create a test image
test_img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.putText(test_img, "Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

print(f"Test image shape: {test_img.shape}")

# Try detection
faces = app.get(test_img)
print(f"Detected {len(faces)} faces")

if len(faces) == 0:
    print("No face detected (expected for blank image)")
    print("\nPlease test with an actual face image.")
    print("Try uploading images through the web interface and check Flask logs.")
else:
    face = faces[0]
    print(f"Face bbox: {face.bbox}")
    if hasattr(face, 'normed_embedding'):
        emb = face.normed_embedding
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding dtype: {emb.dtype}")
    if hasattr(face, 'kps'):
        print(f"Keypoints shape: {face.kps.shape if face.kps is not None else None}")
