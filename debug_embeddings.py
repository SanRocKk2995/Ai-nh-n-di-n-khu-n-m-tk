"""Debug script to test embeddings and compare approaches"""
import cv2
import numpy as np
from modules.face_detector import detect_and_crop
from modules.face_embedder import get_embedding, normalize_lighting
from modules.comparator import cosine_similarity
import sys

def test_embedding(image_path, label):
    """Test different embedding approaches"""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Image: {image_path}")
    print('='*60)
    
    try:
        # Detect and crop
        crop, bbox = detect_and_crop(image_path, multi_face="largest")
        print(f"✓ Detected face at bbox: {bbox}")
        print(f"  Crop size: {crop.shape}")
        
        # Get embedding with current approach
        emb = get_embedding(crop)
        print(f"✓ Embedding shape: {emb.shape}")
        print(f"  Embedding norm: {np.linalg.norm(emb):.4f}")
        print(f"  Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")
        print(f"  Min: {emb.min():.4f}, Max: {emb.max():.4f}")
        
        return emb, crop
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compare_two_images(path1, path2):
    """Compare two images with different approaches"""
    print("\n" + "="*60)
    print("COMPARISON TEST")
    print("="*60)
    
    emb1, crop1 = test_embedding(path1, "Image 1")
    emb2, crop2 = test_embedding(path2, "Image 2")
    
    if emb1 is not None and emb2 is not None:
        sim = cosine_similarity(emb1, emb2)
        print(f"\n{'='*60}")
        print(f"SIMILARITY SCORE: {sim:.4f} ({sim*100:.2f}%)")
        print(f"{'='*60}")
        
        # Try without normalization
        print("\n--- Testing WITHOUT lighting normalization ---")
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", root="~/.insightface")
        app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
        
        faces1 = app.get(crop1)
        faces2 = app.get(crop2)
        
        if len(faces1) > 0 and len(faces2) > 0:
            emb1_raw = faces1[0].normed_embedding
            emb2_raw = faces2[0].normed_embedding
            sim_raw = cosine_similarity(emb1_raw, emb2_raw)
            print(f"Similarity (no normalization): {sim_raw:.4f} ({sim_raw*100:.2f}%)")
            print(f"Difference: {(sim_raw - sim)*100:.2f}%")
        
        # Try with stronger normalization
        print("\n--- Testing WITH stronger lighting normalization ---")
        norm1 = normalize_lighting(crop1)
        norm2 = normalize_lighting(crop2)
        
        faces1_norm = app.get(norm1)
        faces2_norm = app.get(norm2)
        
        if len(faces1_norm) > 0 and len(faces2_norm) > 0:
            emb1_norm = faces1_norm[0].normed_embedding
            emb2_norm = faces2_norm[0].normed_embedding
            sim_norm = cosine_similarity(emb1_norm, emb2_norm)
            print(f"Similarity (with normalization): {sim_norm:.4f} ({sim_norm*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_embeddings.py <image1_path> <image2_path>")
        sys.exit(1)
    
    compare_two_images(sys.argv[1], sys.argv[2])
