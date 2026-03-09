"""Download buffalo_l model for InsightFace"""
import os
from insightface.app import FaceAnalysis

print("Downloading buffalo_l model...")
print("This may take a few minutes depending on your internet connection.")

try:
    # This will trigger auto-download if model doesn't exist
    app = FaceAnalysis(name='buffalo_l', root='~/.insightface')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("\n✓ buffalo_l model downloaded and loaded successfully!")
    print(f"Model location: {os.path.expanduser('~/.insightface/models/buffalo_l')}")
except Exception as e:
    print(f"\n✗ Error downloading model: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: download manually from URL
    import urllib.request
    import zipfile
    
    model_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    model_dir = os.path.expanduser("~/.insightface/models")
    os.makedirs(model_dir, exist_ok=True)
    
    zip_path = os.path.join(model_dir, "buffalo_l.zip")
    
    print(f"Downloading from {model_url}...")
    urllib.request.urlretrieve(model_url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)
    
    print("✓ buffalo_l model downloaded successfully!")
    os.remove(zip_path)
