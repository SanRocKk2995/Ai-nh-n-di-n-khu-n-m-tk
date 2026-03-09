"""
face_detector.py
----------------
Detect and crop a face from an image file using InsightFace.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def normalize_lighting(image_bgr: np.ndarray) -> np.ndarray:
    """Normalize lighting for better detection."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Shared model instance (lazy-loaded once)
_app = None


def _get_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(
            name="buffalo_l",
            root="~/.insightface"
        )
        # Lower detection threshold for better sensitivity (default is 0.5)
        _app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
    return _app


def detect_and_crop(image_path: str, multi_face: str = "error"):
    """
    Detect a face in image_path and return the cropped face as a numpy array.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    multi_face : str
        What to do when multiple faces are found:
            - "error"   → raise ValueError (default)
            - "largest" → use the face with the largest bounding box

    Returns
    -------
    tuple (cropped_face: np.ndarray, bbox: list)
        cropped_face – BGR image cropped to the detected face
        bbox         – [x1, y1, x2, y2]

    Raises
    ------
    ValueError
        If no face is found, or multiple faces are found and multi_face="error".
    FileNotFoundError
        If the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    app = _get_app()
    
    # Try detection with multiple strategies for better robustness
    faces = app.get(img)
    
    # Fallback 1: Try with smaller detection size (better for large images)
    if len(faces) == 0:
        temp_app = FaceAnalysis(name="buffalo_l", root="~/.insightface")
        temp_app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.3)
        faces = temp_app.get(img)
    
    # Fallback 2: Try resizing image if it's too large
    if len(faces) == 0:
        h, w = img.shape[:2]
        if max(h, w) > 1920:
            scale = 1920 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            faces = app.get(resized)
            # Scale back bboxes to original size
            if len(faces) > 0:
                for face in faces:
                    face.bbox = face.bbox / scale
                    if hasattr(face, 'kps') and face.kps is not None:
                        face.kps = face.kps / scale
    
    # Fallback 3: Enhance contrast and retry
    if len(faces) == 0:
        enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        faces = app.get(enhanced)
    
    # Fallback 4: Try histogram equalization for dark images
    if len(faces) == 0:
        normalized = normalize_lighting(img)
        faces = app.get(normalized)
    
    # Fallback 5: Try even lower threshold
    if len(faces) == 0:
        temp_app = FaceAnalysis(name="buffalo_l", root="~/.insightface")
        temp_app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.2)
        faces = temp_app.get(img)

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Filter out very small faces (likely background/false positives)
    if len(faces) > 1:
        # Calculate face sizes
        face_sizes = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        max_size = max(face_sizes)
        
        # Keep only faces that are at least 30% of the largest face
        # This filters out small background faces
        filtered_faces = [f for f, size in zip(faces, face_sizes) if size >= max_size * 0.3]
        faces = filtered_faces

    if len(faces) > 1:
        if multi_face == "error":
            raise ValueError(
                f"{len(faces)} faces detected. Please upload an image with exactly one face."
            )
        elif multi_face == "largest":
            # Pick the face with the largest bounding-box area
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )

    face = faces[0]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]

    # Add 30% padding to help with re-detection in embedding stage
    # More context = better embedding quality
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * 0.3)
    pad_y = int(height * 0.3)
    
    x1 = x1 - pad_x
    y1 = y1 - pad_y
    x2 = x2 + pad_x
    y2 = y2 + pad_y

    # Clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    cropped = img[y1:y2, x1:x2]
    return cropped, [x1, y1, x2, y2]
