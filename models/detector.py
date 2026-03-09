"""
Wrapper around InsightFace detector with quality checks and alignment helpers.
"""

from __future__ import annotations

import cv2
import numpy as np
import skimage.transform as trans
from insightface.app import FaceAnalysis

from config import FACE_DETECTION_THRESHOLD, FACE_SIZE_MIN

# Standard ArcFace alignment template for 112x112 faces
ARCFACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceDetector:
    """Detect faces and provide aligned crops suitable for recognition."""

    def __init__(self, det_size: tuple[int, int] = (640, 640), ctx_id: int = -1) -> None:
        self.app = FaceAnalysis(name="buffalo_l", root="~/.insightface")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=FACE_DETECTION_THRESHOLD)

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR image and filter by size/score.

        Returns
        -------
        list of dict with keys:
            - bbox: list[int, int, int, int]
            - kps: np.ndarray shape (5, 2)
            - score: float
            - face_obj: original InsightFace face object (for embeddings)
        """
        faces = self.app.get(image_bgr)
        results: list[dict] = []

        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            w, h = x2 - x1, y2 - y1
            if min(w, h) < FACE_SIZE_MIN:
                continue
            if float(face.det_score) < FACE_DETECTION_THRESHOLD:
                continue
            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "kps": face.kps.astype(np.float32) if face.kps is not None else None,
                    "score": float(face.det_score),
                    "face_obj": face,
                }
            )

        # Highest confidence first
        results.sort(key=lambda f: f["score"], reverse=True)
        return results

    @staticmethod
    def align(image_bgr: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
        """
        Align a face crop using 5-point landmarks.
        """
        if landmarks is None or landmarks.shape != (5, 2):
            raise ValueError("Cannot align face: invalid landmarks")

        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, ARCFACE_SRC)
        aligned = cv2.warpAffine(
            image_bgr,
            tform.params[0:2, :],
            (output_size, output_size),
            borderValue=0.0,
        )
        return aligned
