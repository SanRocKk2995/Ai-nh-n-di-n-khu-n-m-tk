"""
Face embedding generator built on InsightFace ArcFace.
"""

from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis

from config import EMBEDDING_DIM
from models.detector import FaceDetector


class FaceRecognizer:
    """Compute L2-normalised face embeddings."""

    def __init__(self, ctx_id: int = -1) -> None:
        self.app = FaceAnalysis(name="buffalo_l", root="~/.insightface")
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def embed(
        self,
        image_bgr: np.ndarray,
        face_obj=None,
        landmarks: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generate an embedding for a face image.

        Priority:
        1) Use precomputed normed_embedding from provided face_obj (fast path).
        2) If landmarks are provided, align then embed.
        3) Fallback: run detection internally and embed the best face.
        """
        if face_obj is not None and getattr(face_obj, "normed_embedding", None) is not None:
            emb = face_obj.normed_embedding
            return self._normalize(emb)

        if landmarks is not None:
            aligned = FaceDetector.align(image_bgr, landmarks)
            faces = self.app.get(aligned)
            if not faces:
                raise ValueError("No face found during alignment embedding.")
            emb = faces[0].normed_embedding
            return self._normalize(emb)

        faces = self.app.get(image_bgr)
        if not faces:
            raise ValueError("No face detected for embedding.")
        emb = faces[0].normed_embedding
        return self._normalize(emb)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalise embedding to ensure unit length for cosine/IP search."""
        vec = embedding.astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-10
        vec = vec / norm
        # Ensure expected dimensionality for FAISS
        if vec.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"Unexpected embedding dim {vec.shape[0]}, expected {EMBEDDING_DIM}")
        return vec
