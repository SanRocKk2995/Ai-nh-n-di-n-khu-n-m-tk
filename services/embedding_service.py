"""
Embedding generation and persistence helpers.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List

import numpy as np

from config import EMBEDDINGS_FILE
from models.detector import FaceDetector
from models.recognizer import FaceRecognizer


class EmbeddingService:
    """Generate and persist embeddings keyed by person_id."""

    def __init__(
        self,
        detector: FaceDetector | None = None,
        recognizer: FaceRecognizer | None = None,
        embeddings_path: str = EMBEDDINGS_FILE,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.recognizer = recognizer or FaceRecognizer()
        self.embeddings_path = embeddings_path
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)

    def generate_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        """Detect the best face and return a normalised embedding."""
        detections = self.detector.detect(image_bgr)
        if not detections:
            raise ValueError("No face detected in the provided image.")
        face = detections[0]
        return self.recognizer.embed(
            image_bgr,
            face_obj=face.get("face_obj"),
            landmarks=face.get("kps"),
        )

    def batch_generate_embeddings(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Generate embeddings for a batch of images."""
        return [self.generate_embedding(img) for img in images]

    def save_embeddings(self, person_id: str, embeddings: List[np.ndarray]) -> None:
        """Append embeddings for a person and persist to disk."""
        data = self.load_all_embeddings()
        if person_id not in data:
            data[person_id] = []
        data[person_id].extend([np.asarray(emb, dtype=np.float32) for emb in embeddings])
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(data, f)

    def load_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """Load all stored embeddings keyed by person_id."""
        if not os.path.isfile(self.embeddings_path):
            return {}
        with open(self.embeddings_path, "rb") as f:
            data = pickle.load(f)
        # Ensure consistent dtype
        cleaned: Dict[str, List[np.ndarray]] = {}
        for pid, embs in data.items():
            cleaned[pid] = [np.asarray(e, dtype=np.float32) for e in embs]
        return cleaned

    def delete_person(self, person_id: str) -> None:
        """Remove embeddings for a person."""
        data = self.load_all_embeddings()
        if person_id in data:
            data.pop(person_id)
            with open(self.embeddings_path, "wb") as f:
                pickle.dump(data, f)
