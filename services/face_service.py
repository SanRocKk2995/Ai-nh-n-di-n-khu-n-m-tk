"""
High-level face operations: register, recognize, list, delete.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import FACES_DIR, RECOGNITION_THRESHOLD, FAISS_INDEX_PATH
from database.face_db import FaceDB
from services.embedding_service import EmbeddingService
from services.search_service import SearchService
from utils.logger import log_recognition


class FaceService:
    """Coordinate detection/embedding/search with persistence."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.search_service = SearchService()
        self.db = FaceDB()
        self._ensure_index_loaded()

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register_person(self, person_id: str, name: str, images: List[np.ndarray]) -> int:
        """
        Register a person with multiple images.
        Returns total embeddings stored for that person.
        """
        if not images:
            raise ValueError("No images provided for registration.")

        os.makedirs(os.path.join(FACES_DIR, person_id), exist_ok=True)
        saved_paths = self._save_face_images(person_id, images)

        embeddings = self.embedding_service.batch_generate_embeddings(images)
        self.embedding_service.save_embeddings(person_id, embeddings)

        all_embeddings = self.embedding_service.load_all_embeddings()
        total_for_person = len(all_embeddings.get(person_id, []))

        # Rebuild FAISS index with new data
        self.search_service.build_faiss_index(all_embeddings)

        # Persist metadata
        self.db.upsert_person(person_id, name, saved_paths, total_for_person)
        return total_for_person

    # ------------------------------------------------------------------ #
    # Recognition
    # ------------------------------------------------------------------ #
    def recognize_face(self, image_bgr: np.ndarray) -> Dict[str, object]:
        """Recognize a single face in an image and return match info."""
        self._ensure_index_loaded()
        embedding = self.embedding_service.generate_embedding(image_bgr)
        results = self.search_service.search_similar(embedding, k=5)

        if not results:
            raise ValueError("Recognition failed: index returned no candidates.")

        best_id, score = results[0]
        match = score >= RECOGNITION_THRESHOLD
        person = self.db.get_person(best_id)
        name = person["name"] if person else best_id

        if match:
            log_recognition(best_id, name, score, source="api")
            return {"person_id": best_id, "name": name, "score": round(score, 4)}

        # No confident match
        return {"person_id": None, "name": "unknown", "score": round(score, 4)}

    # ------------------------------------------------------------------ #
    # Database helpers
    # ------------------------------------------------------------------ #
    def get_all_registered_people(self) -> Tuple[List[dict], int]:
        people = self.db.list_people()
        return people, len(people)

    def delete_person(self, person_id: str) -> None:
        # Remove embeddings and metadata
        self.embedding_service.delete_person(person_id)
        self.db.delete_person(person_id)

        # Remove stored images
        person_dir = os.path.join(FACES_DIR, person_id)
        if os.path.isdir(person_dir):
            shutil.rmtree(person_dir, ignore_errors=True)

        # Rebuild or clear index
        all_embeddings = self.embedding_service.load_all_embeddings()
        if all_embeddings:
            self.search_service.build_faiss_index(all_embeddings)
        else:
            # Remove index files if empty
            if os.path.isfile(FAISS_INDEX_PATH):
                os.remove(FAISS_INDEX_PATH)
            meta_path = f"{FAISS_INDEX_PATH}.meta.json"
            if os.path.isfile(meta_path):
                os.remove(meta_path)

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #
    def _save_face_images(self, person_id: str, images: List[np.ndarray]) -> List[str]:
        paths: List[str] = []
        for idx, img in enumerate(images, start=1):
            filename = f"face_{idx}.jpg"
            person_dir = os.path.join(FACES_DIR, person_id)
            os.makedirs(person_dir, exist_ok=True)
            path = os.path.join(person_dir, filename)
            if not cv2.imwrite(path, img):
                raise ValueError(f"Failed to save face image to {path}")
            paths.append(path)
        return paths

    def _ensure_index_loaded(self) -> None:
        if self.search_service.load_faiss_index():
            return
        embeddings = self.embedding_service.load_all_embeddings()
        if embeddings:
            self.search_service.build_faiss_index(embeddings)
