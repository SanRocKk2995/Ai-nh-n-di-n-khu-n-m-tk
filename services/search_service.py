"""
FAISS-based similarity search utilities.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np

from config import EMBEDDING_DIM, FAISS_INDEX_PATH


class SearchService:
    """Manages FAISS index build/load/search operations."""

    def __init__(self, index_path: str = FAISS_INDEX_PATH, embedding_dim: int = EMBEDDING_DIM) -> None:
        self.index_path = index_path
        self.meta_path = f"{index_path}.meta.json"
        self.embedding_dim = embedding_dim
        self._index: faiss.Index | None = None
        self._id_map: List[str] | None = None

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #
    def build_faiss_index(self, embeddings: Dict[str, List[np.ndarray]]) -> faiss.Index:
        """
        Build an IndexFlatIP over all embeddings.
        """
        vectors: List[np.ndarray] = []
        ids: List[str] = []
        for person_id, embs in embeddings.items():
            for emb in embs:
                vectors.append(self._normalize(emb))
                ids.append(person_id)

        if not vectors:
            raise ValueError("No embeddings available to build FAISS index.")

        mat = np.vstack(vectors).astype("float32")
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(mat)

        # Persist meta for mapping index -> person_id
        self._index = index
        self._id_map = ids
        self.save_index(index, ids)
        return index

    def save_index(self, index: faiss.Index | None = None, id_map: Iterable[str] | None = None) -> None:
        """Persist index and id_map to disk."""
        dir_path = os.path.dirname(self.index_path) or "."
        os.makedirs(dir_path, exist_ok=True)
        if index is None:
            index = self._index
        if id_map is None:
            id_map = self._id_map
        if index is None or id_map is None:
            return
        faiss.write_index(index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"ids": list(id_map), "dim": self.embedding_dim}, f)

    def load_faiss_index(self) -> faiss.Index | None:
        """Load index + metadata if present."""
        if self._index is not None:
            return self._index
        if not os.path.isfile(self.index_path) or not os.path.isfile(self.meta_path):
            return None
        index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._index = index
        self._id_map = meta.get("ids", [])
        return index

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search_similar(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k (person_id, score) pairs sorted by cosine similarity."""
        if self._index is None:
            self.load_faiss_index()
        if self._index is None or not self._id_map:
            raise ValueError("FAISS index not built or empty.")

        query = self._normalize(embedding)
        distances, indices = self._index.search(np.array([query]).astype("float32"), k)

        results: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if idx >= len(self._id_map):
                continue
            person_id = self._id_map[idx]
            # Re-rank with explicit cosine for safety
            candidate = self._index.reconstruct(int(idx))
            score = float(np.dot(query, candidate) / (np.linalg.norm(candidate) + 1e-10))
            results.append((person_id, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(v) + 1e-10
        return v / norm
