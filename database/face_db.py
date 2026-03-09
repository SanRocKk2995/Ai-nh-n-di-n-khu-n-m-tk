"""
Simple JSON-backed face metadata store.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from config import FACE_DB_FILE


class FaceDB:
    """Persist person metadata (id, name, image paths, counts)."""

    def __init__(self, db_path: str = FACE_DB_FILE) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict[str, dict]:
        if not os.path.isfile(self.db_path):
            return {}
        with open(self.db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def upsert_person(self, person_id: str, name: str, image_paths: List[str], num_embeddings: int) -> None:
        self._data[person_id] = {
            "person_id": person_id,
            "name": name,
            "image_paths": image_paths,
            "num_embeddings": num_embeddings,
        }
        self._save()

    def delete_person(self, person_id: str) -> None:
        if person_id in self._data:
            self._data.pop(person_id)
            self._save()

    def get_person(self, person_id: str) -> Optional[dict]:
        return self._data.get(person_id)

    def list_people(self) -> List[dict]:
        return list(self._data.values())

    def count(self) -> int:
        return len(self._data)
