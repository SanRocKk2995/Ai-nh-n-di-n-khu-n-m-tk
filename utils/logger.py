"""
Lightweight CSV logger used by the 1:N recognition flow.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Iterable

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
CSV_PATH = os.path.join(LOG_DIR, "recognitions.csv")

_COLUMNS = ["timestamp", "person_id", "name", "score", "source"]


def _ensure_header() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.isfile(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()


def log_recognition(person_id: str, name: str, score: float, source: str = "api") -> None:
    """
    Append a recognition event to the CSV log.
    """
    _ensure_header()
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "person_id": person_id,
        "name": name,
        "score": round(score, 4),
        "source": source,
    }
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writerow(row)


def list_logs(limit: int | None = None) -> Iterable[dict]:
    """Yield log rows (newest last)."""
    if not os.path.isfile(CSV_PATH):
        return []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if limit is not None:
            rows = rows[-limit:]
        return rows
