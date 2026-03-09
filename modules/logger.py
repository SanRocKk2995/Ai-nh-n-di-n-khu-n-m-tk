"""
logger.py
---------
CSV logger for face comparison results.
"""

import csv
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
CSV_PATH = os.path.join(LOG_DIR, "comparisons.csv")

_COLUMNS = ["timestamp", "image1", "image2", "score", "score_pct", "result", "threshold"]


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _ensure_header():
    """Write CSV header if the file doesn't exist or is empty."""
    _ensure_log_dir()
    write_header = not os.path.isfile(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    if write_header:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()


def log_comparison(image1: str, image2: str, result: dict):
    """
    Append a face comparison result to the CSV log.

    Parameters
    ----------
    image1 : str
        Filename or path of the first image.
    image2 : str
        Filename or path of the second image.
    result : dict
        Output from comparator.verify() containing score, result, etc.
    """
    _ensure_header()

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image1": os.path.basename(image1),
        "image2": os.path.basename(image2),
        "score": result.get("score", ""),
        "score_pct": result.get("score_pct", ""),
        "result": result.get("result", ""),
        "threshold": result.get("threshold", ""),
    }

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writerow(row)
