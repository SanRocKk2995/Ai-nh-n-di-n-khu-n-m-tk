"""
comparator.py
-------------
Compare two face embeddings and decide MATCH / NOT MATCH.
"""

import numpy as np

# Default cosine similarity threshold
# Higher threshold for better precision and accuracy
# 0.40 provides good balance between precision and recall
DEFAULT_THRESHOLD = 0.40


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalised embedding vectors.

    For normalised vectors: cosine_sim = dot(emb1, emb2).
    Result is in the range [-1, 1].

    Parameters
    ----------
    emb1, emb2 : np.ndarray
        1-D float32 embedding vectors of the same dimension.

    Returns
    -------
    float
        Cosine similarity score in [-1, 1].
    """
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-10)
    return float(np.dot(emb1, emb2))


def verify(
    emb1: np.ndarray,
    emb2: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Compare two embeddings and return a verification result dict.

    Parameters
    ----------
    emb1, emb2 : np.ndarray
        Face embedding vectors.
    threshold : float
        Minimum cosine similarity to consider a MATCH.

    Returns
    -------
    dict with keys:
        - "match"     : bool
        - "score"     : float  (cosine similarity, rounded to 4 decimals)
        - "score_pct" : float  (score scaled to 0–100 %)
        - "result"    : str    ("MATCH" or "NOT MATCH")
        - "threshold" : float
    """
    score = cosine_similarity(emb1, emb2)
    match = score >= threshold

    # Scale score to a more intuitive 0–100% range
    # cosine similarity for faces is typically in [0, 1] for the same identity
    score_pct = round(max(0.0, min(1.0, score)) * 100, 2)

    return {
        "match": match,
        "score": round(score, 4),
        "score_pct": score_pct,
        "result": "MATCH" if match else "NOT MATCH",
        "threshold": threshold,
    }
