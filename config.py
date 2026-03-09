"""
Global configuration constants for the face recognition system.

These values centralize thresholds and important paths so they can be tuned
without touching the business logic.
"""

FACE_DETECTION_THRESHOLD: float = 0.5
FACE_SIZE_MIN: int = 50  # pixels
RECOGNITION_THRESHOLD: float = 0.35
EMBEDDING_DIM: int = 512

FAISS_INDEX_PATH: str = "index/faiss.index"

DATA_ROOT: str = "data"
FACES_DIR: str = "data/faces"
EMBEDDINGS_FILE: str = "data/embeddings/embeddings.pkl"
FACE_DB_FILE: str = "data/embeddings/face_db.json"

# Webcam-specific config
WEBCAM_RECOGNITION_COOLDOWN: float = 3.0  # seconds between repeated IDs
MAX_FRAME_EDGE: int = 1280  # resize frames down to this for throughput
