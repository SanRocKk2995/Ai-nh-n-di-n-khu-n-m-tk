"""
utils.py
--------
Utility helpers: file validation, saving uploads, reading images.
"""

import os
import uuid
import cv2
import numpy as np

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


def allowed_file(filename: str) -> bool:
    """Return True if filename has an allowed image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_upload(file_storage, folder: str) -> str:
    """
    Save a Flask FileStorage object to `folder` with a UUID-based filename.

    Parameters
    ----------
    file_storage : werkzeug.datastructures.FileStorage
        The uploaded file from Flask's request.files.
    folder : str
        Absolute path to the destination folder.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    os.makedirs(folder, exist_ok=True)
    ext = file_storage.filename.rsplit(".", 1)[-1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(folder, filename)
    file_storage.save(path)
    return path


def read_image(path: str) -> np.ndarray:
    """
    Load an image from disk as a BGR numpy array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or cannot be decoded.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image from: {path}")
    return img


def cleanup_file(path: str):
    """Delete a file silently (no error if it doesn't exist)."""
    try:
        os.remove(path)
    except OSError:
        pass
