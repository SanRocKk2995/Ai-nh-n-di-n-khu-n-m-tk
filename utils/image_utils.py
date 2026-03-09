"""
Image utility helpers for encoding/decoding and basic validation.
"""

from __future__ import annotations

import base64
import os
from typing import Tuple

import cv2
import numpy as np


def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decode a base64-encoded image string into a BGR numpy array.
    Supports data-URI prefixes.
    """
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    try:
        binary = base64.b64decode(base64_str)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid base64 image string") from exc

    buf = np.frombuffer(binary, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Decoded image is empty or unsupported format.")
    return img


def encode_image_base64(image_bgr: np.ndarray, ext: str = ".jpg") -> str:
    """Encode BGR image to base64 string."""
    success, buffer = cv2.imencode(ext, image_bgr)
    if not success:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def resize_image(image_bgr: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image maintaining aspect ratio so the longest edge <= max_size.
    """
    h, w = image_bgr.shape[:2]
    if max(h, w) <= max_size:
        return image_bgr
    scale = max_size / float(max(h, w))
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image_bgr, new_size)


def is_image_valid(image_path: str) -> Tuple[bool, str]:
    """
    Quick sanity check for an image on disk.
    Returns (is_valid, reason_if_not).
    """
    if not os.path.isfile(image_path):
        return False, "File does not exist"
    img = cv2.imread(image_path)
    if img is None:
        return False, "Cannot decode image"
    if min(img.shape[:2]) < 10:
        return False, "Image too small"
    return True, ""
