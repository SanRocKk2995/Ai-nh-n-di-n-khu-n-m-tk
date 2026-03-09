"""
app.py
------
Flask application for face verification (1:1 comparison).
"""

import os
import json
from flask import Flask, render_template, request, jsonify

from modules.face_detector import detect_and_crop
from modules.face_embedder import get_embedding
from modules.comparator import verify
from modules.logger import log_comparison
from modules.utils import allowed_file, save_upload, cleanup_file

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/verify", methods=["POST"])
def verify_faces():
    """
    POST /verify
    Accepts multipart/form-data with:
        - image1: first face image
        - image2: second face image
        - threshold (optional): float, default 0.40
        - multi_face (optional): "error" | "largest", default "error"

    Returns JSON:
        {
            "match": bool,
            "score": float,
            "score_pct": float,
            "result": "MATCH" | "NOT MATCH",
            "threshold": float,
            "error": null | str
        }
    """
    path1 = None
    path2 = None

    try:
        # ── Validate uploads ──────────────────────────────────────────────
        if "image1" not in request.files or "image2" not in request.files:
            return jsonify({"error": "Both image1 and image2 are required."}), 400

        file1 = request.files["image1"]
        file2 = request.files["image2"]

        if file1.filename == "" or file2.filename == "":
            return jsonify({"error": "No file selected for one or both images."}), 400

        if not allowed_file(file1.filename):
            return jsonify({"error": f"File '{file1.filename}' is not a supported image format."}), 400
        if not allowed_file(file2.filename):
            return jsonify({"error": f"File '{file2.filename}' is not a supported image format."}), 400

        # ── Read optional parameters ──────────────────────────────────────
        try:
            threshold = float(request.form.get("threshold", 0.25))
        except (TypeError, ValueError):
            threshold = 0.25

        multi_face = request.form.get("multi_face", "largest")
        if multi_face not in ("error", "largest"):
            multi_face = "largest"

        # ── Save uploads ──────────────────────────────────────────────────
        path1 = save_upload(file1, UPLOAD_FOLDER)
        path2 = save_upload(file2, UPLOAD_FOLDER)

        # ── Pipeline: detect → embed → compare ───────────────────────────
        crop1, _ = detect_and_crop(path1, multi_face=multi_face)
        crop2, _ = detect_and_crop(path2, multi_face=multi_face)

        emb1 = get_embedding(crop1)
        emb2 = get_embedding(crop2)

        result = verify(emb1, emb2, threshold=threshold)

        # ── Log to CSV ────────────────────────────────────────────────────
        log_comparison(path1, path2, result)

        return jsonify({**result, "error": None})

    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 422

    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Unexpected error during verification")
        return jsonify({"error": f"Internal error: {str(exc)}"}), 500

    finally:
        # Always clean up uploaded temp files
        if path1:
            cleanup_file(path1)
        if path2:
            cleanup_file(path2)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
