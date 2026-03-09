"""
Flask application supporting both 1:1 verification and 1:N recognition.
"""

import os
import json
import time
import uuid
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
from io import BytesIO

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

# In-memory cache for pending embeddings (cleared after 10 minutes)
# Format: {session_id: {"embedding": np.array, "threshold": float, "multi_face": str, "timestamp": datetime}}
pending_embeddings = {}

def get_or_create_session_id():
    """Get session ID from cookie or create new one."""
    session_id = request.cookies.get('face_verify_session')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

def cleanup_old_sessions():
    """Remove sessions older than 10 minutes."""
    now = datetime.now()
    expired = [sid for sid, data in pending_embeddings.items() 
               if (now - data['timestamp']).total_seconds() > 600]
    for sid in expired:
        del pending_embeddings[sid]

# Initialize face service with lazy loading to avoid startup errors
face_service = None

def get_face_service():
    """Lazy initialization of FaceService."""
    global face_service
    if face_service is None:
        try:
            from services.face_service import FaceService
            face_service = FaceService()
        except Exception as e:
            app.logger.warning(f"FaceService initialization failed: {e}. Using legacy pipeline only.")
    return face_service


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


@app.route("/process_first", methods=["POST"])
def process_first_image():
    """
    POST /process_first
    Process the first image and store embedding in session.
    Returns preview and status.
    
    Accepts multipart/form-data:memory cache.
    Returns preview and session ID via cookie.
    
    Accepts multipart/form-data:
        - image: first face image
        - threshold (optional): float, default 0.25
        - multi_face (optional): "error" | "largest", default "largest"
    
    Returns JSON:
        {
            "status": "success",
            "message": "Image 1 processed. Upload Image 2 to compare.",
            "session_id": "uuid",
            "has_face": true
        }
    """
    path1 = None
    
    try:
        # Cleanup old sessions
        cleanup_old_sessions()
        
        # Get or create session ID
        session_id = get_or_create_session_id()
        
        # Validate upload
        if "image" not in request.files:
            return jsonify({"error": "Image file is required."}), 400
        
        file1 = request.files["image"]
        
        if file1.filename == "":
            return jsonify({"error": "No file selected."}), 400
        
        if not allowed_file(file1.filename):
            return jsonify({"error": f"File '{file1.filename}' is not a supported image format."}), 400
        
        # Read parameters
        try:
            threshold = float(request.form.get("threshold", 0.25))
        except (TypeError, ValueError):
            threshold = 0.25
        
        multi_face = request.form.get("multi_face", "largest")
        if multi_face not in ("error", "largest"):
            multi_face = "largest"
        
        # Save and process
        path1 = save_upload(file1, UPLOAD_FOLDER)
        
        # Detect and extract embedding
        crop1, _ = detect_and_crop(path1, multi_face=multi_face)
        emb1 = get_embedding(crop1)
        
        # Store in memory cache
        pending_embeddings[session_id] = {
            "embedding": emb1,
            "threshold": threshold,
            "multi_face": multi_face,
            "timestamp": datetime.now()
        }
        
        # Create response with session cookie
        response = jsonify({
            "status": "success",
            "message": "✓ Image 1 processed successfully. Now upload Image 2 to compare.",
            "session_id": session_id,
            "has_face": True,
            "embedding_dim": len(emb1)
        })
        
        # Set session cookie (expires in 10 minutes)
        response.set_cookie('face_verify_session', session_id, max_age=600, httponly=True)
        
        return response
    
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 422
    
    except Exception as exc:
        app.logger.exception("Error processing first image")
        return jsonify({"error": f"Internal error: {str(exc)}"}), 500
    
    finally:
        if path1:
            cleanup_file(path1)


@app.route("/compare_second", methods=["POST"])
def compare_second_image():
    """
    POST /compare_second
    Compare the second image with the first image stored in memory cache.
    
    Accepts multipart/form-data:
        - image: second face image
    
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
    path2 = None
    
    try:
        # Get session ID from cookie
        session_id = request.cookies.get('face_verify_session')
        
        if not session_id or session_id not in pending_embeddings:
            return jsonify({
                "error": "Please upload and process Image 1 first.",
                "needs_image1": True
            }), 400
        
        # Validate upload
        if "image" not in request.files:
            return jsonify({"error": "Image file is required."}), 400
        
        file2 = request.files["image"]
        
        if file2.filename == "":
            return jsonify({"error": "No file selected."}), 400
        
        if not allowed_file(file2.filename):
            return jsonify({"error": f"File '{file2.filename}' is not a supported image format."}), 400
        
        # Get stored data from cache
        cached_data = pending_embeddings[session_id]
        emb1 = cached_data["embedding"]
        threshold = cached_data["threshold"]
        multi_face = cached_data["multi_face"]
        
        # Process second image with FAST MODE for quick results
        # Use fast_mode=True: 320x320 detection + skip quality checks = 3-4x faster!
        path2 = save_upload(file2, UPLOAD_FOLDER)
        crop2, _ = detect_and_crop(path2, multi_face=multi_face, fast_mode=True)
        emb2 = get_embedding(crop2, fast_mode=True)
        
        # Compare
        result = verify(emb1, emb2, threshold=threshold)
        
        # Log comparison
        log_comparison(f"session_{session_id[:8]}_image1", path2, result)
        
        # Clear cache after comparison
        del pending_embeddings[session_id]
        
        # Create response and clear cookie
        response = jsonify({**result, "error": None})
        response.set_cookie('face_verify_session', '', expires=0)
        
        return response
    
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 422
    
    except Exception as exc:
        app.logger.exception("Error comparing second image")
        return jsonify({"error": f"Internal error: {str(exc)}"}), 500
    
    finally:
        if path2:
            cleanup_file(path2)


@app.route("/api/register", methods=["POST"])
def api_register():
    """Register a person with multiple base64 images."""
    start = time.perf_counter()
    try:
        fs = get_face_service()
        if fs is None:
            return jsonify({"error": "Face recognition service not available."}), 503
        
        payload = request.get_json(force=True)
        person_id = payload.get("person_id")
        name = payload.get("name")
        images64 = payload.get("images", [])
        if not person_id or not name or not images64:
            return jsonify({"error": "person_id, name, and images are required."}), 400

        from utils.image_utils import decode_base64_image
        images = [decode_base64_image(b64) for b64 in images64]
        total = fs.register_person(person_id, name, images)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return jsonify({"status": "success", "registered": registered, "processing_time_ms": elapsed_ms})
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Error during registration")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """Recognize a single face from a base64 image."""
    start = time.perf_counter()
    try:
        fs = get_face_service()
        if fs is None:
            return jsonify({"error": "Face recognition service not available."}), 503
        
        payload = request.get_json(force=True)
        image_b64 = payload.get("image")
        if not image_b64:
            return jsonify({"error": "image field is required."}), 400
        from utils.image_utils import decode_base64_image
        image = decode_base64_image(image_b64)
        result = fs.recognize_face(image)
        result["processing_time_ms"] = int((time.perf_counter() - start) * 1000)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Error during recognition")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/database/list", methods=["GET"])
def api_database_list():
    """List registered people."""
    fs = get_face_service()
    if fs is None:
        return jsonify({"error": "Face recognition service not available."}), 503
    people, total = fs.get_all_registered_people()
    return jsonify({"people": people, "total": total})


@app.route("/api/database/person/<person_id>", methods=["DELETE"])
def api_delete_person(person_id: str):
    try:
        fs = get_face_service()
        if fs is None:
            return jsonify({"error": "Face recognition service not available."}), 503
        fs.delete_person(person_id)
        return jsonify({"status": "deleted", "person_id": person_id})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Lightweight health and readiness probe."""
    fs = get_face_service()
    status = {
        "status": "ok",
        "faiss_index_loaded": False,
        "num_registered": 0
    }
    if fs is not None:
        try:
            status["faiss_index_loaded"] = os.path.isfile("index/faiss.index")
            _, total = fs.get_all_registered_people()
            status["num_registered"] = total
        except Exception:
            pass
    return jsonify(status)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
