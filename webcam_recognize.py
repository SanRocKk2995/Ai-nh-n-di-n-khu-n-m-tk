"""
Realtime webcam demo for 1:N face recognition using FAISS.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from config import RECOGNITION_THRESHOLD, WEBCAM_RECOGNITION_COOLDOWN, MAX_FRAME_EDGE
from services.face_service import FaceService
from utils.image_utils import resize_image


def main() -> None:
    face_service = FaceService()
    detector = face_service.embedding_service.detector
    recognizer = face_service.embedding_service.recognizer
    search = face_service.search_service
    db = face_service.db

    # Ensure index is ready
    if search.load_faiss_index() is None:
        embeddings = face_service.embedding_service.load_all_embeddings()
        if embeddings:
            search.build_faiss_index(embeddings)
        else:
            print("No embeddings found. Register faces before running the demo.")
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    last_seen = {}
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_image(frame, MAX_FRAME_EDGE)
        detections = detector.detect(frame)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            if min(w, h) < 60:
                continue  # quality gate

            key = f"{int((x1 + x2) / 2)//20}-{int((y1 + y2) / 2)//20}"
            if key in last_seen and (now - last_seen[key]["ts"]) < WEBCAM_RECOGNITION_COOLDOWN:
                label = last_seen[key]["label"]
                score = last_seen[key]["score"]
            else:
                embedding = recognizer.embed(
                    frame,
                    face_obj=det.get("face_obj"),
                    landmarks=det.get("kps"),
                )
                candidates = search.search_similar(embedding, k=5)
                if not candidates:
                    label, score = "unknown", 0.0
                else:
                    best_id, score = candidates[0]
                    label = (
                        db.get_person(best_id)["name"]
                        if db.get_person(best_id)
                        else best_id
                    )
                    if score < RECOGNITION_THRESHOLD:
                        label = "unknown"
                last_seen[key] = {"ts": now, "label": label, "score": score}

            # Draw UI
            color = (0, 200, 0) if label != "unknown" else (0, 0, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({score:.2f})"
            cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Webcam Recognition (press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
