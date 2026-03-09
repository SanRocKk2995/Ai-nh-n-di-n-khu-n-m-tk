# Face Recognition System (1:1 + 1:N)

Upgrade of the original verification app to support **1:N face recognition** with FAISS while keeping the 1:1 `/verify` endpoint.

Built with: **Python · InsightFace · OpenCV · Flask · FAISS**

---

## ✨ What's New

- 1:N recognition using FAISS `IndexFlatIP` + cosine re-ranking
- Face database with registration API and metadata store
- Batch embedding generation and persisted embeddings
- Realtime webcam demo with cooldown cache & FPS overlay
- Backward-compatible 1:1 verification UI/endpoint

---

## 📁 Project Structure

```
project/
├── app.py                   # Flask app (1:1 + new APIs)
├── config.py                # Thresholds & paths
├── webcam_recognize.py      # Realtime webcam demo
├── requirements.txt
├── README.md
│
├── models/
│   ├── detector.py          # InsightFace detector wrapper + alignment
│   └── recognizer.py        # ArcFace embedding wrapper
├── services/
│   ├── face_service.py      # Register/recognize orchestration
│   ├── embedding_service.py # Detection + embedding + persistence
│   └── search_service.py    # FAISS build/load/search
├── database/
│   └── face_db.py           # Person metadata store (JSON)
├── utils/
│   ├── image_utils.py       # Base64 helpers, resize, validation
│   └── logger.py            # Recognition CSV logger
├── data/
│   ├── faces/               # Stored registration images
│   └── embeddings/          # embeddings.pkl + face_db.json
├── index/
│   └── faiss.index          # Saved FAISS index (auto-built)
│
├── modules/                 # Legacy 1:1 pipeline (unchanged)
│   ├── face_detector.py
│   ├── face_embedder.py
│   ├── comparator.py
│   ├── logger.py
│   └── utils.py
└── tests/                   # Optional sample-driven tests
    ├── test_register.py
    ├── test_recognize.py
    └── test_faiss.py
```

---

## 🚀 Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> InsightFace downloads model weights (buffalo_l) on first run. Ensure network access for that step.

---

## ▶️ Run

```powershell
python app.py
```

Open **http://127.0.0.1:5000** for the legacy 1:1 UI. New APIs are listed below.

---

## 🔌 API (JSON)

### Register (`POST /api/register`)
```json
{
  "person_id": "john_001",
  "name": "John Doe",
  "images": ["<base64_face1>", "<base64_face2>"]
}
```
Response: `{"status": "success", "registered": 2, "processing_time_ms": 45}`

### Recognize (`POST /api/recognize`)
```json
{ "image": "<base64_frame>" }
```
Response: `{"person_id": "john_001", "name": "John Doe", "score": 0.82, "processing_time_ms": 45}`

### Database
- `GET /api/database/list` → list people + counts
- `DELETE /api/database/person/<person_id>` → remove person + embeddings

### Health
- `GET /api/health` → `{status, faiss_index_loaded, num_registered}`

Legacy 1:1 verification remains at `POST /verify` (multipart form-data).

---

## 🎥 Webcam Demo

```powershell
python webcam_recognize.py
```
- Skips repeated recognitions within 3s per face slot
- Draws bbox, name, score, and FPS; press **q** to quit

---

## 🧪 Testing

Optional sample-driven tests (skip automatically if sample images missing):

```powershell
python -m unittest tests/test_faiss.py
python -m unittest tests/test_register.py
python -m unittest tests/test_recognize.py
```

Place sample faces at `tests/data/person1_1.jpg` and `tests/data/person1_2.jpg` to run register/recognize tests.

---

## ⚙️ Key Config (config.py)

- `FACE_DETECTION_THRESHOLD`: 0.5
- `RECOGNITION_THRESHOLD`: 0.35 (cosine/IP)
- `FAISS_INDEX_PATH`: `index/faiss.index`
- `DATA_ROOT`: `data/`

---

## 1:1 Verification (Backward Compatible)

The original workflow is unchanged: upload two images to `/verify` and receive MATCH / NOT MATCH with cosine similarity.
