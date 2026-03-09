# Face Verification System

A **1:1 face verification** web application — upload two face images to check if they belong to the same person.

Built with: **Python · InsightFace · OpenCV · Flask**

---

## ✨ Features

- Detect and crop faces automatically
- Generate 512-dimensional face embeddings via InsightFace (`buffalo_sc`)
- Cosine similarity comparison with configurable threshold
- Instant MATCH / NOT MATCH result with similarity score
- CSV logging of all comparisons (`logs/comparisons.csv`)
- Clean, dark glassmorphism web UI

---

## 📁 Project Structure

```
face_verification_project/
├── app.py                  # Flask application entry point
├── requirements.txt
├── README.md
│
├── modules/
│   ├── face_detector.py    # Face detection + cropping (InsightFace)
│   ├── face_embedder.py    # Embedding generation
│   ├── comparator.py       # Cosine similarity + threshold decision
│   ├── logger.py           # CSV logging
│   └── utils.py            # File helpers
│
├── templates/
│   └── index.html          # Web UI
├── static/
│   └── style.css           # Dark glassmorphism styling
│
├── uploads/                # Temporary uploaded images (auto-cleaned)
├── results/                # Reserved for future use
└── logs/
    └── comparisons.csv     # Auto-generated comparison log
```

---

## 🚀 Installation

### 1. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** InsightFace downloads the `buffalo_sc` model weights (~350 MB) on **first run**. An internet connection is required for this step.

---

## ▶️ Running the App

```powershell
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 🧪 Testing

### Web UI — Manual Tests

| Test | Expected |
|------|----------|
| Upload same person twice | ✅ **MATCH**, score ≥ 40% |
| Upload two different people | ❌ **NOT MATCH**, score < 40% |
| Upload image with no face | ⚠️ Error: "No face detected" |
| Upload image with multiple faces (default mode) | ⚠️ Error: "X faces detected" |
| Upload image with multiple faces (use "largest face" option) | Uses biggest face |

### Checking the CSV log

```powershell
Get-Content logs\comparisons.csv
```

---

## ⚙️ Configuration

Inside the web UI click **Advanced Options** to adjust:

| Option | Default | Description |
|--------|---------|-------------|
| Threshold | `0.40` | Minimum cosine similarity for MATCH |
| Multiple Faces | `error` | What to do when >1 face is found |

---

## 🔌 API Reference

### `POST /verify`

Accepts `multipart/form-data`:

| Field | Type | Required |
|-------|------|----------|
| `image1` | file | ✅ |
| `image2` | file | ✅ |
| `threshold` | float | ❌ (default 0.40) |
| `multi_face` | `error` \| `largest` | ❌ |

Response (JSON):

```json
{
  "match": true,
  "score": 0.7231,
  "score_pct": 72.31,
  "result": "MATCH",
  "threshold": 0.4,
  "error": null
}
```

---

## 📖 Algorithm Pipeline

```
Image A ──► detect face ──► crop ──► embed ──► emb_A ──┐
                                                         ├──► cosine_similarity ──► threshold ──► MATCH / NOT MATCH
Image B ──► detect face ──► crop ──► embed ──► emb_B ──┘
```

---

## Multi-Agent Workflow

This project was orchestrated by **Antigravity** using the `task-router` skill:

| Role | Agent |
|------|-------|
| Orchestration | Antigravity |
| Implementation | Codex CLI (executed by Antigravity) |
| Review | Copilot CLI |