# 07 — MVP-03: Person detection inference API

## What

Deployable **FastAPI** service that runs the mixed **YOLO26s** person detector on uploaded images.

| Item | Value |
| --- | --- |
| Checkpoint | `models/detector/yolo26s_1280_mixed_v1.pt` (~20 MB) |
| Architecture | YOLO26s |
| Inference `imgsz` | **1280** |
| Training mix | VisDrone (aerial) + Caltech Pedestrian (ground-level) |
| Endpoints | `GET /health`, `POST /detect` |
| Streaming | **Not** included (next phase) |

## Why this checkpoint

VisDrone-only YOLO26s was strong on aerial imagery but weak on normal/phone cameras. The mixed model recovered Caltech performance while keeping most VisDrone quality — current MVP candidate for local and free-CPU hosting evaluation.

## Layout

```text
uniform-detection/
├── api/
│   ├── app.py                 # FastAPI app
│   └── requirements.txt       # lean deploy deps
└── models/detector/
    └── yolo26s_1280_mixed_v1.pt
```

Model path is resolved from `Path(__file__)` → repo root. No Lightning, Colab, or machine-specific absolute paths.

## Install

From the repository root (`uniform-detection`):

**Full project env** (data prep + tests + API):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**API-only / host deploy** (smaller):

```powershell
pip install -r api/requirements.txt
```

## Start locally

```powershell
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Test

**Health**

```powershell
curl http://127.0.0.1:8000/health
```

Expected shape: `{"status":"ok","model":"YOLO26s","image_size":1280}`

**Detect** (multipart image upload)

```powershell
curl -X POST http://127.0.0.1:8000/detect -F "file=@path\to\image.jpg"
```

Response shape:

```json
{
  "people": 5,
  "inference_ms": 123.4,
  "image": { "width": 640, "height": 480 },
  "detections": [
    { "confidence": 0.87, "x1": 100, "y1": 50, "x2": 180, "y2": 300 }
  ]
}
```

## Notes for free CPU hosts

- Cold start loads the ~20 MB checkpoint once; first request may be slower than warm inference.
- `imgsz=1280` is accurate but heavier on CPU — measure latency on the target host before promising live FPS.
- CORS currently allows all origins (`*`) for MVP only; tighten before production.
- Do not commit datasets, training `runs/`, or extra `.pt` files; only this named deployment checkpoint is intentionally tracked.
