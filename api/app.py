"""Person-detection inference API (YOLO26s mixed VisDrone + Caltech).

Run from the repository root:

    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# api/app.py → repository root (uniform-detection/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT / "models" / "detector" / "yolo26s_1280_mixed_v1.pt"
)
IMAGE_SIZE = 1280
MODEL_NAME = "YOLO26s"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Deployment checkpoint not found at {MODEL_PATH}. "
        "Expected models/detector/yolo26s_1280_mixed_v1.pt under the repo root."
    )

app = FastAPI(
    title="Person Detection API",
    version="1.0.0",
    description=(
        "MVP person detector: YOLO26s trained on mixed VisDrone + Caltech, "
        f"inference imgsz={IMAGE_SIZE}."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_origin_regex=r"https://.*\.onrender\.com",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup (portable path; no machine-specific directories).
model = YOLO(str(MODEL_PATH))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "image_size": IMAGE_SIZE,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image",
        )

    data = await file.read()

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image",
        ) from None

    width, height = image.size

    start = time.perf_counter()
    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=0.25,
        max_det=1000,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - start) * 1000

    result = results[0]
    detections: list[dict] = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "confidence": round(float(box.conf[0]), 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            )

    return {
        "people": len(detections),
        "inference_ms": round(inference_ms, 2),
        "image": {
            "width": width,
            "height": height,
        },
        "detections": detections,
    }
