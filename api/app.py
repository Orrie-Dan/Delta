"""Person-detection inference API (YOLO26s ONNX Runtime CPU).

Run from the repository root:

    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# api/app.py → repository root (uniform-detection/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT / "models" / "detector" / "yolo26s_1280_mixed_v1.onnx"
)
IMAGE_SIZE = 1280
MODEL_STRIDE = 32
MODEL_NAME = "YOLO26s"
RUNTIME_NAME = "onnxruntime"
EXECUTION_PROVIDER = "CPUExecutionProvider"
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.6
MAX_DETECTIONS = 1000
LETTERBOX_COLOR = 114

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Deployment ONNX model not found at {MODEL_PATH}. "
        "Expected models/detector/yolo26s_1280_mixed_v1.onnx under the repo root."
    )

LOCAL_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def _cors_origins() -> list[str]:
    origins = list(LOCAL_ORIGINS)
    frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()
    if frontend_origin:
        origins.append(frontend_origin.rstrip("/"))
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(origins))


app = FastAPI(
    title="Person Detection API",
    version="1.1.0",
    description=(
        "MVP person detector: YOLO26s ONNX Runtime CPU, "
        f"inference imgsz={IMAGE_SIZE}."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup. CPU only — no Ultralytics/PyTorch/OpenCV.
session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=[EXECUTION_PROVIDER],
)
INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


def letterbox_meta(
    original_width: int,
    original_height: int,
    model_imgsz: int = IMAGE_SIZE,
    stride: int = MODEL_STRIDE,
) -> dict[str, float | int]:
    """Ultralytics LetterBox(auto=True, stride=32) semantics."""
    if original_width <= 0 or original_height <= 0:
        raise ValueError("Image has invalid dimensions.")

    ratio = min(model_imgsz / original_width, model_imgsz / original_height)
    resized_width = int(round(original_width * ratio))
    resized_height = int(round(original_height * ratio))

    pad_w = (model_imgsz - resized_width) % stride
    pad_h = (model_imgsz - resized_height) % stride
    pad_x = int(round(pad_w / 2 - 0.1))
    pad_y = int(round(pad_h / 2 - 0.1))
    pad_right = int(round(pad_w / 2 + 0.1))
    pad_bottom = int(round(pad_h / 2 + 0.1))

    return {
        "original_width": original_width,
        "original_height": original_height,
        "ratio": ratio,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "canvas_width": resized_width + pad_x + pad_right,
        "canvas_height": resized_height + pad_y + pad_bottom,
        "model_imgsz": model_imgsz,
    }


def prepare_model_input(image: Image.Image) -> tuple[np.ndarray, dict[str, float | int]]:
    """Letterbox + float32 NCHW tensor in [0, 1]. No OpenCV."""
    rgb = image.convert("RGB")
    original_width, original_height = rgb.size
    meta = letterbox_meta(original_width, original_height)
    canvas_width = int(meta["canvas_width"])
    canvas_height = int(meta["canvas_height"])

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        (LETTERBOX_COLOR, LETTERBOX_COLOR, LETTERBOX_COLOR),
    )
    resized = rgb.resize(
        (int(meta["resized_width"]), int(meta["resized_height"])),
        Image.Resampling.BILINEAR,
    )
    canvas.paste(resized, (int(meta["pad_x"]), int(meta["pad_y"])))

    array = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC RGB
    tensor = array.transpose(2, 0, 1)[None, ...]  # NCHW
    return np.ascontiguousarray(tensor), meta


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    inter_x1 = max(a["x1"], b["x1"])
    inter_y1 = max(a["y1"], b["y1"])
    inter_x2 = min(a["x2"], b["x2"])
    inter_y2 = min(a["y2"], b["y2"])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(
    candidates: list[dict[str, float]],
    iou_threshold: float = NMS_IOU_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
) -> list[dict[str, float]]:
    ordered = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
    kept: list[dict[str, float]] = []
    for candidate in ordered:
        if len(kept) >= max_detections:
            break
        if any(_iou(candidate, accepted) > iou_threshold for accepted in kept):
            continue
        kept.append(candidate)
    return kept


def decode_and_postprocess(
    output: np.ndarray,
    letterbox: dict[str, float | int],
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> list[dict[str, float]]:
    """Decode YOLO [1, 5, anchors] → original-image xyxy detections."""
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 5:
        raise ValueError(
            f"Unexpected ONNX output shape {tuple(output.shape)}; "
            "expected [1, 5, anchors]."
        )

    anchor_count = int(output.shape[2])
    data = output.reshape(5, anchor_count)
    candidates: list[dict[str, float]] = []

    for index in range(anchor_count):
        confidence = float(data[4, index])
        if confidence < conf_threshold:
            continue
        cx = float(data[0, index])
        cy = float(data[1, index])
        width = float(data[2, index])
        height = float(data[3, index])
        candidates.append(
            {
                "confidence": confidence,
                "x1": cx - width / 2.0,
                "y1": cy - height / 2.0,
                "x2": cx + width / 2.0,
                "y2": cy + height / 2.0,
            }
        )

    kept = nms(candidates, iou_threshold=iou_threshold)
    ratio = float(letterbox["ratio"])
    pad_x = float(letterbox["pad_x"])
    pad_y = float(letterbox["pad_y"])
    original_width = float(letterbox["original_width"])
    original_height = float(letterbox["original_height"])

    detections: list[dict[str, float]] = []
    for box in kept:
        x1 = min(max((box["x1"] - pad_x) / ratio, 0.0), original_width)
        y1 = min(max((box["y1"] - pad_y) / ratio, 0.0), original_height)
        x2 = min(max((box["x2"] - pad_x) / ratio, 0.0), original_width)
        y2 = min(max((box["y2"] - pad_y) / ratio, 0.0), original_height)
        detections.append(
            {
                "confidence": round(box["confidence"], 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        )
    return detections


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "runtime": RUNTIME_NAME,
        "provider": EXECUTION_PROVIDER,
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
    tensor, letterbox = prepare_model_input(image)

    start = time.perf_counter()
    outputs = session.run([OUTPUT_NAME], {INPUT_NAME: tensor})
    inference_ms = (time.perf_counter() - start) * 1000

    detections = decode_and_postprocess(outputs[0], letterbox)

    return {
        "people": len(detections),
        "inference_ms": round(inference_ms, 2),
        "image": {
            "width": width,
            "height": height,
        },
        "detections": detections,
        "model_imgsz": IMAGE_SIZE,
    }
