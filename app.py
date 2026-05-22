from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD YOUR CUSTOM YOLOv8 MODEL
# =========================

MODEL_PATH = "snake_model.pt"  # Place snake_model.pt in same folder as backend.py

print(f"Loading custom YOLOv8 model from {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}. Please place snake_model.pt next to backend.py")

model = YOLO(MODEL_PATH)

CONFIDENCE_THRESHOLD = 0.40

print("✅ YOLOv8 snake model loaded successfully!")

# =========================
# HELPER: Run YOLO detection on PIL image
# =========================

def run_detection(pil_image: Image.Image) -> dict:
    pil_image = pil_image.convert("RGB")

    results = model(pil_image, conf=CONFIDENCE_THRESHOLD, verbose=False)

    detections = []
    snake_detected = False
    best_confidence = 0.0
    best_label = "no snake"

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": round(x1), "y1": round(y1),
                    "x2": round(x2), "y2": round(y2)
                }
            })

            if conf > best_confidence:
                best_confidence = conf
                best_label = label
                snake_detected = True

    return {
        "snake_detected": snake_detected,
        "prediction": best_label,
        "confidence": round(best_confidence, 4),
        "detections": detections,
        "count": len(detections),
        "message": (
            f"⚠️ SNAKE DETECTED: {best_label} ({best_confidence*100:.1f}% confidence) — {len(detections)} detection(s)"
            if snake_detected
            else "✅ No snake detected"
        )
    }

# =========================
# ROOT
# =========================

@app.get("/")
def root():
    return {"message": "Snake Detection Backend Running — YOLOv8 Custom Model"}

# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "classes": model.names,
        "ready": True
    }

# =========================
# MODEL INFO
# =========================

@app.get("/model/info")
def model_info():
    return {
        "model_name": "YOLOv8 Custom Snake Detector",
        "model_file": MODEL_PATH,
        "version": "1.0",
        "status": "active",
        "classes": model.names,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "gpu": torch.cuda.is_available()
    }

# =========================
# IMAGE DETECTION
# =========================

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    try:
        result = run_detection(pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    return {
        "success": True,
        "filename": file.filename,
        **result
    }

# =========================
# VIDEO DETECTION
# =========================

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video")

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        sample_count = min(10, max(1, total_frames))
        frame_indices = [int(i * total_frames / sample_count) for i in range(sample_count)]

        snake_frames = 0
        best_confidence = 0.0
        best_label = "no snake"
        frame_results = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            detection = run_detection(pil_frame)
            frame_results.append({
                "frame": idx,
                "timestamp_sec": round(idx / fps, 2),
                **detection
            })

            if detection["snake_detected"]:
                snake_frames += 1
                if detection["confidence"] > best_confidence:
                    best_confidence = detection["confidence"]
                    best_label = detection["prediction"]

        cap.release()
        os.unlink(tmp_path)

        snake_detected = snake_frames > 0

        return {
            "success": True,
            "filename": file.filename,
            "snake_detected": snake_detected,
            "prediction": best_label if snake_detected else "no snake",
            "confidence": round(best_confidence, 4),
            "frames_analyzed": len(frame_results),
            "snake_frames": snake_frames,
            "message": (
                f"⚠️ SNAKE DETECTED in {snake_frames}/{len(frame_results)} frames — {best_label} ({best_confidence*100:.1f}%)"
                if snake_detected
                else "✅ No snake detected in video"
            ),
            "frame_details": frame_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video detection failed: {str(e)}")
