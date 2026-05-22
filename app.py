from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import cv2
import numpy as np
import tempfile
import os
import time
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
# LOAD MODEL
# =========================

MODEL_PATH = "snake_model.pt"

print(f"Loading YOLOv8 model from {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found: {MODEL_PATH}"
    )

model = YOLO(MODEL_PATH)

CONFIDENCE_THRESHOLD = 0.40

print("✅ Snake model loaded successfully!")

# =========================
# ROOT
# =========================

@app.get("/")
def root():
    return {
        "message": "YOLOv8 Snake Detection API Running"
    }

# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": True,
        "classes": list(model.names.values())
    }

# =========================
# MODEL INFO
# =========================

@app.get("/model/info")
def model_info():
    return {
        "name": "YOLOv8 Snake Detector",
        "version": "1.0",
        "classes": list(model.names.values()),
        "inputSize": 640,
        "gpu": torch.cuda.is_available()
    }

# =========================
# DETECTION FUNCTION
# =========================

def run_detection(pil_image: Image.Image):

    pil_image = pil_image.convert("RGB")

    results = model(
        pil_image,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    detections = []

    snake_detected = False
    best_confidence = 0.0
    best_label = "No Snake"

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

                # FRONTEND COMPATIBLE BOX FORMAT
                "x": round(x1),
                "y": round(y1),
                "width": round(x2 - x1),
                "height": round(y2 - y1)
            })

            if conf > best_confidence:
                best_confidence = conf
                best_label = label
                snake_detected = True

    return {
        "detected": snake_detected,
        "confidence": round(best_confidence, 4),
        "label": best_label,
        "boxes": detections,
        "timestamp": int(time.time())
    }

# =========================
# IMAGE DETECTION
# =========================

@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...)
):

    if (
        not file.content_type or
        not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image"
        )

    try:

        contents = await file.read()

        pil_image = Image.open(
            io.BytesIO(contents)
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {str(e)}"
        )

    try:

        result = run_detection(pil_image)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

    return result

# =========================
# VIDEO DETECTION
# =========================

@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...)
):

    if (
        not file.content_type or
        not file.content_type.startswith("video/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a video"
        )

    try:

        contents = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:

            tmp.write(contents)

            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        sample_count = min(
            10,
            max(1, total_frames)
        )

        frame_indices = [
            int(i * total_frames / sample_count)
            for i in range(sample_count)
        ]

        detections = []

        overall_detected = False

        best_confidence = 0.0

        best_label = "No Snake"

        for idx in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                idx
            )

            ret, frame = cap.read()

            if not ret:
                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            pil_frame = Image.fromarray(
                frame_rgb
            )

            detection = run_detection(
                pil_frame
            )

            detections.append({
                "frame": idx,
                "time": round(idx / fps, 2),
                **detection
            })

            if detection["detected"]:

                overall_detected = True

                if detection["confidence"] > best_confidence:

                    best_confidence = detection["confidence"]

                    best_label = detection["label"]

        cap.release()

        os.unlink(tmp_path)

        return {
            "detected": overall_detected,
            "confidence": best_confidence,
            "label": best_label,
            "framesAnalyzed": len(detections),
            "results": detections,
            "timestamp": int(time.time())
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Video detection failed: {str(e)}"
        )

# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
