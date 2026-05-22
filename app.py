from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

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
# ROOT
# =========================


@app.get("/")
def root():
    return {"message": "Snake Detection Backend Running"}

# =========================
# HEALTH CHECK
# =========================


@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# MODEL INFO
# =========================


@app.get("/model/info")
def model_info():
    return {
        "model_name": "Snake Detection AI",
        "version": "1.0",
        "status": "active",
        "confidence_threshold": 0.40
    }

# =========================
# IMAGE DETECTION
# =========================


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "filename": file.filename,
        "prediction": "snake",
        "confidence": 0.91
    }

# =========================
# VIDEO DETECTION
# =========================


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "filename": file.filename,
        "prediction": "snake detected in video",
        "confidence": 0.88
    }
