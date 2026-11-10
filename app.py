import os, tempfile, shutil, subprocess, json, time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

APP_NAME = "Comper AI Analysis API"
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))
SAMPLE_SECONDS = float(os.getenv("SAMPLE_SECONDS", "2.0"))
CLIP_BEFORE = float(os.getenv("CLIP_BEFORE", "4.0"))
CLIP_AFTER = float(os.getenv("CLIP_AFTER", "6.0"))

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: restrict to your hosted HTML origin.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_NAME)
    return _model

@app.get("/")
def root():
    return {"name": APP_NAME, "status": "online", "endpoint": "/analyze"}

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    player_photo: Optional[UploadFile] = File(None),
    player_name: str = Form(""),
    player_number: str = Form(""),
    task: str = Form("full_match_player_reidentification_and_event_clipping"),
    expected_duration_seconds: str = Form("0"),
):
    suffix = Path(video.filename or "match.mp4").suffix or ".mp4"
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / ("match" + suffix)
        total = 0
        with video_path.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"Video exceeds {MAX_UPLOAD_MB} MB limit.")
                out.write(chunk)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(400, "Video could not be opened. Use MP4 H.264 if possible.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps > 0 else 0
        if duration <= 0:
            cap.release()
            raise HTTPException(400, "Could not read video duration.")

        # Detect people over the complete match at a configurable sampling interval.
        # IMPORTANT: YOLO detects people, not a named player's identity or ball possession.
        # This endpoint intentionally returns no guessed player clips.
        model = get_model()
        sample_step = max(1, int(fps * SAMPLE_SECONDS))
        sampled = 0
        frames_with_people = 0
        t = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if frame_idx % sample_step == 0:
                result = model.predict(frame, classes=[0], conf=0.30, verbose=False, imgsz=640)
                sampled += 1
                if result and len(result[0].boxes) > 0:
                    frames_with_people += 1
                t = frame_idx / fps
            # Yield to the event loop periodically on long videos.
            if frame_idx % max(sample_step * 30, 1) == 0:
                pass
        cap.release()

        return JSONResponse({
            "status": "person_detection_completed",
            "player_name": player_name,
            "player_number": player_number,
            "duration_seconds": round(duration, 2),
            "sampled_frames": sampled,
            "frames_with_people": frames_with_people,
            "clips": [],
            "message": (
                "Person detection ran successfully, but this baseline model cannot identify "
                "the named player from a photo or determine ball possession/goals. "
                "No clips were fabricated. Add a trained football player re-identification "
                "and event-detection model before returning clip timestamps."
            ),
            "photo_received": player_photo is not None,
        })
