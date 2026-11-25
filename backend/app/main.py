"""FastAPI Backend for AI Gym Trainer (Flutter REST + Real-Time WebSocket + Built-in Demo UI)."""
from __future__ import annotations

import base64
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import database as db
from .face_id import ActiveUserTracker, FaceRecognitionEngine
from .profile_ai import SmartFitProfilePredictor
from .pose_engine import (
    SUPPORTED_EXERCISES,
    LiveRepTracker,
    ModelRegistry,
    analyze_postset_video,
    create_pose_landmarker,
    extract_pose_from_result,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AI Gym Trainer Backend",
    version="1.0.0",
    description="Real-time & post-set exercise form analysis, rep counting, voice cues, and facial recognition HUD.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()
face_engine = FaceRecognitionEngine()
model_registry = ModelRegistry()
profile_predictor = SmartFitProfilePredictor()

app.mount("/static/photos", StaticFiles(directory=str(db.PHOTOS_DIR)), name="photos")
app.mount("/static/outputs", StaticFiles(directory=str(db.OUTPUTS_DIR)), name="outputs")


def _decode_image_bytes(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


@app.get("/")
def serve_demo_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "AI Gym Trainer API is running. Open /docs for Swagger UI."}


@app.get("/api/health")
def health():
    face_engine.ensure_loaded()
    return {
        "status": "ok",
        "supported_exercises": SUPPORTED_EXERCISES,
        "loaded_models": model_registry.summary(),
        "profile_ai_model": profile_predictor.mode,
        "face_recognition": face_engine.status,
        "registered_users": len(db.list_users()),
    }


@app.post("/api/users/register")
async def register_user(
    name: str = Form(...),
    age: int = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    gender: str = Form("Other"),
    fitness_level: str = Form("Auto"),
    photo: UploadFile = File(...),
    photo_left: UploadFile | None = File(None),
    photo_right: UploadFile | None = File(None),
):
    raw_front = await photo.read()
    bgr_front = _decode_image_bytes(raw_front)

    templates: list[list[float]] = []
    enrolled_angles: list[str] = []

    det_front = face_engine.detect_and_embed(bgr_front)
    if det_front is not None:
        templates.append(det_front[0].tolist())
        enrolled_angles.append("front")

    for angle_name, extra_file in (("left", photo_left), ("right", photo_right)):
        if extra_file is not None:
            raw_extra = await extra_file.read()
            if raw_extra:
                bgr_extra = _decode_image_bytes(raw_extra)
                det_extra = face_engine.detect_and_embed(bgr_extra)
                if det_extra is not None:
                    templates.append(det_extra[0].tolist())
                    enrolled_angles.append(angle_name)

    if not templates:
        raise HTTPException(
            status_code=422,
            detail="No clear face detected in the captured angle(s). Please face the camera in good lighting.",
        )

    photo_name = f"user_{uuid.uuid4().hex[:10]}.jpg"
    photo_path = db.PHOTOS_DIR / photo_name
    cv2.imwrite(str(photo_path), bgr_front)

    prediction = profile_predictor.predict(
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        gender=gender,
        requested_level=fitness_level,
    )

    stored_embedding: list[float] | list[list[float]] = templates[0] if len(templates) == 1 else templates
    user = db.create_user(
        name=name,
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        fitness_level=prediction["fitness_level"],
        face_embedding=stored_embedding,
        photo_path=str(photo_path),
    )
    return {
        "status": "registered",
        "user": user,
        "angles_enrolled": enrolled_angles,
        "fitness_prediction": prediction,
    }


@app.get("/api/users")
def get_users():
    return {"users": db.list_users(include_embedding=False)}


@app.delete("/api/users/{user_id}")
def remove_user(user_id: int):
    if not db.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    return {"status": "deleted", "user_id": user_id}


@app.get("/api/users/{user_id}/history")
def get_user_history(user_id: int):
    user = db.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"user": user, "sessions": db.list_user_sessions(user_id)}


@app.post("/api/identify")
async def identify_face(photo: UploadFile = File(...)):
    raw = await photo.read()
    bgr = _decode_image_bytes(raw)
    users = db.list_users(include_embedding=True)
    match = face_engine.match_user(bgr, users)
    if match is None:
        return {"recognized": False, "user": None, "similarity": 0.0}
    return {"recognized": True, "user": match["user"], "similarity": match["similarity"]}


@app.post("/api/analyze/video")
async def analyze_video(
    exercise: str = Form("squat"),
    user_id: int | None = Form(None),
    video: UploadFile = File(...),
):
    ex = exercise.lower().strip()
    if ex not in SUPPORTED_EXERCISES:
        raise HTTPException(status_code=400, detail=f"Unsupported exercise. Choose from {SUPPORTED_EXERCISES}.")

    run_id = uuid.uuid4().hex[:10]
    in_path = db.OUTPUTS_DIR / f"upload_{run_id}.mp4"
    out_name = f"annotated_{ex}_{run_id}.mp4"
    out_path = db.OUTPUTS_DIR / out_name

    with in_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        users = db.list_users(include_embedding=True)
        override_user = db.get_user(user_id) if user_id is not None else None
        model = model_registry.get(ex)
        result = analyze_postset_video(
            video_path=in_path,
            output_video_path=out_path,
            exercise=ex,
            model=model,
            face_engine=face_engine,
            registered_users=users,
            override_user=override_user,
            model_registry=model_registry,
        )
    finally:
        in_path.unlink(missing_ok=True)

    video_url = f"/static/outputs/{out_name}"
    rec_user = result["recognized_user"]
    saved = db.save_workout_session(
        user_id=rec_user["id"] if rec_user else None,
        exercise=result["exercise"],
        mode="postset_video",
        total_reps=result["total_reps"],
        correct_reps=result["correct_reps"],
        incorrect_reps=result["incorrect_reps"],
        form_score=result["form_score"],
        cues=result["summary_cues"],
        video_url=video_url,
    )
    result["video_url"] = video_url
    result["session"] = saved
    return result


@app.websocket("/ws/live/{exercise}")
async def live_workout_ws(websocket: WebSocket, exercise: str):
    """Real-time frame WebSocket for Flutter camera stream & Web Demo.

    Client sends JSON: {"frame_b64": "<jpeg_base64>"} (or {"action": "finish"} to save session).
    Server responds with person bounding box, recognized user HUD, skeleton landmarks, rep count, and voice cue.
    """
    await websocket.accept()
    ex = exercise.lower().strip()
    if ex not in SUPPORTED_EXERCISES:
        ex = "squat"

    model = model_registry.get(ex)
    rep_tracker = LiveRepTracker(ex)
    user_tracker = ActiveUserTracker(lock_ttl_seconds=25.0, check_every_n_frames=6)
    registered_users = db.list_users(include_embedding=True)

    with create_pose_landmarker() as detector:
        try:
            while True:
                msg = await websocket.receive_text()
                payload: dict[str, Any] = json.loads(msg)

                if payload.get("action") == "finish":
                    rec_user = user_tracker.locked_user
                    form_score = (
                        (100.0 * rep_tracker.correct_reps / rep_tracker.rep_count)
                        if rep_tracker.rep_count > 0 else 0.0
                    )
                    saved = db.save_workout_session(
                        user_id=rec_user["id"] if rec_user else None,
                        exercise=rep_tracker.exercise if rep_tracker.exercise != "auto" else ex,
                        mode="live_camera",
                        total_reps=rep_tracker.rep_count,
                        correct_reps=rep_tracker.correct_reps,
                        incorrect_reps=rep_tracker.incorrect_reps,
                        form_score=form_score,
                        cues=list(dict.fromkeys(rep_tracker.all_cues))[:5],
                    )
                    await websocket.send_json({"type": "session_saved", "session": saved})
                    continue

                if payload.get("action") == "refresh_users":
                    registered_users = db.list_users(include_embedding=True)
                    user_tracker.reset()
                    await websocket.send_json({"type": "users_refreshed", "count": len(registered_users)})
                    continue

                b64 = payload.get("frame_b64", "")
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                if not b64:
                    continue

                raw = base64.b64decode(b64)
                bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue

                h, w = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                p_info = extract_pose_from_result(res, w, h)

                person_box_norm = None
                landmarks_norm = {}
                rep_state = {
                    "rep_count": rep_tracker.rep_count,
                    "correct_reps": rep_tracker.correct_reps,
                    "incorrect_reps": rep_tracker.incorrect_reps,
                    "stage": rep_tracker.stage,
                    "primary_angle": None,
                    "verdict": rep_tracker.last_verdict,
                    "cues": rep_tracker.last_cues,
                    "spoken_cue": None,
                    "rep_completed": False,
                    "form_score": (
                        round(100.0 * rep_tracker.correct_reps / rep_tracker.rep_count, 1)
                        if rep_tracker.rep_count > 0 else 100.0
                    ),
                }

                if p_info is not None:
                    x1, y1, x2, y2 = p_info["person_box"]
                    person_box_norm = [
                        round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)
                    ]
                    landmarks_norm = p_info["landmarks_norm"]
                    rec_user, sim = user_tracker.update(bgr, p_info["person_box"], face_engine, registered_users)
                    if p_info["angles"] is not None:
                        if rep_tracker.exercise == "auto" and model.exercise == "auto":
                            rep_state = rep_tracker.update(
                                p_info["angles"], p_info["norm_joints"], model, pose_meta=p_info
                            )
                            if rep_tracker.exercise != "auto":
                                model = model_registry.get(rep_tracker.exercise)
                        else:
                            if rep_tracker.exercise != model.exercise:
                                model = model_registry.get(rep_tracker.exercise)
                            rep_state = rep_tracker.update(
                                p_info["angles"], p_info["norm_joints"], model, pose_meta=p_info
                            )
                else:
                    rec_user, sim = user_tracker.update(bgr, None, face_engine, registered_users)

                await websocket.send_json({
                    "type": "frame_result",
                    "exercise": rep_tracker.exercise,
                    "model_used": model.mode,
                    "person_box": person_box_norm,
                    "landmarks": landmarks_norm,
                    "user": rec_user,
                    "recognition_confidence": sim,
                    **rep_state,
                })
        except WebSocketDisconnect:
            pass
