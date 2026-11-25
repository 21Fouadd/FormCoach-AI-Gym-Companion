"""Real-time Facial Recognition using OpenCV YuNet + SFace (ONNX, zero dlib/C++ build issues).

Includes an ActiveUserTracker that locks the recognized registered user's identity
onto their tracked person bounding box so the HUD card stays solid even when the
user turns sideways or looks down mid-exercise.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "face"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

YUNET_URLS = (
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
SFACE_URLS = (
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
)

YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Official OpenCV SFace cosine similarity threshold (higher = stricter match)
COSINE_MATCH_THRESHOLD = 0.363


def _ensure_model(urls: tuple[str, ...], path: Path, min_bytes: int = 100_000) -> Path:
    if path.exists() and path.stat().st_size >= min_bytes:
        return path

    last_exc: Exception | None = None
    for url in urls:
        tmp = path.with_suffix(".part")
        try:
            with requests.get(url, stream=True, timeout=60, allow_redirects=True) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size < min_bytes:
                raise RuntimeError(
                    f"Downloaded file too small ({tmp.stat().st_size} bytes < {min_bytes} bytes) from {url}"
                )
            tmp.replace(path)
            return path
        except Exception as exc:
            last_exc = exc
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {path.name} from all mirrors: {last_exc}")


class FaceRecognitionEngine:
    def __init__(self) -> None:
        self._detector = None
        self._recognizer = None
        self._ready = False
        self._load_error: str | None = None

    def ensure_loaded(self) -> bool:
        if self._ready:
            return True
        try:
            yunet = _ensure_model(YUNET_URLS, YUNET_PATH, min_bytes=150_000)
            sface = _ensure_model(SFACE_URLS, SFACE_PATH, min_bytes=10_000_000)
            self._detector = cv2.FaceDetectorYN.create(
                str(yunet), "", (320, 320), score_threshold=0.75, nms_threshold=0.3, top_k=5
            )
            self._recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
            self._ready = True
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    @property
    def status(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "engine": "OpenCV YuNet + SFace (128-D cosine)",
            "error": self._load_error,
        }

    def detect_and_embed(self, bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        """Detects the largest face in a BGR image and returns (128-D embedding, (x, y, w, h))."""
        if bgr is None or bgr.size == 0:
            return None
        if not self.ensure_loaded():
            return None

        h, w = bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None

        # Choose the largest detected face (active person closest to camera)
        best = max(faces, key=lambda f: float(f[2] * f[3]))
        aligned = self._recognizer.alignCrop(bgr, best)
        feat = self._recognizer.feature(aligned).flatten().astype(np.float32)
        norm = float(np.linalg.norm(feat))
        if norm > 1e-8:
            feat = feat / norm

        x, y, fw, fh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        return feat, (x, y, fw, fh)

    def match_user(
        self,
        bgr: np.ndarray,
        registered_users: list[dict[str, Any]],
        threshold: float = COSINE_MATCH_THRESHOLD,
    ) -> dict[str, Any] | None:
        """Matches the largest face in frame against registered users in SQLite."""
        if not registered_users:
            return None
        detected = self.detect_and_embed(bgr)
        if detected is None:
            return None
        emb, face_box = detected

        best_user = None
        best_score = -1.0
        for u in registered_users:
            ref = np.asarray(u["face_embedding"], dtype=np.float32)
            # Support both 1D single-photo (128,) and 2D multi-angle Face ID templates (K, 128)
            if ref.ndim == 1:
                templates = [ref]
            elif ref.ndim == 2:
                templates = list(ref)
            else:
                continue
            for tpl in templates:
                ref_norm = float(np.linalg.norm(tpl))
                if ref_norm <= 1e-8 or tpl.shape != emb.shape:
                    continue
                score = float(np.dot(emb, tpl / ref_norm))
                if score > best_score:
                    best_score = score
                    best_user = u

        if best_user is not None and best_score >= threshold:
            return {
                "user": {k: v for k, v in best_user.items() if k != "face_embedding"},
                "similarity": round(best_score, 3),
                "face_box": face_box,
            }
        return None


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


class ActiveUserTracker:
    """Locks recognized registered user onto the active person's body box across sideways turns."""

    def __init__(self, lock_ttl_seconds: float = 20.0, check_every_n_frames: int = 8) -> None:
        self.lock_ttl_seconds = lock_ttl_seconds
        self.check_every_n_frames = check_every_n_frames
        self.locked_user: dict[str, Any] | None = None
        self.locked_similarity: float = 0.0
        self.last_person_box: tuple[int, int, int, int] | None = None
        self.last_seen_ts: float = 0.0
        self._frame_count = 0

    def reset(self) -> None:
        self.locked_user = None
        self.locked_similarity = 0.0
        self.last_person_box = None
        self.last_seen_ts = 0.0
        self._frame_count = 0

    def update(
        self,
        bgr: np.ndarray,
        person_box: tuple[int, int, int, int] | None,
        face_engine: FaceRecognitionEngine,
        registered_users: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, float]:
        now = time.monotonic()
        self._frame_count += 1

        # If person disappeared for too long, release lock
        if person_box is None:
            if self.locked_user and (now - self.last_seen_ts) > self.lock_ttl_seconds:
                self.locked_user = None
                self.locked_similarity = 0.0
            return self.locked_user, self.locked_similarity

        # If the active body moved completely to a different region, allow quick re-ID
        jumped = (
            self.last_person_box is not None
            and _box_iou(self.last_person_box, person_box) < 0.15
        )
        self.last_person_box = person_box
        self.last_seen_ts = now

        should_probe = (
            self.locked_user is None
            or jumped
            or (self._frame_count % self.check_every_n_frames == 0)
        )
        if should_probe and registered_users:
            # Crop around the upper half of the person bounding box for faster, cleaner face ID
            h, w = bgr.shape[:2]
            x1, y1, x2, y2 = person_box
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y1 + int((y2 - y1) * 0.65) + pad_y)
            crop = bgr[cy1:cy2, cx1:cx2]
            match = face_engine.match_user(crop if crop.size > 0 else bgr, registered_users)
            if match is not None:
                self.locked_user = match["user"]
                self.locked_similarity = match["similarity"]

        return self.locked_user, self.locked_similarity
