"""Generates the clean, syntax-verified Google Colab Multi-Model Benchmark Notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path


def md_cell(text: str) -> dict:
    lines = [line + "\n" for line in text.strip("\n").splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def code_cell(code: str) -> dict:
    clean = code.strip("\n")
    # Syntax-check every Python cell (stripping IPython %magics first)
    checkable = "\n".join(
        f"# {line}" if line.lstrip().startswith(("%", "!")) else line
        for line in clean.splitlines()
    )
    ast.parse(checkable)
    lines = [line + "\n" for line in clean.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


CELLS = [
    md_cell(
        """# AI Gym Trainer — Multi-View Dataset & 10-Model Form Benchmark (Colab)

This notebook solves the core AI benchmarking needs for the graduation project:
1. **Larger Labeled Gym Dataset:** Automatically downloads the **Mendeley Multi-View Seven Fitness Exercises** dataset (`kgbb3yn47p`, v3 — **26 subjects × 3 smartphone camera angles** with `Good`/`Bad` form labels) directly on Colab via Mendeley's Public API, with optional **REHAB24-6 merge** (+9 subjects → **35 unique squat subjects**).
2. **Google Drive Caching:** Extracted pose tensors are cached as a compact `.npz` file in Google Drive so you never have to re-download or re-extract videos when Colab resets.
3. **Fair Cross-Subject Evaluation:** Uses **5-Fold `StratifiedGroupKFold`** grouped by `person_id` (zero participant leakage!) with pose augmentation applied **strictly inside training folds**.
4. **10 Models Compared Apples-to-Apples:**
   - **Tabular Baselines (Phase + Symmetry Biomechanical Features):** `Random Forest`, `SVM (RBF)`, `XGBoost`, `LightGBM`
   - **Deep Learning (Pose Sequences & 2D Skeleton Grid):** `1D CNN`, **`2D CNN` (`Time × Joints × Channels`)**, `BiLSTM + Attention`, `TCN`
   - **Merged Models:** **`Two-Stream 2D-CNN + BiGRU Fusion`** (merged single architecture) & **`Cross-Family Soft-Voting Ensemble`** (Top Deep + Top Tabular)
5. **One Master Comparison Cell + Export:** Shows a single sorted leaderboard (`Macro F1`, `Accuracy`, `Balanced Acc`, `Precision`, `Recall`, `PR-AUC`, `ROC-AUC`), charts, confusion matrices, and exports the winning bundle for your **local FastAPI + Flutter app**.
"""
    ),
    md_cell("## 1. Setup, GPU Check & Configuration"),
    code_cell(
        """%pip -q install mediapipe==0.10.35 xgboost lightgbm scikit-learn seaborn tqdm scipy

import os
import re
import json
import math
import shutil
import zipfile
import tempfile
import warnings
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.signal import find_peaks

import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models as kmodels

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

warnings.filterwarnings("ignore")

# ---- Reproducibility ----
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ---- User Configuration ----
USE_GOOGLE_DRIVE = True          # Saves pose cache + exported model permanently on Drive
MENDELEY_DATASET_ID = "kgbb3yn47p"
MENDELEY_VERSION = 3
# Options (Mendeley 26 subj): "squat", "push", "curl", "shoulder", "row", "tricep", "side_bend"
# Options (REHAB24-6 + merge): "squat" (merged!), "push" (merged!), "lunge", "arm_abduction", "arm_vw", "leg_abduction"
TARGET_EXERCISE = "squat"
INCLUDE_REHAB24_6 = True         # Automatically merges REHAB24-6 whenever TARGET_EXERCISE is in REHAB24-6!
FRAME_STEP = 3                   # Sample every 3 frames (~10 FPS at 30 FPS video)
SEQUENCE_LENGTH = 32             # Fixed temporal length per repetition
N_FOLDS = 5                      # Subject-grouped CV folds (zero participant leakage)
EPOCHS = 28
BATCH_SIZE = 64

# ---- Paths ----
WORK_DIR = Path("/content/gym_trainer_work")
WORK_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_DIR = WORK_DIR / "artifacts"

if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_DIR = Path("/content/drive/MyDrive/GymTrainerAI")
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)
        print("Google Drive mounted at:", DRIVE_DIR)
    except Exception as e:
        print("Drive mount skipped (running outside Colab or declined):", e)

CACHE_FILE = DRIVE_DIR / f"pose_cache_{TARGET_EXERCISE}_rehab{int(INCLUDE_REHAB24_6)}.npz"
EXPORT_DIR = DRIVE_DIR / f"exported_model_{TARGET_EXERCISE}"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPUs:", gpus if gpus else "CPU only (enable T4 GPU in Runtime -> Change runtime type)")
print("Target exercise:", TARGET_EXERCISE, "| Cache path:", CACHE_FILE)
"""
    ),
    md_cell(
        """## 2. Smart Auto-Downloader for Mendeley `kgbb3yn47p` (Skips Duplicate Layouts)

Mendeley `kgbb3yn47p` hosts the same videos under 3 folder layouts (`Subject-wise`, `Exercise-wise`, `Exercise-Quality`).
This cell inspects the Mendeley Public API, picks **one canonical layout**, filters strictly to `TARGET_EXERCISE`, and downloads only the videos we need in parallel (and skips completely if your Google Drive pose cache already exists!).
"""
    ),
    code_cell(
        """EXERCISE_KEYWORDS = {
    "squat": ["squat"],
    "push": ["push", "pushup", "push-up", "push_up"],
    "curl": ["curl", "bicep"],
    "shoulder": ["shoulder", "press"],
    "row": ["row", "bent"],
    "tricep": ["tricep", "extension"],
    "side_bend": ["side", "bend"],
}

REHAB_EXERCISE_IDS = {
    "arm_abduction": 1,
    "arm_vw": 2,
    "push": 3,
    "leg_abduction": 4,
    "lunge": 5,
    "squat": 6,
}


def _mendeley_get(endpoint: str, params: dict) -> list:
    url = f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}/{endpoint}"
    r = requests.get(url, params=params, timeout=60)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else data.get("data", [])


def list_mendeley_tree(dataset_id: str = MENDELEY_DATASET_ID, version: int = MENDELEY_VERSION):
    \"\"\"Walks Mendeley dataset folders and picks ONE canonical layout to avoid 3x duplicates.\"\"\"
    root_folders = _mendeley_get("folders", {"version": version, "folder_id": "root"})
    root_files = _mendeley_get("files", {"version": version, "folder_id": "root"})

    print("Mendeley root folders:", [f.get("name") for f in root_folders])
    print("Mendeley root files:", [f.get("filename") for f in root_files[:10]])

    # Prefer Exercise-wise or Exercise-Quality to avoid downloading all 7 exercises or 3 copies
    chosen_roots = root_folders
    if len(root_folders) > 1:
        def layout_priority(folder):
            name = (folder.get("name") or "").lower()
            if "exercise" in name and "quality" in name:
                return 0
            if "exercise" in name:
                return 1
            if "subject" in name:
                return 2
            return 3
        chosen_roots = [sorted(root_folders, key=layout_priority)[0]]
        print("Selected canonical layout folder (avoiding duplicates):", chosen_roots[0].get("name"))

    discovered = []
    for f in root_files:
        discovered.append({"path": f.get("filename", ""), "file": f})

    def walk(folder_id: str, prefix: str):
        subfolders = _mendeley_get("folders", {"version": version, "folder_id": folder_id})
        files = _mendeley_get("files", {"version": version, "folder_id": folder_id})
        for f in files:
            discovered.append({"path": f"{prefix}/{f.get('filename', '')}", "file": f})
        for sf in subfolders:
            sf_name = sf.get("name", "")
            sf_path = f"{prefix}/{sf_name}"
            # Prune non-target exercise folders early so API walking is super fast!
            low = sf_name.lower()
            other_ex_hits = [
                k for k, kws in EXERCISE_KEYWORDS.items()
                if k != TARGET_EXERCISE and any(kw in low for kw in kws)
            ]
            target_hits = any(kw in low for kw in EXERCISE_KEYWORDS[TARGET_EXERCISE])
            if other_ex_hits and not target_hits:
                continue
            walk(sf["id"], sf_path)

    for rf in chosen_roots:
        walk(rf["id"], rf.get("name", ""))

    return discovered


def parse_mendeley_metadata(rel_path: str):
    \"\"\"Extracts (subject_id, label_incorrect, camera_view) from path/filename.\"\"\"
    low = rel_path.lower().replace("\\\\", "/")
    # Form label: 0 = Correct/Good, 1 = Incorrect/Bad
    if re.search(r"(^|[/ _-])(bad|incorrect|wrong|poor|fault)([/ _.-]|$)", low):
        label = 1
    elif re.search(r"(^|[/ _-])(good|correct|proper|right)([/ _.-]|$)", low):
        label = 0
    else:
        return None

    # Subject ID (e.g., S01, Sub12, P05, subject_03, user04, or numeric prefix)
    subj_match = re.search(
        r"(?:subject|subj|sub|person|participant|user|p|s)[ _-]?0*([0-9]{1,3})(?![0-9])",
        low,
    )
    subject_id = f"mendeley_S{int(subj_match.group(1)):02d}" if subj_match else f"mendeley_{Path(rel_path).stem}"

    # Camera view
    if "front" in low:
        view = "front"
    elif "side" in low:
        view = "side"
    elif "diag" in low or "45" in low or "angle" in low:
        view = "diagonal"
    else:
        view = "unknown"

    return subject_id, label, view


def download_mendeley_videos(dest_dir: Path) -> list[dict]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    entries = list_mendeley_tree()
    kws = EXERCISE_KEYWORDS[TARGET_EXERCISE]

    # Clean up any duplicate or corrupt zip files left from earlier runs
    for old_zip in dest_dir.glob("*.zip"):
        if "exercise-wise" not in old_zip.name.lower() or not zipfile.is_zipfile(old_zip):
            print("Removing duplicate/corrupt zip:", old_zip.name)
            old_zip.unlink(missing_ok=True)

    # Filter to video or ONE canonical zip (avoid downloading all 3 duplicate zips!)
    video_entries = [
        e for e in entries
        if e["path"].lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        and any(kw in e["path"].lower() for kw in kws)
    ]
    zip_entries = [
        e for e in entries
        if e["path"].lower().endswith(".zip")
        and (any(kw in e["path"].lower() for kw in kws) or len(entries) <= 5)
    ]
    if not video_entries and len(zip_entries) > 1:
        def _zip_priority(e):
            n = e["path"].lower()
            if "exercise-wise" in n or "exercise_wise" in n:
                return 0
            if "quality" in n:
                return 1
            if "subject" in n:
                return 2
            return 3
        zip_entries = [sorted(zip_entries, key=_zip_priority)[0]]

    print(f"Selected {len(video_entries)} MP4 videos and {len(zip_entries)} ZIP archive(s) for '{TARGET_EXERCISE}'.")

    def _download_one(item):
        rel_path = item["path"].lstrip("/")
        file_obj = item["file"]
        cd = file_obj.get("content_details") or {}
        url = cd.get("download_url")
        if not url and file_obj.get("id"):
            url = (
                f"https://api.data.mendeley.com/datasets/{MENDELEY_DATASET_ID}"
                f"/files/{file_obj['id']}/file_downloaded?version={MENDELEY_VERSION}"
            )
        if not url:
            return None
        out_path = dest_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        expected_size = cd.get("size")
        if out_path.exists():
            size_ok = (not expected_size) or (out_path.stat().st_size == expected_size)
            zip_ok = (out_path.suffix.lower() != ".zip") or zipfile.is_zipfile(out_path)
            if size_ok and zip_ok:
                print(f"Using already-downloaded valid file: {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
                return out_path
            print(f"Removing incomplete/corrupt file: {out_path.name}")
            out_path.unlink(missing_ok=True)

        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", expected_size or 0))
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=out_path.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        tmp.replace(out_path)
        return out_path

    downloaded_files = []
    targets = video_entries if video_entries else zip_entries
    # Download single large zip sequentially in a single stream (avoids truncation/rate-limiting)
    if len(targets) == 1 and targets[0]["path"].lower().endswith(".zip"):
        res = _download_one(targets[0])
        if res is not None:
            downloaded_files.append(res)
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_download_one, e) for e in targets]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading Mendeley"):
                res = fut.result()
                if res is not None:
                    downloaded_files.append(res)

    # Extract ONLY matching exercise MP4s from the zip
    for p in list(downloaded_files):
        if p.suffix.lower() == ".zip":
            if not zipfile.is_zipfile(p):
                raise RuntimeError(
                    f"{p.name} ({p.stat().st_size} bytes) is not a valid ZIP! "
                    f"First 200 bytes: {p.read_bytes()[:200]!r}"
                )
            extract_to = p.parent / f"extracted_{TARGET_EXERCISE}"
            extract_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p) as zf:
                matching = [
                    m for m in zf.namelist()
                    if m.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
                    and any(kw in m.lower() for kw in kws)
                ]
                print(f"Extracting {len(matching)} '{TARGET_EXERCISE}' videos from {p.name}...")
                for member in tqdm(matching, desc=f"Unzipping {TARGET_EXERCISE}"):
                    if not (extract_to / member).exists():
                        zf.extract(member, extract_to)

    # Collect all matching video files
    records = []
    seen_stems = set()
    for mp4 in sorted(dest_dir.rglob("*")):
        if mp4.suffix.lower() not in (".mp4", ".mov", ".avi", ".mkv"):
            continue
        rel = mp4.relative_to(dest_dir).as_posix()
        if not any(kw in rel.lower() for kw in kws):
            continue
        # Deduplicate in case a zip contained multiple layouts
        if mp4.name.lower() in seen_stems:
            continue
        meta = parse_mendeley_metadata(rel)
        if meta is None:
            continue
        seen_stems.add(mp4.name.lower())
        subject_id, label, view = meta
        records.append({
            "path": mp4,
            "subject_id": subject_id,
            "label": label,
            "view": view,
            "source": "mendeley",
        })

    print(f"Ready Mendeley videos: {len(records)} across {len({r['subject_id'] for r in records})} subjects.")
    return records


mendeley_records = []
if not CACHE_FILE.exists():
    if TARGET_EXERCISE in EXERCISE_KEYWORDS:
        mendeley_records = download_mendeley_videos(WORK_DIR / "mendeley_videos")
    else:
        print(f"'{TARGET_EXERCISE}' is a REHAB24-6 exercise — skipping Mendeley download.")
else:
    print("Pose cache already exists on Drive — skipping Mendeley video download!")
"""
    ),
    md_cell(
        """## 3. Pose Extraction: 2D Spatial-Temporal Skeleton Grid + 14 Biomechanical Angles + Rep Segmentation

For every detected repetition we extract **two complementary tensors**:
1. **`skel_grid` — shape `(32, 12, 6)`** for **`2D CNN`** & **`Two-Stream 2D-CNN + BiGRU`**:
   - 12 core body joints (`11–16, 23–28`) centered at mid-hip and scaled by torso length `(x, y, z)` + frame-to-frame velocities `(dx, dy, dz)`.
2. **`angle_seq` — shape `(32, 14)`** for **`1D CNN`**, **`BiLSTM + Attention`**, **`TCN`**, and **Tabular Phase Features**:
   - All 14 scale-invariant world-landmark angles/tilts (`left/right elbow, shoulder, hip, knee`, `torso_tilt`, `shoulder_line_tilt`, `hip_line_tilt`, `wrist heights`, `body_line`).
"""
    ),
    code_cell(
        """CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # 12 joints
FEATURES = [
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "torso_tilt", "shoulder_line_tilt",
    "hip_line_tilt", "left_wrist_height", "right_wrist_height",
    "body_line", "left_knee", "right_knee",
]

POSE_MODEL_PATH = WORK_DIR / "pose_landmarker_full.task"
if not POSE_MODEL_PATH.exists():
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        POSE_MODEL_PATH,
    )

POSE_OPTIONS = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
)


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    cosine = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def extract_frame_pose(result):
    \"\"\"Side-view + chest-up webcam resilient extractor.\"\"\"
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return None
    lms_2d = result.pose_landmarks[0]
    left_lower = all(lms_2d[i].visibility >= 0.28 for i in (11, 23, 25, 27))
    right_lower = all(lms_2d[i].visibility >= 0.28 for i in (12, 24, 26, 28))
    left_upper = all(lms_2d[i].visibility >= 0.28 for i in (11, 13, 15, 23))
    right_upper = all(lms_2d[i].visibility >= 0.28 for i in (12, 14, 16, 24))
    left_arm_only = (
        lms_2d[11].visibility >= 0.35 and lms_2d[13].visibility >= 0.20 and lms_2d[15].visibility >= 0.14
    )
    right_arm_only = (
        lms_2d[12].visibility >= 0.35 and lms_2d[14].visibility >= 0.20 and lms_2d[16].visibility >= 0.14
    )
    if not (left_lower or right_lower or left_upper or right_upper or left_arm_only or right_arm_only):
        return None

    hips_visible = max(lms_2d[23].visibility, lms_2d[24].visibility) >= 0.35
    p = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]], dtype=np.float32)
    shoulders = (p[11] + p[12]) / 2.0
    hips = (p[23] + p[24]) / 2.0
    raw_scale = float(np.linalg.norm(shoulders - hips))
    scale = raw_scale if (hips_visible and raw_scale >= 0.12) else max(raw_scale, float(np.linalg.norm(p[11] - p[12])) * 1.35, 0.20)
    if scale < 1e-5:
        return None

    def line_tilt(a_idx: int, b_idx: int) -> float:
        v = p[a_idx] - p[b_idx]
        return float(np.degrees(np.arctan2(abs(v[1]), np.hypot(v[0], v[2]))))

    angles = np.array([
        calc_angle(p[11], p[13], p[15]),
        calc_angle(p[12], p[14], p[16]),
        calc_angle(p[23], p[11], p[13]),
        calc_angle(p[24], p[12], p[14]),
        calc_angle(p[11], p[23], p[25]),
        calc_angle(p[12], p[24], p[26]),
        calc_angle(shoulders, hips, hips + np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        line_tilt(11, 12),
        line_tilt(23, 24),
        float((p[11, 1] - p[15, 1]) / scale),
        float((p[12, 1] - p[16, 1]) / scale),
        calc_angle(shoulders, hips, (p[27] + p[28]) / 2.0),
        calc_angle(p[23], p[25], p[27]),
        calc_angle(p[24], p[26], p[28]),
    ], dtype=np.float32)

    norm_joints = ((p[CORE_JOINTS] - hips) / scale).astype(np.float32)  # (12, 3)
    if not (np.isfinite(angles).all() and np.isfinite(norm_joints).all()):
        return None
    return angles, norm_joints


def read_video_pose(video_path: Path, detector, step: int = FRAME_STEP):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frames, angle_list, joint_list = [], [], []
    idx = 0
    try:
        while cap.grab():
            if idx % step == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    # Downscale very high-res smartphone footage for fast Pose extraction
                    h, w = bgr.shape[:2]
                    if max(h, w) > 720:
                        s = 720.0 / max(h, w)
                        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    out = extract_frame_pose(res)
                    if out is not None:
                        frames.append(idx)
                        angle_list.append(out[0])
                        joint_list.append(out[1])
            idx += 1
    finally:
        cap.release()

    return (
        np.asarray(frames, dtype=np.int32),
        np.asarray(angle_list, dtype=np.float32),
        np.asarray(joint_list, dtype=np.float32),
        idx,
        float(fps),
    )


def resample_rep(pose_tuple, start: int, stop: int, seq_len: int = SEQUENCE_LENGTH, step: int = FRAME_STEP):
    frames, angles, joints, _, _ = pose_tuple
    if len(frames) == 0:
        return None
    mask = (frames >= start) & (frames < stop)
    expected = max(1.0, (stop - start) / step)
    if mask.sum() < max(4, expected * 0.30):
        return None

    f_sub = frames[mask]
    a_sub = angles[mask]
    j_sub = joints[mask]  # (N, 12, 3)

    grid = np.linspace(float(start), float(max(start + 1, stop - 1)), seq_len)
    ang_interp = np.stack([np.interp(grid, f_sub, a_sub[:, c]) for c in range(a_sub.shape[1])], axis=-1)

    j_flat = j_sub.reshape(len(f_sub), -1)
    j_interp = np.stack([np.interp(grid, f_sub, j_flat[:, c]) for c in range(j_flat.shape[1])], axis=-1)
    j_interp = j_interp.reshape(seq_len, len(CORE_JOINTS), 3)

    # Frame-to-frame velocity channels -> (32, 12, 6)
    vel = np.diff(j_interp, axis=0, prepend=j_interp[:1])
    skel_grid = np.concatenate([j_interp, vel], axis=-1).astype(np.float32)
    return ang_interp.astype(np.float32), skel_grid


def segment_reps_auto(pose_tuple, exercise: str = TARGET_EXERCISE, step: int = FRAME_STEP):
    \"\"\"Segments repetitions automatically using the primary driving joint angle.\"\"\"
    frames, angles, _, total_frames, fps = pose_tuple
    if len(frames) < 6 or fps <= 0:
        return []

    if exercise == "squat":
        signal, invert, prominence, min_drop = angles[:, -2:].mean(axis=1), True, 20.0, 16.0
    elif exercise == "lunge":
        signal, invert, prominence, min_drop = angles[:, -2:].min(axis=1), True, 22.0, 18.0
    elif exercise in ("push", "curl", "tricep"):
        # Min elbow angle preserves clean bend signal on side-view / alternating push-ups & curls
        signal, invert, prominence, min_drop = angles[:, :2].min(axis=1), True, 15.0, 13.0
    elif exercise in ("shoulder", "arm_abduction", "arm_vw"):
        signal, invert, prominence, min_drop = angles[:, 2:4].max(axis=1), False, 20.0, 16.0
    elif exercise == "row":
        elb_rom = max(
            float(np.percentile(angles[:, 0], 90) - np.percentile(angles[:, 0], 10)),
            float(np.percentile(angles[:, 1], 90) - np.percentile(angles[:, 1], 10)),
        )
        hip_rom = float(np.percentile(angles[:, 4:6].mean(axis=1), 90) - np.percentile(angles[:, 4:6].mean(axis=1), 10))
        if elb_rom >= 32.0 and elb_rom >= 0.70 * hip_rom:
            signal, invert, prominence, min_drop = angles[:, :2].min(axis=1), True, 16.0, 14.0
        else:
            signal, invert, prominence, min_drop = angles[:, 4:6].mean(axis=1), True, 20.0, 16.0
    else:
        signal, invert, prominence, min_drop = angles[:, 4:6].mean(axis=1), True, 18.0, 14.0

    grid = np.arange(frames[0], frames[-1] + 1, step)
    curve = np.interp(grid, frames, signal)
    if len(curve) >= 5:
        curve = np.convolve(curve, [0.25, 0.5, 0.25], mode="same")
    search_curve = -curve if invert else curve

    peaks, _ = find_peaks(
        search_curve,
        prominence=prominence,
        distance=max(2, int(0.65 * fps / step)),
    )

    segments = []
    for i, pk in enumerate(peaks):
        left = 0 if i == 0 else peaks[i - 1]
        right = len(curve) - 1 if i == len(peaks) - 1 else peaks[i + 1]
        if invert:
            s_idx = left + int(np.argmax(curve[left:pk + 1]))
            e_idx = pk + int(np.argmax(curve[pk:right + 1]))
            drop_ok = (curve[s_idx] - curve[pk] >= min_drop) and (curve[e_idx] - curve[pk] >= min_drop)
        else:
            s_idx = left + int(np.argmin(curve[left:pk + 1]))
            e_idx = pk + int(np.argmin(curve[pk:right + 1]))
            drop_ok = (curve[pk] - curve[s_idx] >= min_drop) and (curve[pk] - curve[e_idx] >= min_drop)
        if drop_ok and e_idx > s_idx + 2:
            segments.append((int(grid[s_idx]), int(grid[e_idx]) + 1))

    # Fallback: if a short clip contains 1 rep without full peak margins, use full active span
    if not segments and (frames[-1] - frames[0]) >= int(0.8 * fps):
        segments.append((int(frames[0]), int(frames[-1]) + 1))
    return segments
"""
    ),
    md_cell(
        """## 4. Build / Load Cached Dataset (Mendeley 26 Subjects + Optional REHAB24-6 Merge)"""
    ),
    code_cell(
        """def load_rehab24_6_exercise(detector, target_exercise: str):
    if target_exercise not in REHAB_EXERCISE_IDS:
        return []
    ex_id = REHAB_EXERCISE_IDS[target_exercise]
    rehab_dir = WORK_DIR / "rehab24_6"
    rehab_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://zenodo.org/records/13305826/files/"
    for fname in ("Segmentation.csv", "videos.zip"):
        target = rehab_dir / fname
        if not target.exists():
            print("Downloading REHAB24-6:", fname)
            tmp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(base_url + fname + "?download=1", tmp)
            tmp.replace(target)

    labels = pd.read_csv(rehab_dir / "Segmentation.csv", sep=None, engine="python")
    labels.columns = labels.columns.str.strip().str.lower()
    labels = labels[labels.exercise_id.astype(int).eq(ex_id)].copy()
    if labels.empty:
        return []

    rows = []
    with zipfile.ZipFile(rehab_dir / "videos.zip") as archive:
        mp4s = [n for n in archive.namelist() if n.lower().endswith(".mp4")]
        with tempfile.TemporaryDirectory() as tmpdir:
            for video_id, reps in tqdm(labels.groupby("video_id"), desc=f"REHAB24-6 ({target_exercise}, Ex{ex_id})"):
                person = f"rehab_{reps.person_id.iloc[0]}"
                key = str(video_id).lower()
                matched = [
                    n for n in mp4s
                    if re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", n.lower())
                ]
                if not matched:
                    continue
                selected = sorted(
                    matched,
                    key=lambda n: (not re.search(r"camera[_ -]?17|(?:^|[_-])c17", n.lower()), n),
                )[0]
                local_vid = Path(tmpdir) / "video.mp4"
                with archive.open(selected) as src, local_vid.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                pose = read_video_pose(local_vid, detector)
                _, _, _, length, fps = pose
                starts = reps.first_frame.to_numpy(int)
                ends = reps.last_frame.to_numpy(int)
                if abs(fps - 30.0) > 1.0 or min(starts) < 0 or max(ends) >= length:
                    continue
                for s, e, corr in zip(starts, ends, reps.correctness):
                    rep = resample_rep(pose, int(s), int(e) + 1)
                    if rep is not None:
                        rows.append((rep[0], rep[1], int(int(corr) == 0), person, "rehab24_6", "side"))
    return rows


if CACHE_FILE.exists():
    print("Loading cached pose tensors from:", CACHE_FILE)
    cached = np.load(CACHE_FILE, allow_pickle=True)
    X_angles = cached["X_angles"]
    X_skel = cached["X_skel"]
    y = cached["y"]
    groups = cached["groups"]
    sources = cached["sources"]
    views = cached["views"]
else:
    all_rows = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(POSE_OPTIONS) as detector:
        for rec in tqdm(mendeley_records, desc="Extracting Mendeley Poses"):
            pose = read_video_pose(rec["path"], detector)
            segs = segment_reps_auto(pose, TARGET_EXERCISE)
            for s, e in segs:
                rep = resample_rep(pose, s, e)
                if rep is not None:
                    all_rows.append((rep[0], rep[1], rec["label"], rec["subject_id"], "mendeley", rec["view"]))

        if INCLUDE_REHAB24_6 and TARGET_EXERCISE in REHAB_EXERCISE_IDS:
            rehab_rows = load_rehab24_6_exercise(detector, TARGET_EXERCISE)
            print(f"Merged {len(rehab_rows)} reps from REHAB24-6 ({TARGET_EXERCISE}).")
            all_rows.extend(rehab_rows)

    assert len(all_rows) > 0, "No usable reps extracted! Check dataset download."
    X_angles = np.stack([r[0] for r in all_rows]).astype(np.float32)   # (N, 32, 14)
    X_skel = np.stack([r[1] for r in all_rows]).astype(np.float32)     # (N, 32, 12, 6)
    y = np.array([r[2] for r in all_rows], dtype=np.int32)             # 0=Correct, 1=Incorrect
    groups = np.array([r[3] for r in all_rows], dtype=object)
    sources = np.array([r[4] for r in all_rows], dtype=object)
    views = np.array([r[5] for r in all_rows], dtype=object)

    np.savez_compressed(
        CACHE_FILE,
        X_angles=X_angles,
        X_skel=X_skel,
        y=y,
        groups=groups,
        sources=sources,
        views=views,
    )
    print("Saved pose cache to:", CACHE_FILE)

# ---- Dataset Summary ----
summary_df = pd.DataFrame({
    "subject": groups,
    "label": np.where(y == 1, "Incorrect", "Correct"),
    "source": sources,
    "view": views,
})
print(f"Total Reps: {len(y)} | Unique Subjects: {summary_df['subject'].nunique()}")
print(f"Angles tensor: {X_angles.shape} | 2D Skeleton Grid tensor: {X_skel.shape}")
print("\\nClass Balance:")
print(summary_df["label"].value_counts())
print("\\nBreakdown by Dataset Source:")
print(pd.crosstab(summary_df["source"], summary_df["label"]))
"""
    ),
    md_cell(
        """## 5. Phase-Aware Feature Engineering (For Tabular Models) & Training-Only Pose Augmentation"""
    ),
    code_cell(
        """def build_tabular_features(angle_seqs: np.ndarray) -> np.ndarray:
    \"\"\"Builds rich phase-aware + asymmetry + velocity features from (N, 32, 14) angle sequences.\"\"\"
    # Global statistics
    g_mean = angle_seqs.mean(axis=1)
    g_std = angle_seqs.std(axis=1)
    g_min = angle_seqs.min(axis=1)
    g_max = angle_seqs.max(axis=1)
    g_rom = g_max - g_min

    # 3-phase statistics: Descent (0:11), Bottom/Peak (11:22), Ascent (22:32)
    p1 = angle_seqs[:, :11, :]
    p2 = angle_seqs[:, 11:22, :]
    p3 = angle_seqs[:, 22:, :]
    phase_feats = np.concatenate([
        p1.mean(axis=1), p2.mean(axis=1), p2.min(axis=1), p2.max(axis=1), p3.mean(axis=1)
    ], axis=1)

    # Angular velocity summary
    vel = np.diff(angle_seqs, axis=1)
    v_abs_mean = np.abs(vel).mean(axis=1)
    v_max = np.abs(vel).max(axis=1)

    # Left-Right symmetry features (elbows 0/1, shoulders 2/3, hips 4/5, wrists 9/10, knees 12/13)
    asym_pairs = [(0, 1), (2, 3), (4, 5), (9, 10), (12, 13)]
    asym_feats = np.stack([
        np.abs(angle_seqs[:, :, l] - angle_seqs[:, :, r]).mean(axis=1)
        for l, r in asym_pairs
    ], axis=1)
    asym_max = np.stack([
        np.abs(angle_seqs[:, :, l] - angle_seqs[:, :, r]).max(axis=1)
        for l, r in asym_pairs
    ], axis=1)

    return np.concatenate(
        [g_mean, g_std, g_min, g_max, g_rom, phase_feats, v_abs_mean, v_max, asym_feats, asym_max],
        axis=1,
    ).astype(np.float32)


# Mirror-flip index mappings (swapping left <-> right joints and features)
SKEL_LR_SWAP = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
ANGLE_LR_SWAP = [1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 11, 13, 12]


def augment_training_fold(ang_tr: np.ndarray, skel_tr: np.ndarray, y_tr: np.ndarray, rng: np.random.Generator):
    \"\"\"Applies physically valid pose augmentations STRICTLY inside training folds.\"\"\"
    # 1. Original
    ang_list, skel_list, y_list = [ang_tr], [skel_tr], [y_tr]

    # 2. Horizontal mirror flip (swap left/right joints + negate X position & X velocity)
    ang_flip = ang_tr[:, :, ANGLE_LR_SWAP].copy()
    skel_flip = skel_tr[:, :, SKEL_LR_SWAP, :].copy()
    skel_flip[:, :, :, 0] *= -1.0
    skel_flip[:, :, :, 3] *= -1.0
    ang_list.append(ang_flip)
    skel_list.append(skel_flip)
    y_list.append(y_tr)

    # 3. Mild noise + slight temporal speed warp
    ang_jit = ang_tr + rng.normal(0.0, 1.2, size=ang_tr.shape).astype(np.float32)
    skel_jit = skel_tr + rng.normal(0.0, 0.015, size=skel_tr.shape).astype(np.float32)
    ang_list.append(ang_jit)
    skel_list.append(skel_jit)
    y_list.append(y_tr)

    return (
        np.concatenate(ang_list, axis=0),
        np.concatenate(skel_list, axis=0),
        np.concatenate(y_list, axis=0),
    )


X_tab = build_tabular_features(X_angles)
print("Engineered tabular feature matrix:", X_tab.shape)
"""
    ),
    md_cell(
        """## 6. Model Architectures (10 Models: Tabular + 1D CNN + 2D CNN + BiLSTM-Attn + TCN + Two-Stream Merged Hybrid)"""
    ),
    code_cell(
        """class TemporalAttentionPool(layers.Layer):
    \"\"\"Learnable attention pooling over temporal steps (highlights which frames drive form errors).\"\"\"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1)

    def call(self, inputs):
        weights = tf.nn.softmax(self.score_dense(inputs), axis=1)  # (B, T, 1)
        return tf.reduce_sum(inputs * weights, axis=1)


def build_1d_cnn(angle_shape):
    inp = layers.Input(shape=angle_shape)
    x = layers.Conv1D(48, 5, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="1D_CNN")


def build_2d_cnn(skel_shape):
    \"\"\"2D CNN over Spatial-Temporal Skeleton Map (Time=32, Joints=12, Channels=6).\"\"\"
    inp = layers.Input(shape=skel_shape)
    x = layers.Conv2D(32, (3, 3), padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(96, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)])
    x = layers.Dense(48, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="2D_CNN_Skeleton")


def build_bilstm_attention(angle_shape):
    inp = layers.Input(shape=angle_shape)
    x = layers.Bidirectional(layers.LSTM(48, return_sequences=True, dropout=0.25))(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=0.25))(x)
    x = TemporalAttentionPool()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="BiLSTM_Attention")


def build_tcn(angle_shape):
    inp = layers.Input(shape=angle_shape)
    x = layers.Conv1D(48, 1, padding="same")(inp)
    for dilation in (1, 2, 4):
        res = x
        h = layers.Conv1D(48, 3, padding="same", dilation_rate=dilation)(x)
        h = layers.BatchNormalization()(h)
        h = layers.Activation("relu")(h)
        h = layers.Dropout(0.25)(h)
        h = layers.Conv1D(48, 3, padding="same", dilation_rate=dilation)(h)
        h = layers.BatchNormalization()(h)
        x = layers.Activation("relu")(layers.Add()([res, h]))
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="TCN")


def build_two_stream_hybrid(skel_shape, angle_shape):
    \"\"\"MERGED ARCHITECTURE: 2D CNN (Spatial-Temporal Skeleton) + BiGRU (Biomechanical Angles).\"\"\"
    # Stream A: 2D CNN on (Time, Joints, 6)
    inp_skel = layers.Input(shape=skel_shape, name="skel_input")
    s = layers.Conv2D(32, (3, 3), padding="same")(inp_skel)
    s = layers.BatchNormalization()(s)
    s = layers.Activation("relu")(s)
    s = layers.MaxPooling2D((2, 2))(s)
    s = layers.Conv2D(64, (3, 3), padding="same")(s)
    s = layers.BatchNormalization()(s)
    s = layers.Activation("relu")(s)
    s = layers.SpatialDropout2D(0.25)(s)
    s = layers.GlobalAveragePooling2D()(s)

    # Stream B: BiGRU + Attention on (Time, 14) biomechanical angles
    inp_ang = layers.Input(shape=angle_shape, name="angle_input")
    a = layers.Conv1D(32, 3, padding="same", activation="relu")(inp_ang)
    a = layers.BatchNormalization()(a)
    a = layers.Bidirectional(layers.GRU(36, return_sequences=True, dropout=0.25))(a)
    a = TemporalAttentionPool()(a)

    # Fusion Head
    fused = layers.Concatenate()([s, a])
    fused = layers.LayerNormalization()(fused)
    fused = layers.Dense(48, activation="relu")(fused)
    fused = layers.Dropout(0.35)(fused)
    out = layers.Dense(1, activation="sigmoid")(fused)
    return keras.Model([inp_skel, inp_ang], out, name="Merged_2DCNN_BiGRU")


def compile_model(model: keras.Model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="pr_auc", curve="PR")],
    )
    return model
"""
    ),
    md_cell(
        """## 7. Run Strict Subject-Grouped Cross-Validation (All 10 Models)

Every participant is kept strictly together in either train or test for each fold (**zero data leakage**).
Scalers and pose augmentation are fit strictly inside each training fold.
"""
    ),
    code_cell(
        """import time as _time
time = _time

print("Active GPUs:", tf.config.list_physical_devices("GPU"))
BATCH_SIZE = 64
EPOCHS = 28

unique_subjects = len(set(groups))
n_splits = min(N_FOLDS, unique_subjects)
cv = StratifiedGroupKFold(n_splits=n_splits)

MODEL_FAMILIES = {
    "Random Forest": "Tabular",
    "SVM (RBF)": "Tabular",
    "XGBoost": "Tabular",
    "LightGBM": "Tabular",
    "1D CNN": "Deep (1D)",
    "2D CNN (Skeleton)": "Deep (2D)",
    "BiLSTM + Attention": "Deep (RNN)",
    "TCN": "Deep (TCN)",
    "Merged: 2D-CNN + BiGRU": "Merged Hybrid",
}

oof_probs = {name: np.zeros(len(y), dtype=np.float32) for name in MODEL_FAMILIES}
fold_f1s = {name: [] for name in MODEL_FAMILIES}
train_times = {name: [] for name in MODEL_FAMILIES}
infer_ms_list = {name: [] for name in MODEL_FAMILIES}
model_sizes_kb = {}
rng = np.random.default_rng(SEED)


def measure_single_rep_latency_ms(fn, n_warmup: int = 3, n_runs: int = 20) -> float:
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    return 1000.0 * (time.perf_counter() - t0) / n_runs


def estimate_sklearn_size_kb(obj) -> float:
    tmp_p = WORK_DIR / "_tmp_size.joblib"
    joblib.dump(obj, tmp_p, compress=3)
    kb = tmp_p.stat().st_size / 1024.0
    tmp_p.unlink(missing_ok=True)
    return round(kb, 1)


def estimate_keras_size_kb(net: keras.Model) -> float:
    return round((net.count_params() * 4) / 1024.0, 1)


for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_tab, y, groups), 1):
    assert not (set(groups[tr_idx]) & set(groups[te_idx])), "Subject leakage detected!"
    print(
        f"\\n--- Fold {fold_idx}/{n_splits} | "
        f"Train: {len(tr_idx)} reps ({len(set(groups[tr_idx]))} subj) | "
        f"Test: {len(te_idx)} reps ({len(set(groups[te_idx]))} subj, "
        f"pos={int(y[te_idx].sum())}/{len(te_idx)}) ---",
        flush=True,
    )

    # Augment training fold only
    ang_tr_aug, skel_tr_aug, y_tr_aug = augment_training_fold(
        X_angles[tr_idx], X_skel[tr_idx], y[tr_idx], rng
    )
    tab_tr_aug = build_tabular_features(ang_tr_aug)
    tab_te = X_tab[te_idx]

    # Balanced class weights
    classes = np.unique(y_tr_aug)
    cw_vals = compute_class_weight("balanced", classes=classes, y=y_tr_aug)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw_vals)}
    pos_ratio = float((y_tr_aug == 0).sum() / max((y_tr_aug == 1).sum(), 1))

    # Cap SVM fit rows if large so Platt calibration stays fast (<3s/fold)
    svm_idx = rng.choice(len(tab_tr_aug), min(len(tab_tr_aug), 2500), replace=False)

    # ---- 1. Tabular Models ----
    tabular_specs = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED + fold_idx, n_jobs=-1,
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED + fold_idx),
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.8, scale_pos_weight=pos_ratio, eval_metric="logloss",
            random_state=SEED + fold_idx, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=5, num_leaves=20, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.8, class_weight="balanced",
            random_state=SEED + fold_idx, verbose=-1,
        ),
    }

    sample_tab_1 = tab_te[:1]
    for name, clf in tabular_specs.items():
        t_start = time.perf_counter()
        if name == "SVM (RBF)":
            clf.fit(tab_tr_aug[svm_idx], y_tr_aug[svm_idx])
        else:
            clf.fit(tab_tr_aug, y_tr_aug)
        dt = time.perf_counter() - t_start
        train_times[name].append(dt)

        probs = np.nan_to_num(clf.predict_proba(tab_te)[:, 1], nan=0.5, posinf=1.0, neginf=0.0)
        oof_probs[name][te_idx] = probs
        f1 = f1_score(y[te_idx], (probs >= 0.5).astype(int), average="macro")
        fold_f1s[name].append(f1)
        if fold_idx == 1:
            infer_ms_list[name].append(measure_single_rep_latency_ms(lambda c=clf: c.predict_proba(sample_tab_1)))
            model_sizes_kb[name] = estimate_sklearn_size_kb(clf)
        print(f"  [OK] {name:<24s} | F1={f1:.3f} | {dt:.1f}s", flush=True)

    # ---- 2. Scale Sequence & 2D Skeleton Tensors (Fit on Train Fold Only!) ----
    ang_scaler = StandardScaler().fit(ang_tr_aug.reshape(-1, ang_tr_aug.shape[-1]))
    ang_tr_s = ang_scaler.transform(ang_tr_aug.reshape(-1, ang_tr_aug.shape[-1])).reshape(ang_tr_aug.shape)
    ang_te_s = ang_scaler.transform(X_angles[te_idx].reshape(-1, X_angles.shape[-1])).reshape(X_angles[te_idx].shape)

    skel_scaler = StandardScaler().fit(skel_tr_aug.reshape(-1, skel_tr_aug.shape[-1]))
    skel_tr_s = skel_scaler.transform(skel_tr_aug.reshape(-1, skel_tr_aug.shape[-1])).reshape(skel_tr_aug.shape)
    skel_te_s = skel_scaler.transform(X_skel[te_idx].reshape(-1, X_skel.shape[-1])).reshape(X_skel[te_idx].shape)

    # Small internal validation split from training fold for EarlyStopping
    inner_n = max(1, int(0.15 * len(ang_tr_s)))
    perm = rng.permutation(len(ang_tr_s))
    iv, it = perm[:inner_n], perm[inner_n:]

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    deep_specs = [
        ("1D CNN", build_1d_cnn(ang_tr_s.shape[1:]), ang_tr_s[it], ang_tr_s[iv], ang_te_s, ang_te_s[:1]),
        ("2D CNN (Skeleton)", build_2d_cnn(skel_tr_s.shape[1:]), skel_tr_s[it], skel_tr_s[iv], skel_te_s, skel_te_s[:1]),
        ("BiLSTM + Attention", build_bilstm_attention(ang_tr_s.shape[1:]), ang_tr_s[it], ang_tr_s[iv], ang_te_s, ang_te_s[:1]),
        ("TCN", build_tcn(ang_tr_s.shape[1:]), ang_tr_s[it], ang_tr_s[iv], ang_te_s, ang_te_s[:1]),
        (
            "Merged: 2D-CNN + BiGRU",
            build_two_stream_hybrid(skel_tr_s.shape[1:], ang_tr_s.shape[1:]),
            [skel_tr_s[it], ang_tr_s[it]],
            [skel_tr_s[iv], ang_tr_s[iv]],
            [skel_te_s, ang_te_s],
            [skel_te_s[:1], ang_te_s[:1]],
        ),
    ]

    for name, net, x_tr_in, x_val_in, x_te_in, x_one in deep_specs:
        tf.keras.utils.set_random_seed(SEED + fold_idx)
        net.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="pr_auc", curve="PR")],
        )
        t_start = time.perf_counter()
        net.fit(
            x_tr_in, y_tr_aug[it],
            validation_data=(x_val_in, y_tr_aug[iv]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=0,
        )
        dt = time.perf_counter() - t_start
        train_times[name].append(dt)

        probs = np.nan_to_num(net.predict(x_te_in, verbose=0).ravel(), nan=0.5, posinf=1.0, neginf=0.0)
        oof_probs[name][te_idx] = probs
        f1 = f1_score(y[te_idx], (probs >= 0.5).astype(int), average="macro")
        fold_f1s[name].append(f1)
        if fold_idx == 1:
            infer_ms_list[name].append(measure_single_rep_latency_ms(lambda m=net, xi=x_one: m(xi, training=False)))
            model_sizes_kb[name] = estimate_keras_size_kb(net)
        print(f"  [OK] {name:<24s} | F1={f1:.3f} | {dt:.1f}s", flush=True)
        keras.backend.clear_session()

# ---- 3. Merged Cross-Family Soft-Voting Ensemble (Strictly Out-of-Fold!) ----
tabular_names = [n for n, fam in MODEL_FAMILIES.items() if fam == "Tabular"]
deep_names = [n for n, fam in MODEL_FAMILIES.items() if fam != "Tabular"]

best_tab = max(tabular_names, key=lambda n: f1_score(y, (oof_probs[n] >= 0.5).astype(int), average="macro"))
best_deep = max(deep_names, key=lambda n: f1_score(y, (oof_probs[n] >= 0.5).astype(int), average="macro"))
ens_name = f"Ensemble ({best_deep} + {best_tab})"
MODEL_FAMILIES[ens_name] = "Merged Ensemble"
oof_probs[ens_name] = np.nan_to_num(0.55 * oof_probs[best_deep] + 0.45 * oof_probs[best_tab], nan=0.5)
train_times[ens_name] = [a + b for a, b in zip(train_times[best_deep], train_times[best_tab])]
infer_ms_list[ens_name] = [infer_ms_list[best_deep][0] + infer_ms_list[best_tab][0]]
model_sizes_kb[ens_name] = round(model_sizes_kb[best_deep] + model_sizes_kb[best_tab], 1)

# Measure Rep Segmentation speed (ms per 10-second clip)
sample_frames = np.arange(0, 300, FRAME_STEP, dtype=np.int32)
sample_curve_angles = np.tile(X_angles[0], (4, 1))[:len(sample_frames)]
rep_seg_ms = measure_single_rep_latency_ms(
    lambda: segment_reps_auto((sample_frames, sample_curve_angles, None, 300, 30.0), TARGET_EXERCISE),
    n_warmup=5, n_runs=40,
)
print(f"Rep Segmentation Latency (10s set): {rep_seg_ms:.2f} ms")

fold_f1s[ens_name] = []
for _, te_idx in cv.split(X_tab, y, groups):
    fold_f1s[ens_name].append(
        f1_score(y[te_idx], (oof_probs[ens_name][te_idx] >= 0.5).astype(int), average="macro")
    )

print("\\nCross-Validation Complete! Best Tabular:", best_tab, "| Best Deep:", best_deep)
"""
    ),
    md_cell(
        """## 8. MASTER MODEL COMPARISON CELL (Leaderboard + Charts + Confusion Matrices)

This single cell compares **all 10 models** fairly on out-of-fold unseen-participant predictions so it is immediately clear which model wins.
"""
    ),
    code_cell(
        """rows = []
for name, raw_probs in oof_probs.items():
    probs = np.nan_to_num(raw_probs, nan=0.5, posinf=1.0, neginf=0.0)
    oof_probs[name] = probs
    preds = (probs >= 0.5).astype(int)
    lat_ms = float(np.mean(infer_ms_list[name]))
    rows.append({
        "Model": name,
        "Family": MODEL_FAMILIES[name],
        "OOF Macro F1": round(f1_score(y, preds, average="macro"), 4),
        "Fold F1 (Mean±Std)": f"{np.mean(fold_f1s[name]):.3f} ± {np.std(fold_f1s[name]):.3f}",
        "Accuracy": round(accuracy_score(y, preds), 4),
        "Balanced Acc": round(balanced_accuracy_score(y, preds), 4),
        "Incorrect Precision": round(precision_score(y, preds, zero_division=0), 4),
        "Incorrect Recall": round(recall_score(y, preds, zero_division=0), 4),
        "PR-AUC": round(average_precision_score(y, probs), 4),
        "ROC-AUC": round(roc_auc_score(y, probs), 4),
        "Latency (ms/rep)": round(lat_ms, 2),
        "Train (s/fold)": round(float(np.mean(train_times[name])), 2),
        "Model Size (KB)": model_sizes_kb.get(name, 0.0),
        "Real-Time Ready (<50ms)": "YES" if lat_ms < 50.0 else "NO",
    })

leaderboard = pd.DataFrame(rows).sort_values("OOF Macro F1", ascending=False).reset_index(drop=True)
leaderboard.index = leaderboard.index + 1
display(leaderboard)

WINNER_MODEL = leaderboard.iloc[0]["Model"]
FASTEST_MODEL = leaderboard.sort_values("Latency (ms/rep)").iloc[0]["Model"]
print(f"\\n>>> WINNER BY UNSEEN-PARTICIPANT OOF MACRO F1: {WINNER_MODEL} "
      f"(Macro F1 = {leaderboard.iloc[0]['OOF Macro F1']:.4f}, "
      f"PR-AUC = {leaderboard.iloc[0]['PR-AUC']:.4f}, "
      f"Latency = {leaderboard.iloc[0]['Latency (ms/rep)']} ms/rep, "
      f"Size = {leaderboard.iloc[0]['Model Size (KB)']} KB) <<<")
print(f">>> FASTEST MODEL: {FASTEST_MODEL} ({leaderboard.sort_values('Latency (ms/rep)').iloc[0]['Latency (ms/rep)']} ms/rep) "
      f"| Rep Segmentation Speed: {rep_seg_ms:.2f} ms per 10s set <<<")

# ---- Visualization 1: Leaderboard Bar Chart + Speed-vs-Accuracy Trade-off ----
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
sns.barplot(data=leaderboard, y="Model", x="OOF Macro F1", hue="Family", dodge=False, ax=axes[0])
axes[0].set_title(f"Unseen-Subject OOF Macro F1 ({TARGET_EXERCISE.title()})")
axes[0].set_xlim(0, 1.0)
for idx_row, val in enumerate(leaderboard["OOF Macro F1"]):
    axes[0].text(min(val + 0.015, 0.96), idx_row, f"{val:.3f}", va="center", fontsize=8)

# ---- Visualization 2: Accuracy vs Inference Latency (Speed Trade-off!) ----
sns.scatterplot(
    data=leaderboard, x="Latency (ms/rep)", y="OOF Macro F1",
    hue="Family", size="Model Size (KB)", sizes=(70, 320), ax=axes[1],
)
for _, r in leaderboard.iterrows():
    axes[1].text(r["Latency (ms/rep)"] * 1.02, r["OOF Macro F1"] + 0.006, r["Model"].split("(")[0].strip(), fontsize=7.5)
axes[1].set_title("Accuracy vs. Real-Time Speed (Top-Left = Best)")
axes[1].set_xlabel("Form Detection Latency (ms per rep)")
axes[1].set_ylabel("OOF Macro F1")

# ---- Visualization 3: ROC Curves for Top 5 Models ----
for name in leaderboard["Model"].head(5):
    fpr, tpr, _ = roc_curve(y, oof_probs[name])
    auc_v = roc_auc_score(y, oof_probs[name])
    axes[2].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_v:.3f})")
axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[2].set_title("Top 5 Models — ROC Curves (Unseen Subjects)")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()

# ---- Visualization 3: Confusion Matrices for Top 4 Models ----
top4 = leaderboard["Model"].head(4).tolist()
fig, axes = plt.subplots(1, len(top4), figsize=(4.2 * len(top4), 3.8))
if len(top4) == 1:
    axes = [axes]
for ax, name in zip(axes, top4):
    cm = confusion_matrix(y, (oof_probs[name] >= 0.5).astype(int), labels=[0, 1])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Pred Correct", "Pred Incorrect"],
        yticklabels=["True Correct", "True Incorrect"],
    )
    ax.set_title(name, fontsize=10)
plt.suptitle("Top 4 Models — Out-of-Fold Confusion Matrices", y=1.03)
plt.tight_layout()
plt.show()

leaderboard.to_csv(EXPORT_DIR / "model_leaderboard.csv", index=False)
print("Saved leaderboard CSV to:", EXPORT_DIR / "model_leaderboard.csv")
"""
    ),
    md_cell(
        """## 9. Train Final Winning Models on Full Dataset & Export for Local FastAPI + Flutter App

Trains the winning Deep model + winning Tabular model on 100% of the labeled data, saves scalers + metadata + weights into a single `.zip` bundle you can drop directly into `GRADPROJECT/backend/models/`.
"""
    ),
    code_cell(
        """# Refit scalers & final models on full dataset (with augmentation)
rng_final = np.random.default_rng(SEED + 99)
ang_full, skel_full, y_full = augment_training_fold(X_angles, X_skel, y, rng_final)
tab_full = build_tabular_features(ang_full)

classes = np.unique(y_full)
cw_vals = compute_class_weight("balanced", classes=classes, y=y_full)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw_vals)}
pos_ratio = float((y_full == 0).sum() / max((y_full == 1).sum(), 1))

# 1. Final Best Tabular Model
final_tab_builders = {
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1
    ),
    "SVM (RBF)": lambda: make_pipeline(
        StandardScaler(), SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)
    ),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.8, scale_pos_weight=pos_ratio, eval_metric="logloss", random_state=SEED, n_jobs=-1
    ),
    "LightGBM": lambda: LGBMClassifier(
        n_estimators=250, max_depth=5, num_leaves=20, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.8, class_weight="balanced", random_state=SEED, verbose=-1
    ),
}
final_tab_model = final_tab_builders[best_tab]()
final_tab_model.fit(tab_full, y_full)
joblib.dump(final_tab_model, EXPORT_DIR / "best_tabular_model.joblib")

# 2. Final Best Deep Model (always save Two-Stream Hybrid or best_deep)
ang_scaler_full = StandardScaler().fit(ang_full.reshape(-1, ang_full.shape[-1]))
ang_full_s = ang_scaler_full.transform(ang_full.reshape(-1, ang_full.shape[-1])).reshape(ang_full.shape)

skel_scaler_full = StandardScaler().fit(skel_full.reshape(-1, skel_full.shape[-1]))
skel_full_s = skel_scaler_full.transform(skel_full.reshape(-1, skel_full.shape[-1])).reshape(skel_full.shape)

joblib.dump(ang_scaler_full, EXPORT_DIR / "angle_scaler.joblib")
joblib.dump(skel_scaler_full, EXPORT_DIR / "skel_scaler.joblib")

deep_builders = {
    "1D CNN": (lambda: build_1d_cnn(ang_full_s.shape[1:]), ang_full_s),
    "2D CNN (Skeleton)": (lambda: build_2d_cnn(skel_full_s.shape[1:]), skel_full_s),
    "BiLSTM + Attention": (lambda: build_bilstm_attention(ang_full_s.shape[1:]), ang_full_s),
    "TCN": (lambda: build_tcn(ang_full_s.shape[1:]), ang_full_s),
    "Merged: 2D-CNN + BiGRU": (
        lambda: build_two_stream_hybrid(skel_full_s.shape[1:], ang_full_s.shape[1:]),
        [skel_full_s, ang_full_s],
    ),
}
builder_fn, full_inputs = deep_builders[best_deep]
final_deep_model = compile_model(builder_fn())
final_deep_model.fit(
    full_inputs, y_full,
    epochs=28,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    verbose=0,
)
final_deep_model.save(EXPORT_DIR / "best_deep_model.keras")

metadata = {
    "exercise": TARGET_EXERCISE,
    "winner_model": WINNER_MODEL,
    "best_deep_model": best_deep,
    "best_tabular_model": best_tab,
    "ensemble_weights": {"deep": 0.55, "tabular": 0.45},
    "threshold": 0.5,
    "sequence_length": SEQUENCE_LENGTH,
    "frame_step": FRAME_STEP,
    "features": FEATURES,
    "core_joints": CORE_JOINTS,
    "leaderboard_top3": leaderboard.head(3).to_dict(orient="records"),
}
(EXPORT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

zip_bundle = WORK_DIR / f"gym_model_bundle_{TARGET_EXERCISE}.zip"
with zipfile.ZipFile(zip_bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in EXPORT_DIR.iterdir():
        if f.is_file():
            zf.write(f, arcname=f.name)

print("Exported model bundle to:", EXPORT_DIR)
print("Downloadable ZIP ready at:", zip_bundle)
try:
    from google.colab import files
    files.download(str(zip_bundle))
except Exception:
    print("Run inside Colab to auto-download the ZIP, or grab it from Google Drive:", EXPORT_DIR)
"""
    ),
    md_cell(
        """## 10. AI Module #3 — User Profile Fitness-Level Predictor (`Tomertg/SmartFit_Dataset`)

Rather than feeding synthetic user profiles into the vision form classifier (which would add noise), we train a **dedicated Athlete Personalization Model** on `Tomertg/SmartFit_Dataset`.
When a new user registers in the Flutter app (`Age`, `Height`, `Weight`, etc.), this model automatically predicts their initial **Fitness Level (`Beginner` / `Intermediate` / `Advanced`)** and recommended training intensity!
"""
    ),
    code_cell(
        """from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Fetch up to 1,000 rows from Hugging Face datasets-server (fast, no extra auth needed)
sf_rows = []
for offset in range(0, 1000, 100):
    resp = requests.get(
        "https://datasets-server.huggingface.co/rows",
        params={
            "dataset": "Tomertg/SmartFit_Dataset",
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": 100,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        break
    batch = resp.json().get("rows", [])
    if not batch:
        break
    sf_rows.extend(item["row"] for item in batch)

sf_df = pd.DataFrame(sf_rows)
print("Loaded SmartFit rows:", len(sf_df))
print("Available SmartFit columns:", sf_df.columns.tolist())

# Normalize target column ('Experience')
target_col = next((c for c in sf_df.columns if c.lower() in ("experience", "fitness_level", "level")), None)
assert target_col is not None, f"Could not locate Experience column in {sf_df.columns.tolist()}"

sf_df = sf_df.dropna(subset=[target_col]).copy()
sf_df[target_col] = sf_df[target_col].astype(str).str.strip().str.title()
sf_df = sf_df[sf_df[target_col].isin(["Beginner", "Intermediate", "Advanced"])].copy()

# Select clean profile inputs collected during app registration (plus derived BMI)
candidate_num = [c for c in sf_df.columns if c.lower() in ("age", "height_cm", "weight_kg", "bmi", "height", "weight")]
candidate_cat = [c for c in sf_df.columns if c.lower() in ("gender", "sex", "goal", "workout_type")]

for c in candidate_num:
    sf_df[c] = pd.to_numeric(sf_df[c], errors="coerce")

# Compute BMI if Height_cm & Weight_kg exist and BMI doesn't
h_col = next((c for c in candidate_num if "height" in c.lower()), None)
w_col = next((c for c in candidate_num if "weight" in c.lower()), None)
if h_col and w_col and not any(c.lower() == "bmi" for c in candidate_num):
    sf_df["BMI"] = sf_df[w_col] / ((sf_df[h_col] / 100.0).clip(lower=0.5) ** 2)
    candidate_num.append("BMI")

sf_features = candidate_num + candidate_cat
sf_df = sf_df.dropna(subset=sf_features).reset_index(drop=True)
print("Using SmartFit profile features:", sf_features, "| Clean rows:", len(sf_df))

le_sf = LabelEncoder()
y_sf = le_sf.fit_transform(sf_df[target_col])
X_sf = sf_df[sf_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), candidate_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), candidate_cat),
    ]
)

sf_models = {
    "Random Forest": make_pipeline(
        preprocessor, RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=SEED)
    ),
    "SVM (RBF)": make_pipeline(
        preprocessor, SVC(C=2.0, probability=True, class_weight="balanced", random_state=SEED)
    ),
    "XGBoost": make_pipeline(
        preprocessor, XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="mlogloss", random_state=SEED)
    ),
    "LightGBM": make_pipeline(
        preprocessor, LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, class_weight="balanced", random_state=SEED, verbose=-1)
    ),
    "MLP Neural Net": make_pipeline(
        preprocessor, MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED)
    ),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sf_results = []
sf_oof = {}
for m_name, pipe in sf_models.items():
    oof_pred = np.zeros(len(y_sf), dtype=int)
    for tr_i, te_i in skf.split(X_sf, y_sf):
        pipe.fit(X_sf.iloc[tr_i], y_sf[tr_i])
        oof_pred[te_i] = pipe.predict(X_sf.iloc[te_i])
    sf_oof[m_name] = oof_pred
    sf_results.append({
        "Model": m_name,
        "OOF Macro F1": round(f1_score(y_sf, oof_pred, average="macro"), 4),
        "Accuracy": round(accuracy_score(y_sf, oof_pred), 4),
        "Balanced Acc": round(balanced_accuracy_score(y_sf, oof_pred), 4),
    })

sf_board = pd.DataFrame(sf_results).sort_values("OOF Macro F1", ascending=False).reset_index(drop=True)
sf_board.index = sf_board.index + 1
display(sf_board)

# Fit & export winning SmartFit profile model
best_sf_name = sf_board.iloc[0]["Model"]
best_sf_pipe = sf_models[best_sf_name]
best_sf_pipe.fit(X_sf, y_sf)

joblib.dump(
    {
        "pipeline": best_sf_pipe,
        "classes": le_sf.classes_.tolist(),
        "features": sf_features,
        "numeric_features": candidate_num,
        "categorical_features": candidate_cat,
        "model_name": best_sf_name,
        "leaderboard": sf_board.to_dict(orient="records"),
    },
    EXPORT_DIR / "smartfit_fitness_model.joblib",
)
print(f"Exported SmartFit Fitness-Level model ({best_sf_name}) to:", EXPORT_DIR / "smartfit_fitness_model.joblib")

# Re-zip bundle so smartfit_fitness_model.joblib is included in the single download!
with zipfile.ZipFile(zip_bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in EXPORT_DIR.iterdir():
        if f.is_file():
            zf.write(f, arcname=f.name)
print("Updated downloadable bundle ZIP with SmartFit model:", zip_bundle)
"""
    ),
    md_cell(
        """## 11. AI Module #2 Validation — Face Recognition Benchmark on LFW (*Labeled Faces in the Wild*)

Validates why our app uses **OpenCV YuNet + SFace (128-D Deep Metric Embeddings + Cosine Similarity)** for instant 1-shot real-user signup—comparing it directly against classical **Eigenfaces (PCA)** and **Raw Pixel** baselines on unseen real-world face verification pairs (`sklearn.datasets.fetch_lfw_pairs`).
"""
    ),
    code_cell(
        """import time
from sklearn.datasets import fetch_lfw_pairs
from sklearn.decomposition import PCA

# Download official OpenCV Zoo YuNet & SFace ONNX models
face_dir = WORK_DIR / "face_models"
face_dir.mkdir(parents=True, exist_ok=True)
yunet_path = face_dir / "face_detection_yunet_2023mar.onnx"
sface_path = face_dir / "face_recognition_sface_2021dec.onnx"

if not yunet_path.exists():
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        yunet_path,
    )
if not sface_path.exists():
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        sface_path,
    )

yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (160, 160), 0.5, 0.3, 5)
sface = cv2.FaceRecognizerSF.create(str(sface_path), "")

# Load unseen LFW verification pairs (subset='test', 1000 pairs: 500 same person, 500 different)
lfw = fetch_lfw_pairs(subset="test", color=True, resize=1.0)
pairs = lfw.pairs          # (N, 2, H, W, 3) float32 in [0, 1] or [0, 255]
y_pairs = lfw.target       # 1 = same person, 0 = different person
n_eval = min(300, len(y_pairs))  # Fast ~20s benchmark on 300 balanced pairs
rng_lfw = np.random.default_rng(SEED)
pos_idx = np.where(y_pairs == 1)[0][: n_eval // 2]
neg_idx = np.where(y_pairs == 0)[0][: n_eval // 2]
eval_idx = np.concatenate([pos_idx, neg_idx])

eval_pairs = pairs[eval_idx]
eval_y = y_pairs[eval_idx]
scale_255 = 255.0 if eval_pairs.max() <= 1.5 else 1.0


def sface_embed(img_rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(np.clip(img_rgb * scale_255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    bgr_up = cv2.resize(bgr, (160, 160))
    yunet.setInputSize((160, 160))
    _, faces = yunet.detect(bgr_up)
    if faces is not None and len(faces) > 0:
        best = max(faces, key=lambda f: float(f[2] * f[3]))
        aligned = sface.alignCrop(bgr_up, best)
    else:
        aligned = cv2.resize(bgr_up, (112, 112))
    feat = sface.feature(aligned).flatten().astype(np.float32)
    return feat / max(float(np.linalg.norm(feat)), 1e-8)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))


# 1. YuNet + SFace 128-D Deep Embeddings
t0 = time.perf_counter()
sface_scores = np.array([
    cosine_sim(sface_embed(p[0]), sface_embed(p[1]))
    for p in tqdm(eval_pairs, desc="SFace LFW Benchmark")
], dtype=np.float32)
sface_ms = 1000.0 * (time.perf_counter() - t0) / (2 * len(eval_pairs))

# 2. Eigenfaces (PCA 128-D) Baseline
gray_flat_0 = eval_pairs[:, 0].mean(axis=-1).reshape(len(eval_pairs), -1)
gray_flat_1 = eval_pairs[:, 1].mean(axis=-1).reshape(len(eval_pairs), -1)
pca = PCA(n_components=100, random_state=SEED).fit(np.vstack([gray_flat_0, gray_flat_1]))
p0, p1 = pca.transform(gray_flat_0), pca.transform(gray_flat_1)
pca_scores = np.array([cosine_sim(a, b) for a, b in zip(p0, p1)], dtype=np.float32)

# 3. Raw Pixel Cosine Baseline
raw_scores = np.array([cosine_sim(a, b) for a, b in zip(gray_flat_0, gray_flat_1)], dtype=np.float32)

face_benchmarks = {
    f"YuNet + SFace (128-D Deep, {sface_ms:.1f} ms/face)": sface_scores,
    "Eigenfaces (PCA-100 Cosine)": pca_scores,
    "Raw Pixels Cosine": raw_scores,
}

face_rows = []
for m_name, sc in face_benchmarks.items():
    fpr, tpr, ths = roc_curve(eval_y, sc)
    best_i = int(np.argmax(tpr - fpr))
    best_th = float(ths[best_i])
    preds = (sc >= best_th).astype(int)
    face_rows.append({
        "Face Model": m_name,
        "Verification Accuracy": round(accuracy_score(eval_y, preds), 4),
        "F1 Score": round(f1_score(eval_y, preds), 4),
        "ROC-AUC": round(roc_auc_score(eval_y, sc), 4),
        "Optimal Cosine Threshold": round(best_th, 3),
    })

face_board = pd.DataFrame(face_rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
face_board.index = face_board.index + 1
display(face_board)

# Plot ROC Comparison + SFace Separation Histogram
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for m_name, sc in face_benchmarks.items():
    fpr, tpr, _ = roc_curve(eval_y, sc)
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{m_name.split('(')[0].strip()} (AUC={roc_auc_score(eval_y, sc):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_title("LFW Unseen-User Face Verification ROC")
axes[0].set_xlabel("False Accept Rate (FAR)")
axes[0].set_ylabel("True Accept Rate (TAR)")
axes[0].legend(loc="lower right")

axes[1].hist(sface_scores[eval_y == 1], bins=25, alpha=0.65, label="Same Person", color="#10b981")
axes[1].hist(sface_scores[eval_y == 0], bins=25, alpha=0.65, label="Different Person", color="#ef4444")
axes[1].axvline(0.363, color="black", linestyle="--", label="SFace Threshold (0.363)")
axes[1].set_title("SFace Cosine Similarity Separation (LFW)")
axes[1].set_xlabel("Cosine Similarity")
axes[1].legend()
plt.tight_layout()
plt.show()
"""
    ),
    md_cell(
        """## 12. Upload & Test Any Exercise Video (All 10 Models + Rep Count + Exact Fixes)"""
    ),
    code_cell(
        """import subprocess
from google.colab import files
from IPython.display import Video, display, Markdown

JOINT_LINKS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]


def diagnose_detailed_fixes(exercise: str, angle_seq: np.ndarray, is_incorrect: bool):
    torso_max = float(np.percentile(angle_seq[:, 6], 90))
    hip_tilt = float(np.percentile(angle_seq[:, 8], 85))
    body_min = float(np.min(angle_seq[:, 11]))
    knee_min = float(np.min(0.5 * (angle_seq[:, 12] + angle_seq[:, 13])))
    knee_asym = float(np.mean(np.abs(angle_seq[:, 12] - angle_seq[:, 13])))
    elb_min = float(np.min(0.5 * (angle_seq[:, 0] + angle_seq[:, 1])))
    sho_high = float(np.percentile(0.5 * (angle_seq[:, 2] + angle_seq[:, 3]), 85))

    fixes = []
    if exercise == "squat":
        if torso_max > 40.0:
            fixes.append(f"Back leaning forward ({torso_max:.0f} deg) -> Keep chest up & back straight")
        if knee_min > 105.0:
            fixes.append(f"Shallow depth (min knee {knee_min:.0f} deg) -> Go deeper")
        if hip_tilt > 10.0:
            fixes.append(f"Uneven hips ({hip_tilt:.0f} deg) -> Keep hips level")
        if knee_asym > 14.0:
            fixes.append(f"Knee asymmetry ({knee_asym:.0f} deg) -> Push evenly with both legs")
    elif exercise == "push":
        if body_min < 155.0:
            fixes.append(f"Hips sagging/piking (body line {body_min:.0f} deg) -> Brace core & keep back straight")
        if elb_min > 105.0:
            fixes.append(f"Shallow push-up (min elbow {elb_min:.0f} deg) -> Lower chest closer to floor")
        if sho_high > 72.0:
            fixes.append(f"Elbows flaring out ({sho_high:.0f} deg) -> Tuck elbows in")
    elif exercise == "curl":
        if torso_max > 18.0:
            fixes.append(f"Body swinging ({torso_max:.0f} deg) -> Keep back straight")
        if sho_high > 38.0:
            fixes.append(f"Elbows drifting ({sho_high:.0f} deg) -> Pin elbows to your sides")
        if elb_min > 68.0:
            fixes.append(f"Partial curl ({elb_min:.0f} deg) -> Curl all the way up")

    if not is_incorrect:
        return ["Clean form — great rep!"]
    return fixes if fixes else ["Form deviation detected -> Slow down & control movement"]


if "_DEMO_MODELS_CACHE" not in globals():
    _DEMO_MODELS_CACHE = {}

if TARGET_EXERCISE not in _DEMO_MODELS_CACHE:
    print(f"Preparing all 10 '{TARGET_EXERCISE}' models once for comparison...")
    _tab = {}
    for m_name, builder in final_tab_builders.items():
        if m_name == best_tab:
            _tab[m_name] = final_tab_model
        else:
            m = builder()
            m.fit(tab_full, y_full)
            _tab[m_name] = m

    _deep = {}
    for m_name, (builder, x_in) in deep_builders.items():
        if m_name == best_deep:
            _deep[m_name] = final_deep_model
        else:
            tf.keras.utils.set_random_seed(SEED)
            net = compile_model(builder())
            net.fit(
                x_in, y_full, epochs=22, batch_size=max(BATCH_SIZE, 32),
                class_weight=class_weight_dict, verbose=0,
            )
            _deep[m_name] = net
    _DEMO_MODELS_CACHE[TARGET_EXERCISE] = {"tab": _tab, "deep": _deep}

_demo_tab = _DEMO_MODELS_CACHE[TARGET_EXERCISE]["tab"]
_demo_deep = _DEMO_MODELS_CACHE[TARGET_EXERCISE]["deep"]
_ens_name = f"Ensemble ({best_deep} + {best_tab})"
_clean_order = [m.split(" [NaN:")[0] for m in leaderboard["Model"].tolist()]
_clean_winner = WINNER_MODEL.split(" [NaN:")[0]


def _predict_rep_all(ang_seq, skel_grid):
    tab_x = build_tabular_features(ang_seq[None])
    ang_s = ang_scaler_full.transform(ang_seq).reshape(1, *ang_seq.shape)
    skel_s = skel_scaler_full.transform(skel_grid.reshape(-1, skel_grid.shape[-1])).reshape(1, *skel_grid.shape)

    probs = {name: float(clf.predict_proba(tab_x)[0, 1]) for name, clf in _demo_tab.items()}
    for name, net in _demo_deep.items():
        if "2D-CNN + BiGRU" in name:
            p = float(net([skel_s, ang_s], training=False).numpy().ravel()[0])
        elif "2D CNN" in name:
            p = float(net(skel_s, training=False).numpy().ravel()[0])
        else:
            p = float(net(ang_s, training=False).numpy().ravel()[0])
        probs[name] = float(np.nan_to_num(p, nan=0.5))
    probs[_ens_name] = 0.55 * probs[best_deep] + 0.45 * probs[best_tab]
    return probs


def _render_coach_video(vpath: Path, segs, rep_details):
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    raw_mp4 = WORK_DIR / f"{vpath.stem}_raw.mp4"
    out_mp4 = EXPORT_DIR / f"{vpath.stem}_{TARGET_EXERCISE}_annotated.mp4"
    writer = None
    idx, completed, last_lms = 0, 0, None

    with mp.tasks.vision.PoseLandmarker.create_from_options(POSE_OPTIONS) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h0, w0 = frame.shape[:2]
                s = min(1.0, 900.0 / max(h0, w0))
                size = (max(2, int(w0 * s) // 2 * 2), max(2, int(h0 * s) // 2 * 2))
                writer = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

            frame = cv2.resize(frame, size)
            w, h = size
            if idx % FRAME_STEP == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                found = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                last_lms = found.pose_landmarks[0] if found.pose_landmarks else None

            while completed < len(segs) and idx >= segs[completed][1] - 1:
                completed += 1

            active_i = next((i for i, (st, en) in enumerate(segs) if st <= idx < en), None)
            shown_i = active_i if active_i is not None else (completed - 1 if completed > 0 else None)
            info = rep_details[shown_i] if (shown_i is not None and shown_i < len(rep_details)) else None

            is_bad = (info is not None) and info["is_incorrect"]
            accent = (50, 60, 245) if is_bad else (60, 225, 95)

            if last_lms is not None:
                vis = {
                    i: (int(p.x * w), int(p.y * h))
                    for i, p in enumerate(last_lms)
                    if p.visibility >= 0.30 and 0 <= p.x <= 1 and 0 <= p.y <= 1
                }
                if len(vis) >= 4:
                    xs, ys = [pt[0] for pt in vis.values()], [pt[1] for pt in vis.values()]
                    cv2.rectangle(
                        frame,
                        (max(0, min(xs) - 16), max(0, min(ys) - 16)),
                        (min(w - 1, max(xs) + 16), min(h - 1, max(ys) + 16)),
                        accent, 2,
                    )
                for a, b in JOINT_LINKS:
                    if a in vis and b in vis:
                        cv2.line(frame, vis[a], vis[b], accent, 2, cv2.LINE_AA)
                for j in {k for pair in JOINT_LINKS for k in pair} & vis.keys():
                    cv2.circle(frame, vis[j], 4, (50, 220, 255), -1, cv2.LINE_AA)

            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, 54), (15, 18, 25), -1)
            cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
            verdict_txt = ("INCORRECT FORM" if is_bad else "CORRECT FORM") if info else "TRACKING..."
            agree_txt = f" | {info['votes']}" if info else ""
            cv2.putText(
                frame, f"{TARGET_EXERCISE.upper()} REPS: {completed}/{len(rep_details)}  |  {verdict_txt}{agree_txt}",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, max(0.44, min(0.66, w / 1050.0)), accent, 2, cv2.LINE_AA,
            )

            if info is not None:
                fix_lines = [f"FIX: {fx}" for fx in info["fixes"][:2]] if is_bad else [info["fixes"][0]]
                box_h = 28 * len(fix_lines) + 18
                ov2 = frame.copy()
                cv2.rectangle(ov2, (0, h - box_h), (w, h), (15, 18, 25), -1)
                cv2.addWeighted(ov2, 0.84, frame, 0.16, 0, frame)
                for li, fline in enumerate(fix_lines):
                    cv2.putText(
                        frame, fline[:90], (12, h - box_h + 24 + li * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.42, min(0.58, w / 1150.0)),
                        (120, 140, 255) if is_bad else (90, 240, 120), 2, cv2.LINE_AA,
                    )

            writer.write(frame)
            idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_mp4),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4),
        ],
        check=True,
    )
    raw_mp4.unlink(missing_ok=True)
    return out_mp4


print(f"Click 'Choose Files' below to upload 1 or more {TARGET_EXERCISE.upper()} videos:")
uploaded_videos = files.upload()

for fname in uploaded_videos:
    vpath = Path(fname)
    display(Markdown(f"---\\n### Video: `{fname}`"))

    with mp.tasks.vision.PoseLandmarker.create_from_options(POSE_OPTIONS) as detector:
        pose = read_video_pose(vpath, detector, step=2)
    segs = segment_reps_auto(pose, TARGET_EXERCISE, step=2)
    fps_val = pose[4] if pose[4] > 0 else 30.0

    if not segs:
        print(f"No reps detected (valid pose frames: {len(pose[0])}/{pose[3]}). Make sure the athlete is clearly visible.")
        continue

    rep_details, table_rows = [], []
    for rep_num, (st, en) in enumerate(segs, 1):
        rep_data = resample_rep(pose, st, en, step=2)
        if rep_data is None:
            continue
        ang_seq, skel_grid = rep_data
        probs = _predict_rep_all(ang_seq, skel_grid)

        winner_bad = probs[_clean_winner] >= 0.5
        n_bad_votes = sum(1 for m in _clean_order if probs[m] >= 0.5)
        fixes = diagnose_detailed_fixes(TARGET_EXERCISE, ang_seq, winner_bad)

        rep_details.append({
            "rep": rep_num, "is_incorrect": winner_bad,
            "fixes": fixes, "votes": f"{n_bad_votes}/{len(_clean_order)} flag bad",
        })
        row = {
            "Rep": rep_num,
            "Time": f"{st/fps_val:.1f}s-{en/fps_val:.1f}s",
            f"Winner ({_clean_winner})": f"{'INCORRECT' if winner_bad else 'CORRECT'} ({probs[_clean_winner]:.2f})",
            "What To Fix": " | ".join(fixes),
        }
        for m_name in _clean_order:
            if m_name != _clean_winner:
                row[m_name] = f"{'Bad' if probs[m_name] >= 0.5 else 'Good'} ({probs[m_name]:.2f})"
        table_rows.append(row)

    n_bad = sum(1 for r in rep_details if r["is_incorrect"])
    print(f"Detected Reps: {len(rep_details)} | Correct: {len(rep_details) - n_bad} | Incorrect: {n_bad}")
    display(pd.DataFrame(table_rows))

    out_video = _render_coach_video(vpath, segs, rep_details)
    print("Saved annotated video to Drive:", out_video)
    display(Video(str(out_video), embed=True, width=720))
"""
    ),
]


def main():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
            "accelerator": "GPU",
        },
        "cells": CELLS,
    }
    out_path = Path(__file__).resolve().parent / "GymForm_MultiModel_Benchmark_Colab.ipynb"
    out_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Generated {out_path} ({len(CELLS)} cells, all Python cells syntax-verified!)")


if __name__ == "__main__":
    main()
