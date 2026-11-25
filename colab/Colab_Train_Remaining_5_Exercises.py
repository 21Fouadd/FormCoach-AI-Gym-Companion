# === Train Real ML Models (Random Forest + 4-Model Subject-Grouped Benchmark) for All 5 Remaining Exercises ===
# Paste this ENTIRE script into ONE Google Colab cell and run it (~5-8 min total on Colab).
# Trains: curl, shoulder, row (from Mendeley kgbb3yn47p v3) + lunge, arm_abduction (from REHAB24-6 Zenodo)
# Exports ONE ZIP (`gym_model_bundle_remaining5.zip`) compatible 1-to-1 with `backend/app/pose_engine.py`.

import importlib, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request, zipfile
from pathlib import Path

for pkg in ("mediapipe", "lightgbm", "xgboost", "scikit-learn", "joblib", "scipy", "tqdm"):
    mod = "sklearn" if pkg == "scikit-learn" else pkg
    if importlib.util.find_spec(mod) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm
from xgboost import XGBClassifier

try:
    from google.colab import drive, files
except ImportError:
    drive = None
    files = None

SEED = 42
FRAME_STEP = 3
SEQ_LEN = 32
N_FOLDS = 5
EXERCISES_TO_TRAIN = ["curl", "shoulder", "row", "lunge", "arm_abduction"]

MENDELEY_KEYWORDS = {
    "curl": ["curl", "bicep"],
    "shoulder": ["shoulder", "press"],
    "row": ["row", "bent"],
}
REHAB_IDS = {
    "arm_abduction": 1,
    "lunge": 5,
}
CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
FEATURES = [
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "torso_tilt", "shoulder_line_tilt",
    "hip_line_tilt", "left_wrist_height", "right_wrist_height",
    "body_line", "left_knee", "right_knee",
]
ANGLE_LR_SWAP = [1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 11, 13, 12]

WORK_DIR = Path("/content/gymform_all5")
WORK_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_DIR = Path("/content/drive/MyDrive/GymForm_Graduation_Project")
if drive is not None and not os.path.exists("/content/drive/MyDrive"):
    try:
        drive.mount("/content/drive")
    except Exception:
        pass
SAVE_ROOT = DRIVE_DIR if os.path.exists("/content/drive/MyDrive") else WORK_DIR
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

POSE_TASK = WORK_DIR / "pose_landmarker_full.task"
if not POSE_TASK.exists() or POSE_TASK.stat().st_size < 100_000:
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        POSE_TASK,
    )
POSE_OPTS = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_TASK)),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
)


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-8), -1.0, 1.0))))


def extract_frame_pose(res):
    if not res.pose_landmarks or not res.pose_world_landmarks:
        return None
    l2 = res.pose_landmarks[0]
    chains = [
        all(l2[i].visibility >= 0.28 for i in (11, 23, 25, 27)),
        all(l2[i].visibility >= 0.28 for i in (12, 24, 26, 28)),
        all(l2[i].visibility >= 0.28 for i in (11, 13, 15, 23)),
        all(l2[i].visibility >= 0.28 for i in (12, 14, 16, 24)),
        l2[11].visibility >= 0.35 and l2[13].visibility >= 0.20 and l2[15].visibility >= 0.14,
        l2[12].visibility >= 0.35 and l2[14].visibility >= 0.20 and l2[16].visibility >= 0.14,
    ]
    if not any(chains):
        return None
    hv = max(l2[23].visibility, l2[24].visibility) >= 0.35
    p = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_world_landmarks[0]], dtype=np.float32)
    sh, hp = 0.5 * (p[11] + p[12]), 0.5 * (p[23] + p[24])
    rs = float(np.linalg.norm(sh - hp))
    sc = rs if (hv and rs >= 0.12) else max(rs, float(np.linalg.norm(p[11] - p[12])) * 1.35, 0.20)
    if sc < 1e-5:
        return None
    tilt = lambda i, j: float(np.degrees(np.arctan2(abs(p[i, 1] - p[j, 1]), np.hypot(p[i, 0] - p[j, 0], p[i, 2] - p[j, 2]))))
    ang = np.array([
        calc_angle(p[11], p[13], p[15]), calc_angle(p[12], p[14], p[16]),
        calc_angle(p[23], p[11], p[13]), calc_angle(p[24], p[12], p[14]),
        calc_angle(p[11], p[23], p[25]), calc_angle(p[12], p[24], p[26]),
        calc_angle(sh, hp, hp + np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        tilt(11, 12), tilt(23, 24),
        float((p[11, 1] - p[15, 1]) / sc), float((p[12, 1] - p[16, 1]) / sc),
        calc_angle(sh, hp, 0.5 * (p[27] + p[28])),
        calc_angle(p[23], p[25], p[27]), calc_angle(p[24], p[26], p[28]),
    ], dtype=np.float32)
    return ang if np.isfinite(ang).all() else None


def read_video_angles(vpath: Path, det, step: int = FRAME_STEP):
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frames, angles, idx = [], [], 0
    try:
        while cap.grab():
            if idx % step == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    h, w = bgr.shape[:2]
                    if max(h, w) > 640:
                        s = 640.0 / max(h, w)
                        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
                    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                    ang = extract_frame_pose(res)
                    if ang is not None:
                        frames.append(idx)
                        angles.append(ang)
            idx += 1
    finally:
        cap.release()
    return np.asarray(frames, np.int32), np.asarray(angles, np.float32), idx, float(fps)


def _rom(x: np.ndarray) -> float:
    return float(np.percentile(x, 90) - np.percentile(x, 10)) if len(x) else 0.0


def segment_reps(frames: np.ndarray, angles: np.ndarray, fps: float, ex: str, step: int = FRAME_STEP):
    if len(frames) < 6 or fps <= 0:
        return []
    if ex == "lunge":
        sig, inv, pr, dr = angles[:, 12:14].min(1), True, 20.0, 16.0
    elif ex == "curl":
        sig, inv, pr, dr = angles[:, :2].min(1), True, 15.0, 13.0
    elif ex in ("shoulder", "arm_abduction"):
        sig, inv, pr, dr = angles[:, 2:4].max(1), False, 20.0, 16.0
    elif ex == "row":
        if max(_rom(angles[:, 0]), _rom(angles[:, 1])) >= max(30.0, 0.7 * _rom(angles[:, 4:6].mean(1))):
            sig, inv, pr, dr = angles[:, :2].min(1), True, 15.0, 13.0
        else:
            sig, inv, pr, dr = angles[:, 4:6].mean(1), True, 18.0, 15.0
    else:
        sig, inv, pr, dr = angles[:, 12:14].mean(1), True, 20.0, 16.0

    grid = np.arange(frames[0], frames[-1] + 1, step)
    curve = np.interp(grid, frames, sig)
    if len(curve) >= 5:
        curve = np.convolve(curve, [0.25, 0.5, 0.25], mode="same")
    pks, _ = find_peaks(-curve if inv else curve, prominence=pr, distance=max(2, int(0.65 * fps / step)))
    segs = []
    for i, pk in enumerate(pks):
        l, r = (0 if i == 0 else pks[i - 1]), (len(curve) - 1 if i == len(pks) - 1 else pks[i + 1])
        s_i = l + int(np.argmax(curve[l:pk + 1]) if inv else np.argmin(curve[l:pk + 1]))
        e_i = pk + int(np.argmax(curve[pk:r + 1]) if inv else np.argmin(curve[pk:r + 1]))
        ok = (curve[s_i] - curve[pk] >= dr and curve[e_i] - curve[pk] >= dr) if inv else (curve[pk] - curve[s_i] >= dr and curve[pk] - curve[e_i] >= dr)
        if ok and e_i > s_i + 2:
            segs.append((int(grid[s_i]), int(grid[e_i]) + 1))
    if not segs and (frames[-1] - frames[0]) >= int(0.8 * fps):
        segs.append((int(frames[0]), int(frames[-1]) + 1))
    return segs


def resample_angles(frames: np.ndarray, angles: np.ndarray, start: int, stop: int, step: int = FRAME_STEP):
    m = (frames >= start) & (frames < stop)
    if m.sum() < max(4, ((stop - start) / step) * 0.25):
        return None
    grid = np.linspace(float(start), float(max(start + 1, stop - 1)), SEQ_LEN)
    return np.stack([np.interp(grid, frames[m], angles[m, c]) for c in range(14)], axis=-1).astype(np.float32)


def build_tabular_features(angle_seqs: np.ndarray) -> np.ndarray:
    """1-to-1 match with `backend/app/pose_engine.py::build_tabular_features` (178 features)."""
    p1, p2, p3 = angle_seqs[:, :11, :], angle_seqs[:, 11:22, :], angle_seqs[:, 22:, :]
    vel = np.diff(angle_seqs, axis=1)
    pairs = [(0, 1), (2, 3), (4, 5), (9, 10), (12, 13)]
    return np.concatenate([
        angle_seqs.mean(1), angle_seqs.std(1), angle_seqs.min(1), angle_seqs.max(1), angle_seqs.max(1) - angle_seqs.min(1),
        p1.mean(1), p2.mean(1), p2.min(1), p2.max(1), p3.mean(1),
        np.abs(vel).mean(1), np.abs(vel).max(1),
        np.stack([np.abs(angle_seqs[:, :, l] - angle_seqs[:, :, r]).mean(1) for l, r in pairs], 1),
        np.stack([np.abs(angle_seqs[:, :, l] - angle_seqs[:, :, r]).max(1) for l, r in pairs], 1),
    ], axis=1).astype(np.float32)


def ensure_mendeley_zip() -> Path:
    # Reuse already-downloaded zip from earlier Colab runs if present!
    for cand in [
        Path("/content/gymform_work/mendeley_videos/Dataset Exercise-wise.zip"),
        WORK_DIR / "mendeley" / "Dataset Exercise-wise.zip",
    ]:
        if cand.exists() and zipfile.is_zipfile(cand):
            print("Reusing cached Mendeley ZIP:", cand)
            return cand

    dest = WORK_DIR / "mendeley" / "Dataset Exercise-wise.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        "https://data.mendeley.com/public-api/datasets/kgbb3yn47p/files?version=3&folder_id=root",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    resp.raise_for_status()
    items = resp.json()
    items = items if isinstance(items, list) else items.get("data", [])
    chosen = next((x for x in items if "exercise-wise" in x.get("filename", "").lower()), items[0])
    url = (chosen.get("content_details") or {}).get("download_url") or (
        f"https://api.data.mendeley.com/datasets/kgbb3yn47p/files/{chosen['id']}/file_downloaded?version=3"
    )
    total = int((chosen.get("content_details") or {}).get("size") or 0)
    print(f"Downloading Mendeley {chosen['filename']} ({total/1e9:.2f} GB)...")
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Mendeley ZIP") as pb:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pb.update(len(chunk))
    tmp.replace(dest)
    return dest


def ensure_rehab_files() -> tuple[Path, Path]:
    for base in [Path("/content/gymform_work/rehab24_6"), WORK_DIR / "rehab24_6"]:
        seg, vid = base / "Segmentation.csv", base / "videos.zip"
        if seg.exists() and vid.exists() and zipfile.is_zipfile(vid):
            print("Reusing cached REHAB24-6 files:", base)
            return seg, vid
    base = WORK_DIR / "rehab24_6"
    base.mkdir(parents=True, exist_ok=True)
    for fn in ("Segmentation.csv", "videos.zip"):
        p = base / fn
        if not p.exists():
            print("Downloading REHAB24-6:", fn)
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(f"https://zenodo.org/records/13305826/files/{fn}?download=1", tmp)
            tmp.replace(p)
    return base / "Segmentation.csv", base / "videos.zip"


def parse_mendeley_meta(rel: str):
    low = rel.lower().replace("\\", "/")
    if re.search(r"(^|[/ _-])(bad|incorrect|wrong|poor|fault)([/ _.-]|$)", low):
        lbl = 1
    elif re.search(r"(^|[/ _-])(good|correct|proper|right)([/ _.-]|$)", low):
        lbl = 0
    else:
        return None
    m = re.search(r"(?:subject|subj|sub|person|participant|user|p|s)[ _-]?0*([0-9]{1,3})(?![0-9])", low)
    subj = f"mendeley_S{int(m.group(1)):02d}" if m else f"mendeley_{Path(rel).stem}"
    return subj, lbl


def extract_mendeley_exercise(ex: str, zip_path: Path, det):
    kws = MENDELEY_KEYWORDS[ex]
    rows = []
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as td:
        members = [
            m for m in zf.namelist()
            if m.lower().endswith((".mp4", ".mov", ".avi", ".mkv")) and any(kw in m.lower() for kw in kws)
        ]
        seen = set()
        for mem in tqdm(members, desc=f"Extracting Mendeley ({ex})"):
            bn = Path(mem).name.lower()
            if bn in seen:
                continue
            meta = parse_mendeley_meta(mem)
            if meta is None:
                continue
            seen.add(bn)
            subj, lbl = meta
            local_v = Path(td) / "clip.mp4"
            with zf.open(mem) as src, local_v.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            fr, ang, _, fps = read_video_angles(local_v, det)
            for s, e in segment_reps(fr, ang, fps, ex):
                seq = resample_angles(fr, ang, s, e)
                if seq is not None:
                    rows.append((seq, lbl, subj))
    return rows


def extract_rehab_exercise(ex: str, seg_csv: Path, vid_zip: Path, det):
    ex_id = REHAB_IDS[ex]
    df = pd.read_csv(seg_csv, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.lower()
    df = df[df.exercise_id.astype(int).eq(ex_id)].copy()
    rows = []
    with zipfile.ZipFile(vid_zip) as zf, tempfile.TemporaryDirectory() as td:
        mp4s = [n for n in zf.namelist() if n.lower().endswith(".mp4")]
        for vid_id, reps in tqdm(df.groupby("video_id"), desc=f"Extracting REHAB24-6 ({ex}, Ex{ex_id})"):
            subj = f"rehab_{reps.person_id.iloc[0]}"
            key = str(vid_id).lower()
            matched = [n for n in mp4s if re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", n.lower())]
            if not matched:
                continue
            chosen = sorted(matched, key=lambda n: (not re.search(r"camera[_ -]?17|(?:^|[_-])c17", n.lower()), n))[0]
            local_v = Path(td) / "clip.mp4"
            with zf.open(chosen) as src, local_v.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            fr, ang, length, fps = read_video_angles(local_v, det)
            starts, ends = reps.first_frame.to_numpy(int), reps.last_frame.to_numpy(int)
            if abs(fps - 30.0) > 1.5 or min(starts) < 0 or max(ends) >= length:
                continue
            for s, e, corr in zip(starts, ends, reps.correctness):
                seq = resample_angles(fr, ang, int(s), int(e) + 1)
                if seq is not None:
                    rows.append((seq, int(int(corr) == 0), subj))
    return rows


def train_and_export_exercise(ex: str, rows: list[tuple[np.ndarray, int, str]], export_root: Path):
    X_ang = np.stack([r[0] for r in rows]).astype(np.float32)
    y = np.asarray([r[1] for r in rows], dtype=np.int32)
    groups = np.asarray([r[2] for r in rows], dtype=object)
    X_tab = build_tabular_features(X_ang)
    n_subj = len(set(groups))
    print(f"\n=== [{ex.upper()}] Reps: {len(y)} | Subjects: {n_subj} | Correct={(y==0).sum()} / Incorrect={(y==1).sum()} ===")

    n_splits = min(N_FOLDS, n_subj)
    cv = StratifiedGroupKFold(n_splits=n_splits)
    rng = np.random.default_rng(SEED)
    names = ["Random Forest", "LightGBM", "XGBoost", "SVM (RBF)"]
    oof = {n: np.zeros(len(y), dtype=np.float32) for n in names}
    lat = {n: [] for n in names}

    for fold_i, (tr, te) in enumerate(cv.split(X_tab, y, groups), 1):
        # Training-only mirror flip + mild jitter augmentation (zero leakage)
        ang_tr = np.concatenate([
            X_ang[tr],
            X_ang[tr][:, :, ANGLE_LR_SWAP],
            X_ang[tr] + rng.normal(0.0, 1.2, size=X_ang[tr].shape).astype(np.float32),
        ], axis=0)
        y_tr = np.concatenate([y[tr], y[tr], y[tr]], axis=0)
        tab_tr = build_tabular_features(ang_tr)
        pos_ratio = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

        clfs = {
            "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=SEED + fold_i, n_jobs=-1),
            "LightGBM": LGBMClassifier(n_estimators=200, max_depth=5, num_leaves=20, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8, class_weight="balanced", random_state=SEED + fold_i, verbose=-1),
            "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8, scale_pos_weight=pos_ratio, eval_metric="logloss", random_state=SEED + fold_i, n_jobs=-1),
            "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED + fold_i)),
        }
        for name, clf in clfs.items():
            clf.fit(tab_tr, y_tr)
            oof[name][te] = clf.predict_proba(X_tab[te])[:, 1]
            t0 = time.perf_counter()
            for _ in range(10):
                clf.predict_proba(X_tab[te[:1]])
            lat[name].append(1000.0 * (time.perf_counter() - t0) / 10.0)

    board_rows = []
    for name in names:
        preds = (oof[name] >= 0.5).astype(int)
        board_rows.append({
            "Model": name,
            "Family": "Tabular",
            "OOF Macro F1": round(float(f1_score(y, preds, average="macro")), 4),
            "Accuracy": round(float(accuracy_score(y, preds)), 4),
            "Balanced Acc": round(float(balanced_accuracy_score(y, preds)), 4),
            "Incorrect Precision": round(float(precision_score(y, preds, zero_division=0)), 4),
            "Incorrect Recall": round(float(recall_score(y, preds, zero_division=0)), 4),
            "ROC-AUC": round(float(roc_auc_score(y, oof[name])), 4),
            "Latency (ms/rep)": round(float(np.mean(lat[name])), 2),
        })
    board = pd.DataFrame(board_rows).sort_values("OOF Macro F1", ascending=False).reset_index(drop=True)
    print(board.to_string(index=False))

    # Fit final Random Forest on 100% of augmented data (100% compatible with local sklearn venv!)
    ang_full = np.concatenate([X_ang, X_ang[:, :, ANGLE_LR_SWAP]], axis=0)
    y_full = np.concatenate([y, y], axis=0)
    rf_final = RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_final.fit(build_tabular_features(ang_full), y_full)

    ex_dir = export_root / ex
    ex_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_final, ex_dir / "best_tabular_model.joblib", compress=3)
    board.to_csv(ex_dir / "model_leaderboard.csv", index=False)
    meta = {
        "exercise": ex,
        "winner_model": str(board.iloc[0]["Model"]),
        "best_tabular_model": "Random Forest",
        "threshold": 0.5,
        "sequence_length": SEQ_LEN,
        "frame_step": FRAME_STEP,
        "features": FEATURES,
        "core_joints": CORE_JOINTS,
        "total_reps": int(len(y)),
        "unique_subjects": int(n_subj),
        "leaderboard": board.to_dict(orient="records"),
    }
    (ex_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported {ex} Random Forest bundle -> {ex_dir}")


# === Run All 5 Exercises in One Batch ===
mendeley_zip = ensure_mendeley_zip()
rehab_csv, rehab_zip = ensure_rehab_files()
EXPORT_ALL5 = SAVE_ROOT / "exported_remaining5"
EXPORT_ALL5.mkdir(parents=True, exist_ok=True)

with mp.tasks.vision.PoseLandmarker.create_from_options(POSE_OPTS) as detector:
    for ex in EXERCISES_TO_TRAIN:
        cache_p = SAVE_ROOT / f"pose_angles_cache_{ex}.npz"
        if cache_p.exists():
            c = np.load(cache_p, allow_pickle=True)
            rows = list(zip(c["X_ang"], c["y"], c["groups"]))
            print(f"Loaded cached {ex} poses ({len(rows)} reps)")
        else:
            rows = extract_mendeley_exercise(ex, mendeley_zip, detector) if ex in MENDELEY_KEYWORDS else extract_rehab_exercise(ex, rehab_csv, rehab_zip, detector)
            np.savez_compressed(cache_p, X_ang=np.stack([r[0] for r in rows]), y=np.asarray([r[1] for r in rows]), groups=np.asarray([r[2] for r in rows], dtype=object))
        train_and_export_exercise(ex, rows, EXPORT_ALL5)

zip_out = SAVE_ROOT / "gym_model_bundle_remaining5.zip"
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in EXPORT_ALL5.rglob("*"):
        if p.is_file():
            zf.write(p, arcname=p.relative_to(EXPORT_ALL5).as_posix())
print("\nALL 5 MODELS PACKAGED:", zip_out, f"({zip_out.stat().st_size / 1e6:.2f} MB)")
if files is not None:
    files.download(str(zip_out))
