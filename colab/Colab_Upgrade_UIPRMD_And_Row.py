# === Upgrade Models with UI-PRMD (Zenodo 23104965) + Train Missing Bent-Over Row (Mendeley) ===
# 100% Open Downloads (Zero Forms/Logins) -> Run in Google Colab (~4-6 min total)
# Upgrades: squat (+10 UI-PRMD subj), lunge (+10 UI-PRMD subj), arm_abduction (+10 UI-PRMD subj) + trains row (Mendeley)
# Downloads ONE ZIP (`gym_model_bundle_upgraded.zip`) ready for `scripts/install_model_bundles.py`.

import importlib, io, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request, zipfile
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
SEQ_LEN = 32
FRAME_STEP = 3
ANGLE_LR_SWAP = [1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 11, 13, 12]
CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
FEATURES = [
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "torso_tilt", "shoulder_line_tilt",
    "hip_line_tilt", "left_wrist_height", "right_wrist_height",
    "body_line", "left_knee", "right_knee",
]

WORK_DIR = Path("/content/gymform_upgrade")
WORK_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_DIR = Path("/content/drive/MyDrive/GymForm_Graduation_Project")
if drive is not None and not os.path.exists("/content/drive/MyDrive"):
    try:
        drive.mount("/content/drive")
    except Exception:
        pass
SAVE_ROOT = DRIVE_DIR if os.path.exists("/content/drive/MyDrive") else WORK_DIR
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# -------- 1. UI-PRMD Exact Forward Kinematics (from Animation.m, Zenodo 23104965) --------
def _euler_zyx_batch(deg_xyz: np.ndarray) -> np.ndarray:
    """Converts (T, 3) [gamma_x, beta_y, alpha_z] degrees -> (T, 3, 3) Rz @ Ry @ Rx matrices."""
    rad = np.radians(deg_xyz)
    cx, sx = np.cos(rad[:, 0]), np.sin(rad[:, 0])
    cy, sy = np.cos(rad[:, 1]), np.sin(rad[:, 1])
    cz, sz = np.cos(rad[:, 2]), np.sin(rad[:, 2])
    R = np.zeros((len(deg_xyz), 3, 3), dtype=np.float32)
    R[:, 0, 0] = cz * cy
    R[:, 0, 1] = cz * sy * sx - sz * cx
    R[:, 0, 2] = cz * sy * cx + sz * sx
    R[:, 1, 0] = sz * cy
    R[:, 1, 1] = sz * sy * sx + cz * cx
    R[:, 1, 2] = sz * sy * cx - cz * sx
    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sx
    R[:, 2, 2] = cy * cx
    return R


def uiprmd_fk_to_14_angles(pos_arr: np.ndarray, ang_arr: np.ndarray) -> np.ndarray | None:
    """Converts UI-PRMD Kinect (T, 66) positions & (T, 66) Euler angles -> exact (T, 14) biomechanical angles."""
    T = min(len(pos_arr), len(ang_arr))
    if T < 8 or pos_arr.shape[1] < 66 or ang_arr.shape[1] < 66:
        return None
    J = pos_arr[:T, :66].reshape(T, 22, 3).astype(np.float32).copy()
    A = ang_arr[:T, :66].reshape(T, 22, 3).astype(np.float32)

    def apply_chain(chain_idxs: list[int], root_rot_idx: int, root_pos_idx: int):
        rot = _euler_zyx_batch(A[:, root_rot_idx])
        parent = root_pos_idx
        for idx in chain_idxs:
            J[:, idx] = np.einsum("tij,tj->ti", rot, J[:, idx]) + J[:, parent]
            rot = np.einsum("tij,tjk->tik", rot, _euler_zyx_batch(A[:, idx]))
            parent = idx

    # Exact kinematic chains from UI-PRMD Animation.m (0-indexed!)
    apply_chain([1, 2, 3, 4, 5], 0, 0)       # Waist -> Spine -> Chest -> Neck -> Head -> HeadTip
    apply_chain([6, 7, 8, 9], 2, 2)          # Chest -> L.Collar -> L.Shoulder(7) -> L.Elbow(8) -> L.Wrist(9)
    apply_chain([10, 11, 12, 13], 2, 2)      # Chest -> R.Collar -> R.Shoulder(11) -> R.Elbow(12) -> R.Wrist(13)
    apply_chain([14, 15, 16, 17], 0, 0)      # Waist -> L.Hip(14) -> L.Knee(15) -> L.Ankle(16) -> L.Toes
    apply_chain([18, 19, 20, 21], 0, 0)      # Waist -> R.Hip(18) -> R.Knee(19) -> R.Ankle(20) -> R.Toes

    def ang_vec(a, b, c):
        u, v = a - b, c - b
        cos = np.sum(u * v, axis=-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-8)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).astype(np.float32)

    sh = 0.5 * (J[:, 7] + J[:, 11])
    hp = 0.5 * (J[:, 14] + J[:, 18])
    ank = 0.5 * (J[:, 16] + J[:, 20])
    sc = np.maximum(np.linalg.norm(sh - hp, axis=-1), 1e-5)

    def tilt(i, j):
        v = J[:, i] - J[:, j]
        return np.degrees(np.arctan2(np.abs(v[:, 1]), np.maximum(np.hypot(v[:, 0], v[:, 2]), 1e-8))).astype(np.float32)

    up_target = hp + np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # +Y is UP in UI-PRMD
    out = np.stack([
        ang_vec(J[:, 7], J[:, 8], J[:, 9]),
        ang_vec(J[:, 11], J[:, 12], J[:, 13]),
        ang_vec(J[:, 14], J[:, 7], J[:, 8]),
        ang_vec(J[:, 18], J[:, 11], J[:, 12]),
        ang_vec(J[:, 7], J[:, 14], J[:, 15]),
        ang_vec(J[:, 11], J[:, 18], J[:, 19]),
        ang_vec(sh, hp, up_target),
        tilt(7, 11),
        tilt(14, 18),
        ((J[:, 9, 1] - J[:, 7, 1]) / sc).astype(np.float32),
        ((J[:, 13, 1] - J[:, 11, 1]) / sc).astype(np.float32),
        ang_vec(sh, hp, ank),
        ang_vec(J[:, 14], J[:, 15], J[:, 16]),
        ang_vec(J[:, 18], J[:, 19], J[:, 20]),
    ], axis=-1).astype(np.float32)
    return out if np.isfinite(out).all() else None


def resample_32(angles_sub: np.ndarray) -> np.ndarray:
    t_old = np.linspace(0.0, 1.0, len(angles_sub))
    t_new = np.linspace(0.0, 1.0, SEQ_LEN)
    return np.stack([np.interp(t_new, t_old, angles_sub[:, c]) for c in range(14)], axis=-1).astype(np.float32)


def download_file(url: str, dest: Path, desc: str) -> Path:
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"[cached] {desc}: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pb:
            for ch in r.iter_content(chunk_size=1024 * 1024):
                if ch:
                    f.write(ch)
                    pb.update(len(ch))
    tmp.replace(dest)
    return dest


def _parse_uiprmd_txt(raw_bytes: bytes) -> np.ndarray | None:
    text = raw_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    delim = "," if "," in text.splitlines()[0] else None
    try:
        arr = np.loadtxt(io.StringIO(text), delimiter=delim, dtype=np.float32)
        return arr if arr.ndim == 2 else None
    except Exception:
        return None


def _file_key(name: str) -> tuple[str, str, str] | None:
    bn = Path(name).name.lower()
    m = re.search(r"(m0[1-9]|m10)", bn)
    s = re.search(r"s(\d+)", bn)
    e = re.search(r"e(\d+)", bn)
    if not (m and s):
        return None
    return (m.group(1), f"s{int(s.group(1)):02d}", f"e{int(e.group(1)):02d}" if e else "e00")


def load_uiprmd_rows() -> dict[str, list[tuple[np.ndarray, int, str]]]:
    """Downloads UI-PRMD Correct & Incorrect Kinect skeletons and returns rows for squat, lunge, arm_abduction."""
    z_corr = download_file(
        "https://zenodo.org/records/23104965/files/Segmented%20Movements.zip?download=1",
        WORK_DIR / "uiprmd_correct.zip",
        "UI-PRMD Correct (207 MB)",
    )
    z_inc = download_file(
        "https://zenodo.org/records/23104965/files/Incorrect%20Movements.zip?download=1",
        WORK_DIR / "uiprmd_incorrect.zip",
        "UI-PRMD Incorrect (278 MB)",
    )
    mv_map = {"m01": "squat", "m03": "lunge", "m07": "arm_abduction"}
    collected: dict[str, list[tuple[np.ndarray, int, str]]] = {ex: [] for ex in mv_map.values()}

    for zpath, label in ((z_corr, 0), (z_inc, 1)):
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".txt") and "kinect" in n.lower()]
            if not names:
                names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            pos_files = [n for n in names if "position" in n.lower()]
            ang_files = [n for n in names if "angle" in n.lower()]
            ang_by_key = {}
            for an in ang_files:
                k = _file_key(an)
                if k is not None:
                    ang_by_key[k] = an
            print(f"UI-PRMD label={label}: {len(pos_files)} pos files, {len(ang_files)} ang files | sample: {pos_files[:2]} | {ang_files[:2]}")

            for p_name in tqdm(pos_files, desc=f"Parsing UI-PRMD (label={label})"):
                k = _file_key(p_name)
                if k is None or k[0] not in mv_map or k not in ang_by_key:
                    continue
                ex = mv_map[k[0]]
                pos_arr = _parse_uiprmd_txt(zf.read(p_name))
                ang_arr = _parse_uiprmd_txt(zf.read(ang_by_key[k]))
                if pos_arr is None or ang_arr is None:
                    continue
                ang14 = uiprmd_fk_to_14_angles(pos_arr, ang_arr)
                if ang14 is None:
                    continue
                subj = f"uiprmd_{k[1].upper()}"
                if len(ang14) > 220:
                    sig = ang14[:, 12:14].mean(1) if ex in ("squat", "lunge") else ang14[:, 2:4].max(1)
                    inv = ex in ("squat", "lunge")
                    pks, _ = find_peaks(-sig if inv else sig, prominence=15.0, distance=22)
                    added = 0
                    for i, pk in enumerate(pks):
                        l = 0 if i == 0 else pks[i - 1]
                        r = len(sig) - 1 if i == len(pks) - 1 else pks[i + 1]
                        s_i = l + int(np.argmax(sig[l:pk + 1]) if inv else np.argmin(sig[l:pk + 1]))
                        e_i = pk + int(np.argmax(sig[pk:r + 1]) if inv else np.argmin(sig[pk:r + 1]))
                        if e_i - s_i >= 10:
                            collected[ex].append((resample_32(ang14[s_i:e_i + 1]), label, subj))
                            added += 1
                    if added == 0:
                        # Split evenly into ~100-frame windows if peaks are shallow (common in incorrect reps!)
                        n_chunks = max(1, len(ang14) // 100)
                        for chunk in np.array_split(ang14, n_chunks):
                            if len(chunk) >= 12:
                                collected[ex].append((resample_32(chunk), label, subj))
                else:
                    collected[ex].append((resample_32(ang14), label, subj))

    for ex, rows in collected.items():
        print(f"UI-PRMD [{ex}]: {len(rows)} reps across {len({r[2] for r in rows})} subjects (Correct={sum(r[1]==0 for r in rows)}, Incorrect={sum(r[1]==1 for r in rows)})")
    return collected


# -------- 2. Extract Bent-Over Row from Mendeley kgbb3yn47p (with auto folder discovery!) --------
def extract_mendeley_row() -> list[tuple[np.ndarray, int, str]]:
    zip_cands = [
        Path("/content/gymform_work/mendeley_videos/Dataset Exercise-wise.zip"),
        Path("/content/gymform_all5/mendeley/Dataset Exercise-wise.zip"),
        WORK_DIR / "Dataset Exercise-wise.zip",
    ]
    zpath = next((p for p in zip_cands if p.exists() and zipfile.is_zipfile(p)), None)
    if zpath is None:
        resp = requests.get(
            "https://data.mendeley.com/public-api/datasets/kgbb3yn47p/files?version=3&folder_id=root",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=60,
        ).json()
        items = resp if isinstance(resp, list) else resp.get("data", [])
        chosen = next((x for x in items if "exercise-wise" in x.get("filename", "").lower()), items[0])
        url = (chosen.get("content_details") or {}).get("download_url") or (
            f"https://api.data.mendeley.com/datasets/kgbb3yn47p/files/{chosen['id']}/file_downloaded?version=3"
        )
        zpath = download_file(url, WORK_DIR / "Dataset Exercise-wise.zip", "Mendeley Exercise-wise ZIP")

    pose_task = WORK_DIR / "pose_landmarker_full.task"
    if not pose_task.exists():
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            pose_task,
        )
    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_task)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )

    def calc_a(a, b, c):
        u, v = a - b, c - b
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-8), -1.0, 1.0))))

    rows = []
    with zipfile.ZipFile(zpath) as zf, tempfile.TemporaryDirectory() as td, mp.tasks.vision.PoseLandmarker.create_from_options(opts) as det:
        mp4s = [m for m in zf.namelist() if m.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
        folders = sorted({"/".join(m.split("/")[:3]) for m in mp4s})
        print("Mendeley folders found:", folders[:20])
        row_mp4s = [m for m in mp4s if any(k in m.lower() for k in ("row", "bent", "bor", "back_row", "dumbbell_row"))]
        print(f"Matched {len(row_mp4s)} Bent-Over Row videos in Mendeley.")

        seen = set()
        for mem in tqdm(row_mp4s, desc="Extracting Mendeley Row"):
            bn = Path(mem).name.lower()
            if bn in seen:
                continue
            low = mem.lower().replace("\\", "/")
            if re.search(r"(^|[/ _-])(bad|incorrect|wrong|poor|fault)([/ _.-]|$)", low):
                lbl = 1
            elif re.search(r"(^|[/ _-])(good|correct|proper|right)([/ _.-]|$)", low):
                lbl = 0
            else:
                continue
            seen.add(bn)
            sm = re.search(r"(?:subject|subj|sub|person|participant|user|p|s)[ _-]?0*([0-9]{1,3})(?![0-9])", low)
            subj = f"mendeley_S{int(sm.group(1)):02d}" if sm else f"mendeley_{Path(mem).stem}"
            lv = Path(td) / "clip.mp4"
            with zf.open(mem) as src, lv.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            cap = cv2.VideoCapture(str(lv))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frs, angs, idx = [], [], 0
            while cap.grab():
                if idx % FRAME_STEP == 0:
                    ok, bgr = cap.retrieve()
                    if ok:
                        h, w = bgr.shape[:2]
                        if max(h, w) > 540:
                            s = 540.0 / max(h, w)
                            bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
                        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                        if res.pose_landmarks and res.pose_world_landmarks:
                            p = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_world_landmarks[0]], np.float32)
                            sh, hp = 0.5 * (p[11] + p[12]), 0.5 * (p[23] + p[24])
                            sc = max(float(np.linalg.norm(sh - hp)), 0.15)
                            t = lambda i, j: float(np.degrees(np.arctan2(abs(p[i, 1] - p[j, 1]), np.hypot(p[i, 0] - p[j, 0], p[i, 2] - p[j, 2]))))
                            a = np.array([
                                calc_a(p[11], p[13], p[15]), calc_a(p[12], p[14], p[16]),
                                calc_a(p[23], p[11], p[13]), calc_a(p[24], p[12], p[14]),
                                calc_a(p[11], p[23], p[25]), calc_a(p[12], p[24], p[26]),
                                calc_a(sh, hp, hp + np.array([0, -1, 0], np.float32)),
                                t(11, 12), t(23, 24), float((p[11, 1] - p[15, 1]) / sc), float((p[12, 1] - p[16, 1]) / sc),
                                calc_a(sh, hp, 0.5 * (p[27] + p[28])), calc_a(p[23], p[25], p[27]), calc_a(p[24], p[26], p[28]),
                            ], np.float32)
                            if np.isfinite(a).all():
                                frs.append(idx); angs.append(a)
                idx += 1
            cap.release()
            if len(frs) < 6:
                continue
            fa, aa = np.asarray(frs), np.asarray(angs)
            sig = aa[:, :2].min(1)
            grid = np.arange(fa[0], fa[-1] + 1, FRAME_STEP)
            c = np.convolve(np.interp(grid, fa, sig), [0.25, 0.5, 0.25], "same") if len(grid) >= 5 else np.interp(grid, fa, sig)
            pks, _ = find_peaks(-c, prominence=14.0, distance=max(2, int(0.65 * fps / FRAME_STEP)))
            for i, pk in enumerate(pks):
                l, r = (0 if i == 0 else pks[i - 1]), (len(c) - 1 if i == len(pks) - 1 else pks[i + 1])
                s_i = l + int(np.argmax(c[l:pk + 1]))
                e_i = pk + int(np.argmax(c[pk:r + 1]))
                if (c[s_i] - c[pk] >= 12.0) and (c[e_i] - c[pk] >= 12.0) and e_i > s_i + 2:
                    m = (fa >= grid[s_i]) & (fa < grid[e_i] + 1)
                    if m.sum() >= 4:
                        rows.append((resample_32(aa[m]), lbl, subj))
    return rows


def build_tabular_features(x: np.ndarray) -> np.ndarray:
    p1, p2, p3 = x[:, :11, :], x[:, 11:22, :], x[:, 22:, :]
    vel = np.diff(x, axis=1)
    pairs = [(0, 1), (2, 3), (4, 5), (9, 10), (12, 13)]
    return np.concatenate([
        x.mean(1), x.std(1), x.min(1), x.max(1), x.max(1) - x.min(1),
        p1.mean(1), p2.mean(1), p2.min(1), p2.max(1), p3.mean(1),
        np.abs(vel).mean(1), np.abs(vel).max(1),
        np.stack([np.abs(x[:, :, l] - x[:, :, r]).mean(1) for l, r in pairs], 1),
        np.stack([np.abs(x[:, :, l] - x[:, :, r]).max(1) for l, r in pairs], 1),
    ], axis=1).astype(np.float32)


def load_existing_drive_cache(ex: str) -> list[tuple[np.ndarray, int, str]]:
    cands = [
        SAVE_ROOT / f"pose_cache_{ex}_rehab1.npz",
        SAVE_ROOT / f"pose_cache_{ex}_rehab0.npz",
        SAVE_ROOT / f"pose_angles_cache_{ex}.npz",
    ]
    for p in cands:
        if p.exists():
            c = np.load(p, allow_pickle=True)
            x_key = "X_angles" if "X_angles" in c else "X_ang"
            rows = list(zip(c[x_key], c["y"], c["groups"]))
            print(f"Loaded existing Drive cache for [{ex}]: {len(rows)} reps from {p.name}")
            return rows
    return []


def train_and_export(ex: str, rows: list[tuple[np.ndarray, int, str]], out_root: Path):
    if not rows:
        print(f"SKIP [{ex}]: 0 reps available.")
        return
    X_ang = np.stack([r[0] for r in rows]).astype(np.float32)
    y = np.asarray([r[1] for r in rows], dtype=np.int32)
    g = np.asarray([r[2] for r in rows], dtype=object)
    X_tab = build_tabular_features(X_ang)
    n_subj = len(set(g))
    print(f"\n=== UPGRADED [{ex.upper()}] Reps: {len(y)} | Subjects: {n_subj} | Correct={(y==0).sum()} / Incorrect={(y==1).sum()} ===")
    if len(np.unique(y)) < 2:
        print(f"SKIP [{ex}]: need both Correct (0) and Incorrect (1) samples, got classes {np.unique(y).tolist()}.")
        return

    cv = StratifiedGroupKFold(n_splits=min(5, n_subj))
    rng = np.random.default_rng(SEED)
    names = ["Random Forest", "LightGBM", "XGBoost", "SVM (RBF)"]
    oof = {n: np.zeros(len(y), np.float32) for n in names}

    for fi, (tr, te) in enumerate(cv.split(X_tab, y, g), 1):
        ang_tr = np.concatenate([X_ang[tr], X_ang[tr][:, :, ANGLE_LR_SWAP], X_ang[tr] + rng.normal(0, 1.2, X_ang[tr].shape).astype(np.float32)], 0)
        y_tr = np.concatenate([y[tr], y[tr], y[tr]], 0)
        tab_tr = build_tabular_features(ang_tr)
        pos_r = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        clfs = {
            "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=SEED + fi, n_jobs=-1),
            "LightGBM": LGBMClassifier(n_estimators=200, max_depth=5, num_leaves=20, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8, class_weight="balanced", random_state=SEED + fi, verbose=-1),
            "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8, scale_pos_weight=pos_r, eval_metric="logloss", random_state=SEED + fi, n_jobs=-1),
            "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED + fi)),
        }
        for name, clf in clfs.items():
            clf.fit(tab_tr, y_tr)
            oof[name][te] = clf.predict_proba(X_tab[te])[:, 1]

    b_rows = []
    for name in names:
        preds = (oof[name] >= 0.5).astype(int)
        b_rows.append({
            "Model": name, "Family": "Tabular",
            "OOF Macro F1": round(float(f1_score(y, preds, average="macro")), 4),
            "Accuracy": round(float(accuracy_score(y, preds)), 4),
            "Balanced Acc": round(float(balanced_accuracy_score(y, preds)), 4),
            "ROC-AUC": round(float(roc_auc_score(y, oof[name])), 4),
        })
    board = pd.DataFrame(b_rows).sort_values("OOF Macro F1", ascending=False).reset_index(drop=True)
    print(board.to_string(index=False))

    rf_final = RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_final.fit(build_tabular_features(np.concatenate([X_ang, X_ang[:, :, ANGLE_LR_SWAP]], 0)), np.concatenate([y, y], 0))

    ex_dir = out_root / ex
    ex_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_final, ex_dir / "best_tabular_model.joblib", compress=3)
    board.to_csv(ex_dir / "model_leaderboard.csv", index=False)
    (ex_dir / "model_metadata.json").write_text(json.dumps({
        "exercise": ex, "winner_model": str(board.iloc[0]["Model"]), "best_tabular_model": "Random Forest",
        "threshold": 0.5, "sequence_length": SEQ_LEN, "features": FEATURES, "core_joints": CORE_JOINTS,
        "total_reps": int(len(y)), "unique_subjects": int(n_subj), "leaderboard": board.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")


# === Run Upgrade Pipeline ===
EXPORT_UP = SAVE_ROOT / "exported_upgraded"
EXPORT_UP.mkdir(parents=True, exist_ok=True)

uiprmd = load_uiprmd_rows()
for ex in ("squat", "lunge", "arm_abduction"):
    merged = load_existing_drive_cache(ex) + uiprmd[ex]
    train_and_export(ex, merged, EXPORT_UP)

row_rows = extract_mendeley_row()
train_and_export("row", row_rows, EXPORT_UP)

zip_out = SAVE_ROOT / "gym_model_bundle_upgraded.zip"
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in EXPORT_UP.rglob("*"):
        if p.is_file():
            zf.write(p, arcname=p.relative_to(EXPORT_UP).as_posix())
print(f"\nPACKAGED UPGRADED BUNDLES: {zip_out} ({zip_out.stat().st_size/1e6:.2f} MB)")
if files is not None:
    files.download(str(zip_out))
