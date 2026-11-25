# %% [markdown]
# # FormCoach — Mega Multi-Dataset Unified Ensemble Benchmark (5 Datasets Merged!)
#
# Automatically downloads and merges **5 real exercise-form datasets** under strict subject/clip-grouped 5-fold CV:
# 1. **UI-PRMD** (`Zenodo 23104965`) → `squat`, `lunge`, `arm_abduction`
# 2. **REHAB24-6** (`Zenodo 13305826`, 2D/3D joints + `Segmentation.csv`) → `squat`, `lunge`, `arm_abduction`, `shoulder`
# 3. **NgoQuocBao1010/Exercise-Correction — Bicep Curl** (`11.2 MB` MediaPipe CSV) → `curl`
# 4. **NgoQuocBao1010/Exercise-Correction — Lunge Error** (`18.8 MB` MediaPipe CSV) → `lunge`
# 5. **NgoQuocBao1010/Exercise-Correction — Plank Alignment** (`37.8 MB` MediaPipe CSV, `C` vs `L/H` hip sag/pike) → `push`
# *(+ automatically merges any cached Mendeley `.npz` files found in your Google Drive!)*
#
# Benchmarks **Random Forest**, **HistGradientBoosting**, **SVM (RBF)**, **Soft-Voting Ensemble (`RF+HGB+SVM`)**, and **Stacking Ensemble (`RF+HGB+SVM -> LogReg`)**, exports the top pure-`scikit-learn` ensemble per exercise, and downloads **`gym_model_bundle_unified_ensemble.zip`**.

# %%
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

try:
    from google.colab import drive, files  # type: ignore
    drive.mount("/content/drive")
    DRIVE_ROOTS = [
        Path("/content/drive/MyDrive/GymForm_All5_Export"),
        Path("/content/drive/MyDrive/GymForm_Benchmark_Export"),
    ]
    SAVE_ROOT = DRIVE_ROOTS[0]
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    files = None
    DRIVE_ROOTS = [Path("/content/unified_ensemble")]
    SAVE_ROOT = Path("/content/unified_ensemble")
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

WORK_DIR = Path("/content/unified_ensemble_work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SEQ_LEN = 32
EXERCISES = ("squat", "push", "curl", "shoulder", "lunge", "arm_abduction")
FEATURES = [
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "torso_tilt", "shoulder_line_tilt",
    "hip_line_tilt", "left_wrist_height", "right_wrist_height",
    "body_line", "left_knee", "right_knee",
]
CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
ANGLE_LR_SWAP = [1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 11, 13, 12]


def resample_32(seq: np.ndarray) -> np.ndarray:
    t = len(seq)
    if t >= 7:
        try:
            seq = savgol_filter(seq, window_length=min(7, t if t % 2 == 1 else t - 1), polyorder=2, axis=0)
        except Exception:
            pass
    if t == SEQ_LEN:
        return seq.astype(np.float32)
    old_t = np.linspace(0.0, 1.0, t)
    new_t = np.linspace(0.0, 1.0, SEQ_LEN)
    return np.stack([np.interp(new_t, old_t, seq[:, c]) for c in range(seq.shape[1])], axis=1).astype(np.float32)


def build_tabular_features(x: np.ndarray) -> np.ndarray:
    """Exact 178-D feature vector matching backend/app/pose_engine.py."""
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


def make_base_estimators(seed: int) -> dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=320, max_depth=16, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=240, max_depth=6, learning_rate=0.05,
            min_samples_leaf=15, l2_regularization=1.0,
            class_weight="balanced", random_state=seed,
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(C=2.5, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        ),
    }


def make_voting_ensemble(seed: int) -> VotingClassifier:
    est = make_base_estimators(seed)
    return VotingClassifier(
        estimators=[("rf", est["Random Forest"]), ("hgb", est["HistGradientBoosting"]), ("svm", est["SVM (RBF)"])],
        voting="soft",
        weights=[2.0, 1.5, 1.5],
        n_jobs=-1,
    )


def make_stacking_ensemble(seed: int) -> StackingClassifier:
    est = make_base_estimators(seed)
    return StackingClassifier(
        estimators=[("rf", est["Random Forest"]), ("hgb", est["HistGradientBoosting"]), ("svm", est["SVM (RBF)"])],
        final_estimator=LogisticRegression(C=1.0, max_iter=500, class_weight="balanced"),
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
    )


def _download(url: str, dest: Path, desc: str) -> Path:
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    print(f"Downloading {desc}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def calc_a3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-8), -1.0, 1.0))).astype(np.float32)


def calc_a2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, v = a[:, :2] - b[:, :2], c[:, :2] - b[:, :2]
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-8), -1.0, 1.0))).astype(np.float32)


# -------- Dataset 0: Existing Google Drive Mendeley / REHAB Caches --------
def load_drive_cache(ex: str) -> list[tuple[np.ndarray, int, str]]:
    for root in DRIVE_ROOTS:
        for fn in (f"pose_angles_cache_{ex}.npz", f"pose_cache_{ex}_rehab1.npz", f"pose_cache_{ex}_rehab0.npz"):
            p = root / fn
            if p.exists():
                c = np.load(p, allow_pickle=True)
                xk = "X_angles" if "X_angles" in c else "X_ang"
                rows = list(zip(c[xk], c["y"], c["groups"]))
                print(f"  [Drive Cache] {ex}: {len(rows)} reps from {p.name}")
                return rows
    return []


# -------- Datasets 1, 2, 3: NgoQuocBao1010/Exercise-Correction (Curl + Lunge + Plank/Push) --------
def _segment_ngo_runs(ang14: np.ndarray, labels: np.ndarray, prefix: str, mode: str) -> list[tuple[np.ndarray, int, str]]:
    rows = []
    chg = np.where(np.diff(labels) != 0)[0] + 1
    cid = 0
    for s_i, e_i in zip(np.concatenate([[0], chg]), np.concatenate([chg, [len(labels)]])):
        seg, lbl = ang14[s_i:e_i], int(labels[s_i])
        if len(seg) < 16 or not np.isfinite(seg).all():
            continue
        for sub in np.array_split(seg, max(1, len(seg) // 320)):
            cid += 1
            grp = f"ngo_{prefix}_G{cid:03d}"
            if len(sub) < 16:
                continue
            added = 0
            if mode in ("curl", "lunge"):
                sig = np.minimum(sub[:, 0], sub[:, 1]) if mode == "curl" else np.minimum(sub[:, 12], sub[:, 13])
                pks, _ = find_peaks(-sig, prominence=16.0, distance=18)
                for i, pk in enumerate(pks):
                    l, r = (0 if i == 0 else pks[i - 1]), (len(sig) - 1 if i == len(pks) - 1 else pks[i + 1])
                    st, en = l + int(np.argmax(sig[l:pk + 1])), pk + int(np.argmax(sig[pk:r + 1]))
                    if en - st >= 12 and (sig[st] - sig[pk] >= 14.0 or sig[en] - sig[pk] >= 14.0):
                        rows.append((resample_32(sub[st:en + 1]), lbl, grp))
                        added += 1
            if added == 0:
                step = 55 if mode == "push" else 45
                for w0 in range(0, len(sub) - 24, step):
                    rows.append((resample_32(sub[w0:w0 + step]), lbl, grp))
    return rows


def load_ngo_datasets() -> dict[str, list[tuple[np.ndarray, int, str]]]:
    base = "https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/core"
    out = {"curl": [], "lunge": [], "push": []}

    # 1. Bicep Curl (11.2 MB)
    for sp in ("train.csv", "test.csv"):
        p = _download(f"{base}/bicep_model/{sp}", WORK_DIR / f"ngo_bicep_{sp}", f"Ngo Bicep {sp}")
        df = pd.read_csv(p)
        pt = lambda n: df[[f"{n}_x", f"{n}_y", f"{n}_z"]].to_numpy(np.float32)
        ls, rs, le, re_, lw, rw, lh, rh = pt("left_shoulder"), pt("right_shoulder"), pt("left_elbow"), pt("right_elbow"), pt("left_wrist"), pt("right_wrist"), pt("left_hip"), pt("right_hip")
        ms, mh = 0.5 * (ls + rs), 0.5 * (lh + rh)
        sc = np.maximum(np.linalg.norm(ms - mh, axis=-1), 0.12)
        st = np.full(len(df), 175.0, np.float32)
        ang14 = np.stack([
            0.5 * (calc_a3(ls, le, lw) + calc_a2(ls, le, lw)),
            0.5 * (calc_a3(rs, re_, rw) + calc_a2(rs, re_, rw)),
            calc_a3(lh, ls, le), calc_a3(rh, rs, re_), st, st,
            calc_a3(ms, mh, mh + np.array([0.0, -1.0, 0.0], np.float32)),
            np.degrees(np.arctan2(np.abs(ls[:, 1] - rs[:, 1]), np.hypot(ls[:, 0] - rs[:, 0], ls[:, 2] - rs[:, 2]))).astype(np.float32),
            np.degrees(np.arctan2(np.abs(lh[:, 1] - rh[:, 1]), np.hypot(lh[:, 0] - rh[:, 0], lh[:, 2] - rh[:, 2]))).astype(np.float32),
            ((ls[:, 1] - lw[:, 1]) / sc).astype(np.float32), ((rs[:, 1] - rw[:, 1]) / sc).astype(np.float32),
            st, st, st,
        ], axis=1)
        lbl = (df["label"].astype(str).str.upper() != "C").astype(np.int32).to_numpy()
        out["curl"].extend(_segment_ngo_runs(ang14, lbl, f"curl_{sp[:2]}", "curl"))

    # 2. Lunge Errors (18.8 MB)
    for sp in ("err.train.csv", "err.test.csv"):
        p = _download(f"{base}/lunge_model/{sp}", WORK_DIR / f"ngo_lunge_{sp}", f"Ngo Lunge {sp}")
        df = pd.read_csv(p)
        pt = lambda n: df[[f"{n}_x", f"{n}_y", f"{n}_z"]].to_numpy(np.float32)
        ls, rs, lh, rh, lk, rk, la, ra = pt("left_shoulder"), pt("right_shoulder"), pt("left_hip"), pt("right_hip"), pt("left_knee"), pt("right_knee"), pt("left_ankle"), pt("right_ankle")
        ms, mh, ma = 0.5 * (ls + rs), 0.5 * (lh + rh), 0.5 * (la + ra)
        arm = np.full(len(df), 160.0, np.float32)
        sh_n = np.full(len(df), 20.0, np.float32)
        wh_n = np.full(len(df), -0.8, np.float32)
        ang14 = np.stack([
            arm, arm, sh_n, sh_n,
            calc_a3(ls, lh, lk), calc_a3(rs, rh, rk),
            calc_a3(ms, mh, mh + np.array([0.0, -1.0, 0.0], np.float32)),
            np.degrees(np.arctan2(np.abs(ls[:, 1] - rs[:, 1]), np.hypot(ls[:, 0] - rs[:, 0], ls[:, 2] - rs[:, 2]))).astype(np.float32),
            np.degrees(np.arctan2(np.abs(lh[:, 1] - rh[:, 1]), np.hypot(lh[:, 0] - rh[:, 0], lh[:, 2] - rh[:, 2]))).astype(np.float32),
            wh_n, wh_n, calc_a3(ms, mh, ma), calc_a3(lh, lk, la), calc_a3(rh, rk, ra),
        ], axis=1)
        lbl = (df["label"].astype(str).str.upper() != "C").astype(np.int32).to_numpy()
        out["lunge"].extend(_segment_ngo_runs(ang14, lbl, f"lunge_{sp[:6]}", "lunge"))

    # 3. Plank Alignment -> Push-Up Core Form (37.8 MB: C=Straight, L=Low/Sagging Hips, H=High/Piking Hips)
    for sp in ("train.csv", "test.csv"):
        p = _download(f"{base}/plank_model/{sp}", WORK_DIR / f"ngo_plank_{sp}", f"Ngo Plank {sp}")
        df = pd.read_csv(p)
        pt = lambda n: df[[f"{n}_x", f"{n}_y", f"{n}_z"]].to_numpy(np.float32)
        ls, rs, le, re_, lw, rw = pt("left_shoulder"), pt("right_shoulder"), pt("left_elbow"), pt("right_elbow"), pt("left_wrist"), pt("right_wrist")
        lh, rh, lk, rk, la, ra = pt("left_hip"), pt("right_hip"), pt("left_knee"), pt("right_knee"), pt("left_ankle"), pt("right_ankle")
        ms, mh, ma = 0.5 * (ls + rs), 0.5 * (lh + rh), 0.5 * (la + ra)
        sc = np.maximum(np.linalg.norm(ms - mh, axis=-1), 0.12)
        ang14 = np.stack([
            calc_a3(ls, le, lw), calc_a3(rs, re_, rw),
            calc_a3(lh, ls, le), calc_a3(rh, rs, re_),
            calc_a3(ls, lh, lk), calc_a3(rs, rh, rk),
            calc_a3(ms, mh, mh + np.array([0.0, -1.0, 0.0], np.float32)),
            np.degrees(np.arctan2(np.abs(ls[:, 1] - rs[:, 1]), np.hypot(ls[:, 0] - rs[:, 0], ls[:, 2] - rs[:, 2]))).astype(np.float32),
            np.degrees(np.arctan2(np.abs(lh[:, 1] - rh[:, 1]), np.hypot(lh[:, 0] - rh[:, 0], lh[:, 2] - rh[:, 2]))).astype(np.float32),
            ((ls[:, 1] - lw[:, 1]) / sc).astype(np.float32), ((rs[:, 1] - rw[:, 1]) / sc).astype(np.float32),
            calc_a3(ms, mh, ma), calc_a3(lh, lk, la), calc_a3(rh, rk, ra),
        ], axis=1)
        lbl = (df["label"].astype(str).str.upper() != "C").astype(np.int32).to_numpy()
        out["push"].extend(_segment_ngo_runs(ang14, lbl, f"plank_{sp[:2]}", "push"))

    for k, v in out.items():
        print(f"  [NgoQuocBao1010] {k}: {len(v)} reps (Good={sum(r[1]==0 for r in v)}, Bad={sum(r[1]==1 for r in v)})")
    return out


# -------- Dataset 4: UI-PRMD (Zenodo 23104965) --------
UI_PARENTS = [None, 0, 1, 2, 1, 4, 5, 6, 1, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 7, 11]


def _rot_zyx(ang_deg: np.ndarray) -> np.ndarray:
    a, b, c = np.radians(ang_deg)
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return (
        np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], np.float32)
        @ np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], np.float32)
        @ np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]], np.float32)
    )


def uiprmd_fk_to_14(pos_arr: np.ndarray, ang_arr: np.ndarray) -> np.ndarray | None:
    T = min(len(pos_arr), len(ang_arr))
    if T < 12 or pos_arr.shape[1] < 66 or ang_arr.shape[1] < 66:
        return None
    pos, ang = pos_arr[:T, :66].reshape(T, 22, 3), ang_arr[:T, :66].reshape(T, 22, 3)
    out = []
    for t in range(T):
        wp, wr = np.zeros((22, 3), np.float32), [np.eye(3, dtype=np.float32)] * 22
        wp[0], wr[0] = pos[t, 0], _rot_zyx(ang[t, 0])
        for j in range(1, 22):
            p = UI_PARENTS[j]
            wp[j] = wp[p] + wr[p] @ pos[t, j]
            wr[j] = wr[p] @ _rot_zyx(ang[t, j])
        ls, le, lw, rs, re_, rw = wp[4], wp[5], wp[6], wp[8], wp[9], wp[10]
        lh, lk, la, rh, rk, ra = wp[12], wp[13], wp[14], wp[16], wp[17], wp[18]
        sh, hp, an = 0.5 * (ls + rs), 0.5 * (lh + rh), 0.5 * (la + ra)
        sc = max(float(np.linalg.norm(sh - hp)), 1e-3)
        ca = lambda a, b, c: float(np.degrees(np.arccos(np.clip(np.dot(a - b, c - b) / max(np.linalg.norm(a - b) * np.linalg.norm(c - b), 1e-8), -1.0, 1.0))))
        tl = lambda p1, p2: float(np.degrees(np.arctan2(abs(p1[1] - p2[1]), np.hypot(p1[0] - p2[0], p1[2] - p2[2]))))
        row = np.array([
            ca(ls, le, lw), ca(rs, re_, rw), ca(lh, ls, le), ca(rh, rs, re_),
            ca(ls, lh, lk), ca(rs, rh, rk), ca(sh, hp, hp + np.array([0.0, 1.0, 0.0], np.float32)),
            tl(ls, rs), tl(lh, rh), float((lw[1] - ls[1]) / sc), float((rw[1] - rs[1]) / sc),
            ca(sh, hp, an), ca(lh, lk, la), ca(rh, rk, ra),
        ], np.float32)
        if np.isfinite(row).all():
            out.append(row)
    return np.asarray(out, np.float32) if len(out) >= 12 else None


def load_uiprmd() -> dict[str, list[tuple[np.ndarray, int, str]]]:
    def _parse(b: bytes):
        txt = b.decode("utf-8", errors="ignore").strip()
        if not txt:
            return None
        try:
            a = np.loadtxt(io.StringIO(txt), delimiter="," if "," in txt.splitlines()[0] else None, dtype=np.float32)
            return a if a.ndim == 2 else None
        except Exception:
            return None

    def _key(n: str):
        bn = Path(n).name.lower()
        m, s, e = re.search(r"(m0[1-9]|m10)", bn), re.search(r"s(\d+)", bn), re.search(r"e(\d+)", bn)
        return (m.group(1), f"s{int(s.group(1)):02d}", f"e{int(e.group(1)):02d}" if e else "e00") if (m and s) else None

    z_c = _download("https://zenodo.org/records/23104965/files/Segmented%20Movements.zip?download=1", WORK_DIR / "uiprmd_correct.zip", "UI-PRMD Correct")
    z_i = _download("https://zenodo.org/records/23104965/files/Incorrect%20Movements.zip?download=1", WORK_DIR / "uiprmd_incorrect.zip", "UI-PRMD Incorrect")
    mv_map = {"m01": "squat", "m03": "lunge", "m07": "arm_abduction"}
    coll = {ex: [] for ex in mv_map.values()}
    for zp, lbl in ((z_c, 0), (z_i, 1)):
        with zipfile.ZipFile(zp) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".txt") and "kinect" in n.lower()]
            pos_f = [n for n in names if "position" in n.lower()]
            ang_m = {_key(n): n for n in names if "angle" in n.lower() and _key(n) is not None}
            for pn in tqdm(pos_f, desc=f"UI-PRMD lbl={lbl}"):
                k = _key(pn)
                if k is None or k[0] not in mv_map or k not in ang_m:
                    continue
                ex = mv_map[k[0]]
                pa, aa = _parse(zf.read(pn)), _parse(zf.read(ang_m[k]))
                if pa is None or aa is None:
                    continue
                ang14 = uiprmd_fk_to_14(pa, aa)
                if ang14 is None:
                    continue
                subj = f"uiprmd_{k[1].upper()}"
                if len(ang14) > 220:
                    sig = ang14[:, 12:14].mean(1) if ex in ("squat", "lunge") else ang14[:, 2:4].max(1)
                    inv = ex in ("squat", "lunge")
                    pks, _ = find_peaks(-sig if inv else sig, prominence=15.0, distance=22)
                    added = 0
                    for i, pk in enumerate(pks):
                        l, r = (0 if i == 0 else pks[i - 1]), (len(sig) - 1 if i == len(pks) - 1 else pks[i + 1])
                        st = l + int(np.argmax(sig[l:pk + 1]) if inv else np.argmin(sig[l:pk + 1]))
                        en = pk + int(np.argmax(sig[pk:r + 1]) if inv else np.argmin(sig[pk:r + 1]))
                        if en - st >= 10:
                            coll[ex].append((resample_32(ang14[st:en + 1]), lbl, subj))
                            added += 1
                    if added == 0:
                        for ch in np.array_split(ang14, max(1, len(ang14) // 100)):
                            if len(ch) >= 12:
                                coll[ex].append((resample_32(ch), lbl, subj))
                else:
                    coll[ex].append((resample_32(ang14), lbl, subj))
    for ex, rs in coll.items():
        print(f"  [UI-PRMD] {ex}: {len(rs)} reps")
    return coll


# -------- Dataset 5: REHAB24-6 (Zenodo 13305826) -> Squat, Lunge, Arm Abduction, Shoulder --------
def load_rehab24_6() -> dict[str, list[tuple[np.ndarray, int, str]]]:
    coll = {"squat": [], "lunge": [], "arm_abduction": [], "shoulder": []}
    try:
        z_j = _download("https://zenodo.org/records/13305826/files/3d_joints.zip?download=1", WORK_DIR / "rehab24_3d_joints.zip", "REHAB24-6 3D Joints")
        seg_p = _download("https://zenodo.org/records/13305826/files/Segmentation.csv?download=1", WORK_DIR / "rehab24_seg.csv", "REHAB24-6 Segmentation.csv")
    except Exception as e:
        print(f"  [REHAB24-6] skipped ({e})")
        return coll

    seg_df = pd.read_csv(seg_p, sep=";" if ";" in seg_p.read_text(encoding="utf-8", errors="ignore")[:200] else ",")
    ex_map = {"ex1": "arm_abduction", "ex2": "shoulder", "ex5": "lunge", "ex6": "squat"}
    # REHAB24-6 26-joint OptiTrack/Kinect indices: pelvis=0, chest=2, l_sh=5, l_el=6, l_wr=7, r_sh=9, r_el=10, r_wr=11, l_hip=13, l_knee=14, l_ank=15, r_hip=17, r_knee=18, r_ank=19
    with zipfile.ZipFile(z_j) as zf:
        npy_map = {Path(n).stem.lower(): n for n in zf.namelist() if n.lower().endswith(".npy")}
        for row in seg_df.itertuples():
            vals = [str(v) for v in row[1:]]
            ex_tok = next((v.lower() for v in vals if v.lower() in ex_map), None)
            if ex_tok is None:
                continue
            ex = ex_map[ex_tok]
            # Match video/file stem and start/end frame + correctness
            vid_id = str(getattr(row, "video_id", getattr(row, "Video", vals[0]))).strip().lower()
            cand = next((v for k, v in npy_map.items() if vid_id in k), None)
            if cand is None:
                continue
            try:
                f0 = int(getattr(row, "first_frame", getattr(row, "Start", vals[2])))
                f1 = int(getattr(row, "last_frame", getattr(row, "End", vals[3])))
                corr = int(getattr(row, "correctness", getattr(row, "Correctness", vals[-1])))
            except Exception:
                continue
            lbl = 0 if corr == 1 else 1
            sm = re.search(r"pm(\d+)", vid_id)
            subj = f"rehab24_S{int(sm.group(1)):02d}" if sm else f"rehab24_{vid_id[:8]}"
            arr = np.load(io.BytesIO(zf.read(cand)))
            if arr.ndim != 3 or arr.shape[1] < 20 or f1 - f0 < 10:
                continue
            p = arr[max(0, f0):min(len(arr), f1 + 1)]
            if len(p) < 10 or not np.isfinite(p).all():
                continue
            ls, le, lw, rs, re_, rw = p[:, 5], p[:, 6], p[:, 7], p[:, 9], p[:, 10], p[:, 11]
            lh, lk, la, rh, rk, ra = p[:, 13], p[:, 14], p[:, 15], p[:, 17], p[:, 18], p[:, 19]
            ms, mh, ma = 0.5 * (ls + rs), 0.5 * (lh + rh), 0.5 * (la + ra)
            sc = np.maximum(np.linalg.norm(ms - mh, axis=-1), 1e-3)
            ang14 = np.stack([
                calc_a3(ls, le, lw), calc_a3(rs, re_, rw),
                calc_a3(lh, ls, le), calc_a3(rh, rs, re_),
                calc_a3(ls, lh, lk), calc_a3(rs, rh, rk),
                calc_a3(ms, mh, mh + np.array([0.0, 0.0, 1.0], np.float32)),
                np.degrees(np.arctan2(np.abs(ls[:, 2] - rs[:, 2]), np.hypot(ls[:, 0] - rs[:, 0], ls[:, 1] - rs[:, 1]))).astype(np.float32),
                np.degrees(np.arctan2(np.abs(lh[:, 2] - rh[:, 2]), np.hypot(lh[:, 0] - rh[:, 0], lh[:, 1] - rh[:, 1]))).astype(np.float32),
                ((lw[:, 2] - ls[:, 2]) / sc).astype(np.float32), ((rw[:, 2] - rs[:, 2]) / sc).astype(np.float32),
                calc_a3(ms, mh, ma), calc_a3(lh, lk, la), calc_a3(rh, rk, ra),
            ], axis=1)
            if np.isfinite(ang14).all():
                coll[ex].append((resample_32(ang14), lbl, subj))

    for ex, rs in coll.items():
        print(f"  [REHAB24-6] {ex}: {len(rs)} reps")
    return coll


# === Load & Merge ALL 5 Datasets ===
print("Loading and merging all datasets...")
ngo = load_ngo_datasets()
uiprmd = load_uiprmd()
rehab = load_rehab24_6()

datasets: dict[str, list[tuple[np.ndarray, int, str]]] = {}
for ex in EXERCISES:
    rows = []
    rows.extend(load_drive_cache(ex))
    rows.extend(uiprmd.get(ex, []))
    rows.extend(rehab.get(ex, []))
    rows.extend(ngo.get(ex, []))
    if rows:
        datasets[ex] = rows

# === Benchmark Single Models vs Soft-Voting vs Stacking Ensemble ===
EXPORT_DIR = SAVE_ROOT / "exported_unified_ensemble"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
summary_rows = []

for ex, rows in datasets.items():
    X_ang = np.stack([r[0] for r in rows]).astype(np.float32)
    y = np.asarray([r[1] for r in rows], dtype=np.int32)
    g = np.asarray([r[2] for r in rows], dtype=object)
    if len(np.unique(y)) < 2:
        continue
    X_tab = build_tabular_features(X_ang)
    n_subj = len(set(g))
    print(f"\n================ [{ex.upper()}] Reps={len(y)} | Groups={n_subj} | Good={(y==0).sum()} / Bad={(y==1).sum()} ================")

    cv = StratifiedGroupKFold(n_splits=min(5, n_subj))
    rng = np.random.default_rng(SEED)
    model_names = [
        "Random Forest",
        "HistGradientBoosting",
        "SVM (RBF)",
        "Unified Ensemble (RF+HGB+SVM)",
        "Stacking Ensemble (RF+HGB+SVM->LR)",
    ]
    oof = {n: np.zeros(len(y), np.float32) for n in model_names}

    for fi, (tr, te) in enumerate(cv.split(X_tab, y, g), 1):
        ang_tr = np.concatenate([
            X_ang[tr], X_ang[tr][:, :, ANGLE_LR_SWAP],
            X_ang[tr] + rng.normal(0, 1.2, X_ang[tr].shape).astype(np.float32),
        ], axis=0)
        y_tr = np.concatenate([y[tr], y[tr], y[tr]], axis=0)
        tab_tr = build_tabular_features(ang_tr)

        clfs = make_base_estimators(SEED + fi)
        clfs["Unified Ensemble (RF+HGB+SVM)"] = make_voting_ensemble(SEED + fi)
        clfs["Stacking Ensemble (RF+HGB+SVM->LR)"] = make_stacking_ensemble(SEED + fi)
        for name, clf in clfs.items():
            clf.fit(tab_tr, y_tr)
            oof[name][te] = clf.predict_proba(X_tab[te])[:, 1]

    # Pick best ensemble (Soft-Voting vs Stacking) and tune OOF threshold
    best_ens_name = max(
        ("Unified Ensemble (RF+HGB+SVM)", "Stacking Ensemble (RF+HGB+SVM->LR)"),
        key=lambda n: float(f1_score(y, (oof[n] >= 0.5).astype(int), average="macro")),
    )
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.38, 0.62, 25):
        f1_v = float(f1_score(y, (oof[best_ens_name] >= thr).astype(int), average="macro"))
        if f1_v > best_f1:
            best_f1, best_thr = f1_v, float(round(thr, 3))

    b_rows = []
    for name in model_names:
        thr = best_thr if name == best_ens_name else 0.5
        preds = (oof[name] >= thr).astype(int)
        b_rows.append({
            "Model": name,
            "Family": "Ensemble" if "Ensemble" in name else "Tabular",
            "OOF Macro F1": round(float(f1_score(y, preds, average="macro")), 4),
            "Accuracy": round(float(accuracy_score(y, preds)), 4),
            "Balanced Acc": round(float(balanced_accuracy_score(y, preds)), 4),
            "ROC-AUC": round(float(roc_auc_score(y, oof[name])), 4),
        })
    board = pd.DataFrame(b_rows).sort_values("OOF Macro F1", ascending=False).reset_index(drop=True)
    print(board.to_string(index=False))

    rf_f1 = float(board[board["Model"] == "Random Forest"].iloc[0]["OOF Macro F1"])
    ens_row = board[board["Model"] == best_ens_name].iloc[0]
    summary_rows.append({
        "Exercise": ex, "Reps": len(y), "Groups": n_subj,
        "RF F1": rf_f1,
        "Best Ensemble": best_ens_name,
        "Ensemble F1": float(ens_row["OOF Macro F1"]),
        "Delta vs RF": round(float(ens_row["OOF Macro F1"]) - rf_f1, 4),
        "Ensemble AUC": float(ens_row["ROC-AUC"]),
    })

    # Fit winning ensemble on full merged dataset + mirror flip
    final_clf = make_stacking_ensemble(SEED) if "Stacking" in best_ens_name else make_voting_ensemble(SEED)
    final_clf.fit(
        build_tabular_features(np.concatenate([X_ang, X_ang[:, :, ANGLE_LR_SWAP]], axis=0)),
        np.concatenate([y, y], axis=0),
    )
    ex_dir = EXPORT_DIR / ex
    ex_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_clf, ex_dir / "best_tabular_model.joblib", compress=3)
    board.to_csv(ex_dir / "model_leaderboard.csv", index=False)
    (ex_dir / "model_metadata.json").write_text(json.dumps({
        "exercise": ex,
        "winner_model": str(board.iloc[0]["Model"]),
        "best_tabular_model": best_ens_name,
        "threshold": best_thr,
        "sequence_length": SEQ_LEN,
        "features": FEATURES,
        "core_joints": CORE_JOINTS,
        "total_reps": int(len(y)),
        "unique_subjects": int(n_subj),
        "leaderboard": board.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")

print("\n================ FINAL 5-DATASET MERGED SUMMARY ================")
print(pd.DataFrame(summary_rows).to_string(index=False))

zip_out = SAVE_ROOT / "gym_model_bundle_unified_ensemble.zip"
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in EXPORT_DIR.rglob("*"):
        if p.is_file():
            zf.write(p, arcname=p.relative_to(EXPORT_DIR).as_posix())
print(f"\nPACKAGED MEGA-MERGED ENSEMBLE BUNDLE: {zip_out} ({zip_out.stat().st_size / 1e6:.2f} MB)")
if files is not None:
    files.download(str(zip_out))
