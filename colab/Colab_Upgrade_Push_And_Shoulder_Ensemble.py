# %% [markdown]
# # FormCoach — Upgrade Push-Up & Shoulder Press to Unified Ensemble (`RF + HGB + SVM`) + Custom Webcam Clips
#
# Upgrades your remaining 2 single-Random-Forest exercises (`push` and `shoulder`) to the **Unified Soft-Voting / Stacking Ensemble (`Random Forest + HistGradientBoosting + SVM-RBF`)** with:
# 1. **Savitzky-Golay angle smoothing** (matches live inference in `pose_engine.py`)
# 2. **Mendeley Multi-View (`kgbb3yn47p` v3)** — reuses Drive cache if present, else auto-downloads & caches in Google Drive
# 3. **Optional Custom Team Clips (Domain Matching!):** Drop any short webcam/phone videos into:
#    - `MyDrive/GymForm_Custom_Clips/push/good/` & `.../push/bad/`
#    - `MyDrive/GymForm_Custom_Clips/shoulder/good/` & `.../shoulder/bad/`
#    *(and they are automatically extracted & mixed into training!)*
# 4. Exports **`gym_model_bundle_push_shoulder_ensemble.zip`** — drop-in compatible with `backend/models/`.

# %%
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mediapipe"])

# %%
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
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
    CUSTOM_ROOT = Path("/content/drive/MyDrive/GymForm_Custom_Clips")
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    CUSTOM_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    files = None
    DRIVE_ROOTS = [Path("/content/push_shoulder_up")]
    SAVE_ROOT = Path("/content/push_shoulder_up")
    CUSTOM_ROOT = Path("/content/custom_clips")
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    CUSTOM_ROOT.mkdir(parents=True, exist_ok=True)

WORK_DIR = Path("/content/push_shoulder_work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SEQ_LEN = 32
FRAME_STEP = 3
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
            n_estimators=350, max_depth=16, min_samples_leaf=2,
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
        weights=[1.8, 1.5, 2.0],  # slightly higher SVM weight (strong on push & shoulder!)
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


def get_pose_landmarker():
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
    return mp.tasks.vision.PoseLandmarker.create_from_options(opts)


def extract_reps_from_video(video_path: Path, ex: str, lbl: int, subj: str, det) -> list[tuple[np.ndarray, int, str]]:
    def calc_a(a, b, c):
        u, v = a - b, c - b
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-8), -1.0, 1.0))))

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
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
                        frs.append(idx)
                        angs.append(a)
        idx += 1
    cap.release()
    if len(frs) < 6:
        return []

    fa, aa = np.asarray(frs), np.asarray(angs)
    sig = aa[:, :2].min(1) if ex == "push" else aa[:, 2:4].max(1)
    inv = (ex == "push")
    grid = np.arange(fa[0], fa[-1] + 1, FRAME_STEP)
    c = np.convolve(np.interp(grid, fa, sig), [0.25, 0.5, 0.25], "same") if len(grid) >= 5 else np.interp(grid, fa, sig)
    pks, _ = find_peaks(-c if inv else c, prominence=14.0, distance=max(2, int(0.65 * fps / FRAME_STEP)))
    rows = []
    for i, pk in enumerate(pks):
        l, r = (0 if i == 0 else pks[i - 1]), (len(c) - 1 if i == len(pks) - 1 else pks[i + 1])
        s_i = l + int(np.argmax(c[l:pk + 1]) if inv else np.argmin(c[l:pk + 1]))
        e_i = pk + int(np.argmax(c[pk:r + 1]) if inv else np.argmin(c[pk:r + 1]))
        if abs(c[s_i] - c[pk]) >= 12.0 and abs(c[e_i] - c[pk]) >= 12.0 and e_i > s_i + 2:
            m = (fa >= grid[s_i]) & (fa < grid[e_i] + 1)
            if m.sum() >= 4:
                rows.append((resample_32(aa[m]), lbl, subj))
    return rows


def load_or_extract_mendeley(ex: str) -> list[tuple[np.ndarray, int, str]]:
    for root in DRIVE_ROOTS:
        for fn in (f"pose_angles_cache_{ex}.npz", f"pose_cache_{ex}_rehab1.npz", f"pose_cache_{ex}_rehab0.npz"):
            p = root / fn
            if p.exists():
                c = np.load(p, allow_pickle=True)
                xk = "X_angles" if "X_angles" in c else "X_ang"
                rows = [(resample_32(x), int(y), str(g)) for x, y, g in zip(c[xk], c["y"], c["groups"])]
                print(f"Loaded Drive cache for [{ex}]: {len(rows)} reps from {p.name}")
                return rows

    # Auto-download Mendeley Exercise-wise ZIP once and cache in Drive
    cache_dir = SAVE_ROOT / "dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zpath = cache_dir / "Dataset_Exercise_wise.zip"
    if not (zpath.exists() and zipfile.is_zipfile(zpath)):
        print("Downloading Mendeley Exercise-wise ZIP (cached to Drive for future runs)...")
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
        subprocess.run(["curl", "-fsSL", "-o", str(zpath), url], check=True)

    kw = ("push",) if ex == "push" else ("shoulder", "press", "ohp")
    rows = []
    with zipfile.ZipFile(zpath) as zf, tempfile.TemporaryDirectory() as td, get_pose_landmarker() as det:
        mp4s = [m for m in zf.namelist() if m.lower().endswith((".mp4", ".mov", ".avi")) and any(k in m.lower() for k in kw)]
        seen = set()
        for mem in tqdm(mp4s, desc=f"Extracting Mendeley [{ex}]"):
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
            rows.extend(extract_reps_from_video(lv, ex, lbl, subj, det))

    if rows:
        np.savez_compressed(
            SAVE_ROOT / f"pose_angles_cache_{ex}.npz",
            X_angles=np.stack([r[0] for r in rows]),
            y=np.asarray([r[1] for r in rows], np.int32),
            groups=np.asarray([r[2] for r in rows], dtype=object),
        )
    print(f"Extracted & cached [{ex}]: {len(rows)} reps")
    return rows


def load_custom_team_clips(ex: str) -> list[tuple[np.ndarray, int, str]]:
    """Optional: loads your team's real webcam/phone clips from MyDrive/GymForm_Custom_Clips/<ex>/{good,bad}/."""
    rows = []
    base = CUSTOM_ROOT / ex
    for folder, lbl in (("good", 0), ("bad", 1)):
        d = base / folder
        d.mkdir(parents=True, exist_ok=True)
        clips = [p for p in d.glob("*") if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm")]
        if clips:
            with get_pose_landmarker() as det:
                for p in clips:
                    rows.extend(extract_reps_from_video(p, ex, lbl, f"team_{p.stem}", det))
    if rows:
        print(f"Loaded {len(rows)} custom team webcam reps for [{ex}]!")
    return rows


EXPORT_DIR = SAVE_ROOT / "exported_push_shoulder"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
summary = []

for ex in ("push", "shoulder"):
    rows = load_or_extract_mendeley(ex) + load_custom_team_clips(ex)
    if not rows:
        continue
    X_ang = np.stack([r[0] for r in rows]).astype(np.float32)
    y = np.asarray([r[1] for r in rows], dtype=np.int32)
    g = np.asarray([r[2] for r in rows], dtype=object)
    X_tab = build_tabular_features(X_ang)
    n_subj = len(set(g))
    print(f"\n================ [{ex.upper()}] Reps={len(y)} | Subjects={n_subj} | Good={(y==0).sum()} / Bad={(y==1).sum()} ================")

    cv = StratifiedGroupKFold(n_splits=min(5, n_subj))
    rng = np.random.default_rng(SEED)
    names = ["Random Forest", "HistGradientBoosting", "SVM (RBF)", "Unified Ensemble (RF+HGB+SVM)", "Stacking Ensemble (RF+HGB+SVM->LR)"]
    oof = {n: np.zeros(len(y), np.float32) for n in names}

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

    best_ens = max(
        ("Unified Ensemble (RF+HGB+SVM)", "Stacking Ensemble (RF+HGB+SVM->LR)"),
        key=lambda n: float(f1_score(y, (oof[n] >= 0.5).astype(int), average="macro")),
    )
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.38, 0.62, 25):
        fv = float(f1_score(y, (oof[best_ens] >= thr).astype(int), average="macro"))
        if fv > best_f1:
            best_f1, best_thr = fv, float(round(thr, 3))

    b_rows = []
    for name in names:
        thr = best_thr if name == best_ens else 0.5
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
    er = board[board["Model"] == best_ens].iloc[0]
    summary.append({"Exercise": ex, "Reps": len(y), "Subjects": n_subj, "RF F1": rf_f1, "Best Ensemble": best_ens, "Ensemble F1": float(er["OOF Macro F1"]), "Delta": round(float(er["OOF Macro F1"]) - rf_f1, 4), "AUC": float(er["ROC-AUC"])})

    final_clf = make_stacking_ensemble(SEED) if "Stacking" in best_ens else make_voting_ensemble(SEED)
    final_clf.fit(build_tabular_features(np.concatenate([X_ang, X_ang[:, :, ANGLE_LR_SWAP]], 0)), np.concatenate([y, y], 0))
    ex_dir = EXPORT_DIR / ex
    ex_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_clf, ex_dir / "best_tabular_model.joblib", compress=3)
    board.to_csv(ex_dir / "model_leaderboard.csv", index=False)
    (ex_dir / "model_metadata.json").write_text(json.dumps({
        "exercise": ex, "winner_model": str(board.iloc[0]["Model"]), "best_tabular_model": best_ens,
        "threshold": best_thr, "sequence_length": SEQ_LEN, "features": FEATURES, "core_joints": CORE_JOINTS,
        "total_reps": int(len(y)), "unique_subjects": int(n_subj), "leaderboard": board.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")

print("\n================ PUSH & SHOULDER ENSEMBLE SUMMARY ================")
print(pd.DataFrame(summary).to_string(index=False))

zip_out = SAVE_ROOT / "gym_model_bundle_push_shoulder_ensemble.zip"
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in EXPORT_DIR.rglob("*"):
        if p.is_file():
            zf.write(p, arcname=p.relative_to(EXPORT_DIR).as_posix())
print(f"\nPACKAGED: {zip_out} ({zip_out.stat().st_size / 1e6:.2f} MB)")
if files is not None:
    files.download(str(zip_out))
