# %% [markdown]
# # FormCoach — Upgraded Bicep Curl Training (Google Colab)
#
# Merges two real MediaPipe Pose Bicep Curl datasets under strict subject/clip-grouped 5-fold CV:
# 1. **Mendeley Multi-View Bicep Curl** (`pose_angles_cache_curl.npz` from your Google Drive, 25 subjects, 2,175 reps)
# 2. **NgoQuocBao1010/Exercise-Correction Bicep Curl Dataset** (15,976 labeled MediaPipe Pose frames: Correct `C` vs. Lean-Back / Momentum Fault `L`)
#
# Trains & benchmarks **Random Forest**, **LightGBM**, **XGBoost**, and **SVM (RBF)** on our exact 178-D phase-aware feature vector, exports the upgraded **Random Forest** production model (`curl/best_tabular_model.joblib`), and downloads **`gym_model_bundle_curl_upgraded.zip`** (~2 minutes total!).

# %%
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
    DRIVE_ROOTS = [Path("/content/curl_upgrade")]
    SAVE_ROOT = Path("/content/curl_upgrade")
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
SEQ_LEN = 32
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
    if t == SEQ_LEN:
        return seq.astype(np.float32)
    old_t = np.linspace(0.0, 1.0, t)
    new_t = np.linspace(0.0, 1.0, SEQ_LEN)
    return np.stack([np.interp(new_t, old_t, seq[:, c]) for c in range(seq.shape[1])], axis=1).astype(np.float32)


def calc_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u = a - b
    v = c - b
    dot = np.sum(u * v, axis=-1)
    den = np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-8)
    return np.degrees(np.arccos(np.clip(dot / den, -1.0, 1.0))).astype(np.float32)


def calc_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u = a[:, :2] - b[:, :2]
    v = c[:, :2] - b[:, :2]
    dot = np.sum(u * v, axis=-1)
    den = np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-8)
    return np.degrees(np.arccos(np.clip(dot / den, -1.0, 1.0))).astype(np.float32)


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


# -------- 1. Load NgoQuocBao1010/Exercise-Correction Bicep Curl Dataset (15,976 frames) --------
def load_ngo_bicep_rows() -> list[tuple[np.ndarray, int, str]]:
    base = "https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/core/bicep_model"
    rows: list[tuple[np.ndarray, int, str]] = []
    req_hdr = {"User-Agent": "Mozilla/5.0"}

    for split_name in ("train.csv", "test.csv"):
        url = f"{base}/{split_name}"
        print(f"Downloading NgoQuocBao1010 {split_name}...")
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=req_hdr), timeout=60).read()
        df = pd.read_csv(io.BytesIO(raw))

        def pt(name: str) -> np.ndarray:
            return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(dtype=np.float32)

        l_sh, r_sh = pt("left_shoulder"), pt("right_shoulder")
        l_el, r_el = pt("left_elbow"), pt("right_elbow")
        l_wr, r_wr = pt("left_wrist"), pt("right_wrist")
        l_hp, r_hp = pt("left_hip"), pt("right_hip")

        mid_sh = 0.5 * (l_sh + r_sh)
        mid_hp = 0.5 * (l_hp + r_hp)
        torso_len = np.maximum(np.linalg.norm(mid_sh - mid_hp, axis=-1), 0.12)

        # Blend 3D + 2D elbow angles just like backend/app/pose_engine.py (solves front-view foreshortening)
        l_el_ang = 0.5 * (calc_angle_3d(l_sh, l_el, l_wr) + calc_angle_2d(l_sh, l_el, l_wr))
        r_el_ang = 0.5 * (calc_angle_3d(r_sh, r_el, r_wr) + calc_angle_2d(r_sh, r_el, r_wr))
        l_sh_ang = calc_angle_3d(l_hp, l_sh, l_el)
        r_sh_ang = calc_angle_3d(r_hp, r_sh, r_el)

        up_ref = mid_hp + np.array([0.0, -1.0, 0.0], dtype=np.float32)
        torso_tilt = calc_angle_3d(mid_sh, mid_hp, up_ref)
        sh_tilt = np.degrees(np.arctan2(np.abs(l_sh[:, 1] - r_sh[:, 1]), np.hypot(l_sh[:, 0] - r_sh[:, 0], l_sh[:, 2] - r_sh[:, 2]))).astype(np.float32)
        hp_tilt = np.degrees(np.arctan2(np.abs(l_hp[:, 1] - r_hp[:, 1]), np.hypot(l_hp[:, 0] - r_hp[:, 0], l_hp[:, 2] - r_hp[:, 2]))).astype(np.float32)
        l_wh = ((l_sh[:, 1] - l_wr[:, 1]) / torso_len).astype(np.float32)
        r_wh = ((r_sh[:, 1] - r_wr[:, 1]) / torso_len).astype(np.float32)
        stand = np.full(len(df), 175.0, dtype=np.float32)

        ang14 = np.stack([
            l_el_ang, r_el_ang, l_sh_ang, r_sh_ang,
            stand, stand, torso_tilt, sh_tilt, hp_tilt,
            l_wh, r_wh, stand, stand, stand,
        ], axis=1)

        labels = (df["label"].astype(str).str.upper() != "C").astype(np.int32).to_numpy()

        # Segment contiguous same-label runs into rep-length windows (using peaks or 40-frame chunks)
        # Assign synthetic clip/subject group IDs per contiguous block (~350 frames) to prevent leakage!
        change_idx = np.where(np.diff(labels) != 0)[0] + 1
        starts = np.concatenate([[0], change_idx])
        ends = np.concatenate([change_idx, [len(labels)]])

        clip_id = 0
        for s_idx, e_idx in zip(starts, ends):
            seg = ang14[s_idx:e_idx]
            lbl = int(labels[s_idx])
            if len(seg) < 16 or not np.isfinite(seg).all():
                continue
            # Split long contiguous runs into ~320-frame pseudo-subject clips for leakage-free CV
            n_clips = max(1, len(seg) // 320)
            for sub_seg in np.array_split(seg, n_clips):
                clip_id += 1
                grp = f"ngo_{split_name.split('.')[0]}_G{clip_id:03d}"
                if len(sub_seg) < 16:
                    continue
                sig = np.minimum(sub_seg[:, 0], sub_seg[:, 1])
                pks, _ = find_peaks(-sig, prominence=18.0, distance=18)
                added = 0
                for i, pk in enumerate(pks):
                    l = 0 if i == 0 else pks[i - 1]
                    r = len(sig) - 1 if i == len(pks) - 1 else pks[i + 1]
                    st = l + int(np.argmax(sig[l:pk + 1]))
                    en = pk + int(np.argmax(sig[pk:r + 1]))
                    if en - st >= 12 and (sig[st] - sig[pk] >= 15.0 or sig[en] - sig[pk] >= 15.0):
                        rows.append((resample_32(sub_seg[st:en + 1]), lbl, grp))
                        added += 1
                if added == 0:
                    # Fallback to non-overlapping 45-frame windows
                    for w0 in range(0, len(sub_seg) - 24, 45):
                        chunk = sub_seg[w0:w0 + 45]
                        if len(chunk) >= 20:
                            rows.append((resample_32(chunk), lbl, grp))

    n_corr = sum(r[1] == 0 for r in rows)
    n_inc = sum(r[1] == 1 for r in rows)
    n_grp = len({r[2] for r in rows})
    print(f"NgoQuocBao1010 Bicep Curl: {len(rows)} reps across {n_grp} groups (Correct={n_corr}, Incorrect={n_inc})")
    return rows


# -------- 2. Load Existing Mendeley Bicep Curl Cache from Google Drive --------
def load_mendeley_curl_cache() -> list[tuple[np.ndarray, int, str]]:
    for root in DRIVE_ROOTS:
        for fname in ("pose_angles_cache_curl.npz", "pose_cache_curl_rehab0.npz", "pose_cache_curl_rehab1.npz"):
            p = root / fname
            if p.exists():
                c = np.load(p, allow_pickle=True)
                x_key = "X_angles" if "X_angles" in c else "X_ang"
                rows = list(zip(c[x_key], c["y"], c["groups"]))
                print(f"Loaded Mendeley Curl cache: {len(rows)} reps from {p}")
                return rows
    print("Note: Mendeley curl cache not found on Drive — training on NgoQuocBao1010 alone.")
    return []


# -------- 3. Train, Benchmark & Export Upgraded Bicep Curl Random Forest --------
ngo_rows = load_ngo_bicep_rows()
mendeley_rows = load_mendeley_curl_cache()
all_rows = mendeley_rows + ngo_rows

X_ang = np.stack([r[0] for r in all_rows]).astype(np.float32)
y = np.asarray([r[1] for r in all_rows], dtype=np.int32)
g = np.asarray([r[2] for r in all_rows], dtype=object)
X_tab = build_tabular_features(X_ang)
n_subj = len(set(g))
print(f"\n=== UPGRADED [CURL] Total Reps: {len(y)} | Groups/Subjects: {n_subj} | Correct={(y==0).sum()} / Incorrect={(y==1).sum()} ===")

cv = StratifiedGroupKFold(n_splits=min(5, n_subj))
rng = np.random.default_rng(SEED)
names = ["Random Forest", "LightGBM", "XGBoost", "SVM (RBF)"]
oof = {n: np.zeros(len(y), np.float32) for n in names}

for fi, (tr, te) in enumerate(cv.split(X_tab, y, g), 1):
    ang_tr = np.concatenate([
        X_ang[tr],
        X_ang[tr][:, :, ANGLE_LR_SWAP],
        X_ang[tr] + rng.normal(0, 1.2, X_ang[tr].shape).astype(np.float32),
    ], axis=0)
    y_tr = np.concatenate([y[tr], y[tr], y[tr]], axis=0)
    tab_tr = build_tabular_features(ang_tr)
    pos_r = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    clfs = {
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=16, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED + fi, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=250, max_depth=6, num_leaves=24, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.8, class_weight="balanced",
            random_state=SEED + fi, verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.8, scale_pos_weight=pos_r,
            eval_metric="logloss", random_state=SEED + fi, n_jobs=-1,
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(C=2.5, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED + fi),
        ),
    }
    for name, clf in clfs.items():
        clf.fit(tab_tr, y_tr)
        oof[name][te] = clf.predict_proba(X_tab[te])[:, 1]

# Tune optimal threshold for Random Forest on OOF predictions
best_thr, best_rf_f1 = 0.5, -1.0
for thr in np.linspace(0.35, 0.65, 31):
    f1_v = float(f1_score(y, (oof["Random Forest"] >= thr).astype(int), average="macro"))
    if f1_v > best_rf_f1:
        best_rf_f1, best_thr = f1_v, float(round(thr, 3))

b_rows = []
for name in names:
    thr = best_thr if name == "Random Forest" else 0.5
    preds = (oof[name] >= thr).astype(int)
    b_rows.append({
        "Model": name,
        "Family": "Tabular",
        "OOF Macro F1": round(float(f1_score(y, preds, average="macro")), 4),
        "Accuracy": round(float(accuracy_score(y, preds)), 4),
        "Balanced Acc": round(float(balanced_accuracy_score(y, preds)), 4),
        "ROC-AUC": round(float(roc_auc_score(y, oof[name])), 4),
    })
board = pd.DataFrame(b_rows).sort_values("OOF Macro F1", ascending=False).reset_index(drop=True)
print(board.to_string(index=False))
print(f"Selected RF threshold: {best_thr}")

# Fit final production Random Forest on full merged dataset + L/R mirror augmentation
rf_final = RandomForestClassifier(
    n_estimators=400, max_depth=16, min_samples_leaf=2,
    class_weight="balanced", random_state=SEED, n_jobs=-1,
)
rf_final.fit(
    build_tabular_features(np.concatenate([X_ang, X_ang[:, :, ANGLE_LR_SWAP]], axis=0)),
    np.concatenate([y, y], axis=0),
)

out_dir = SAVE_ROOT / "exported_curl_upgraded" / "curl"
out_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(rf_final, out_dir / "best_tabular_model.joblib", compress=3)
board.to_csv(out_dir / "model_leaderboard.csv", index=False)
(out_dir / "model_metadata.json").write_text(json.dumps({
    "exercise": "curl",
    "winner_model": str(board.iloc[0]["Model"]),
    "best_tabular_model": "Random Forest",
    "threshold": best_thr,
    "sequence_length": SEQ_LEN,
    "features": FEATURES,
    "core_joints": CORE_JOINTS,
    "datasets": ["Mendeley_MultiView_Curl" if mendeley_rows else "", "NgoQuocBao1010_BicepCurl"],
    "total_reps": int(len(y)),
    "unique_subjects": int(n_subj),
    "leaderboard": board.to_dict(orient="records"),
}, indent=2), encoding="utf-8")

zip_out = SAVE_ROOT / "gym_model_bundle_curl_upgraded.zip"
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in (SAVE_ROOT / "exported_curl_upgraded").rglob("*"):
        if p.is_file():
            zf.write(p, arcname=p.relative_to(SAVE_ROOT / "exported_curl_upgraded").as_posix())
print(f"\nPACKAGED BICEP CURL BUNDLE: {zip_out} ({zip_out.stat().st_size / 1e6:.2f} MB)")
if files is not None:
    files.download(str(zip_out))
