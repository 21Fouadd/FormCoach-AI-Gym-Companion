"""Locally trains real Random Forest (+ SVM & HistGB benchmark) models for Lunge & Arm Abduction from REHAB24-6 3D joints."""
from __future__ import annotations

import json
import re
import sys
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.pose_engine import CORE_JOINTS, FEATURES, SEQUENCE_LENGTH, build_tabular_features  # noqa: E402

WORK_DIR = Path(r"C:\Users\riadk\.gemini\antigravity\brain\dd804d97-60b5-4387-bd3d-b2732aa1bdf8\scratch\local_train\rehab24_6")
WORK_DIR.mkdir(parents=True, exist_ok=True)
MODELS_ROOT = ROOT / "backend" / "models"

SEED = 42
ANGLE_LR_SWAP = [1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 11, 13, 12]
REHAB_EXERCISES = {"arm_abduction": 1, "lunge": 5}

# Mapping from REHAB24-6 26-joint skeleton (joints_names.txt) -> MediaPipe equivalent roles:
# 7: LeftArm (shoulder), 8: LeftForeArm (elbow), 9: LeftHand (wrist)
# 12: RightArm (shoulder), 13: RightForeArm (elbow), 14: RightHand (wrist)
# 16: LeftUpLeg (hip), 17: LeftLeg (knee), 18: LeftFoot (ankle)
# 21: RightUpLeg (hip), 22: RightLeg (knee), 23: RightFoot (ankle)
J_LS, J_LE, J_LW = 7, 8, 9
J_RS, J_RE, J_RW = 12, 13, 14
J_LH, J_LK, J_LA = 16, 17, 18
J_RH, J_RK, J_RA = 21, 22, 23


def download_with_progress(url: str, dest: Path, label: str) -> Path:
    if dest.exists() and dest.stat().st_size > 10_000:
        if dest.suffix != ".zip" or zipfile.is_zipfile(dest):
            print(f"[cached] {label}: {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
            return dest
        dest.unlink(missing_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {label} from {url} ...", flush=True)
    with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        next_log = 50 * 1024 * 1024
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total > 0 and done >= next_log:
                        print(f"  {label}: {done/1e6:.0f}/{total/1e6:.0f} MB ({100*done/total:.0f}%)", flush=True)
                        next_log += 50 * 1024 * 1024
    tmp.replace(dest)
    print(f"[done] {label}: {dest.stat().st_size / 1e6:.1f} MB", flush=True)
    return dest


def load_joints_array(zf: zipfile.ZipFile, member: str) -> np.ndarray | None:
    low = member.lower()
    with zf.open(member) as f:
        if low.endswith(".npy"):
            arr = np.load(f)
        elif low.endswith(".csv") or low.endswith(".txt"):
            df = pd.read_csv(f, sep=None, engine="python", header=None)
            # Drop header row if non-numeric
            arr = pd.to_numeric(df.iloc[0], errors="coerce")
            if arr.isna().mean() > 0.5:
                df = df.iloc[1:]
            vals = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            # Drop leading frame index column if ncols == 79
            if vals.shape[1] == 79:
                vals = vals[:, 1:]
            arr = vals
        else:
            return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 78:
        arr = arr.reshape(-1, 26, 3)
    elif arr.ndim == 3 and arr.shape[1] == 26 and arr.shape[2] == 3:
        pass
    elif arr.ndim == 3 and arr.shape[2] == 26 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 1))
    else:
        return None
    return arr


def calc_angle_vec(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u = a - b
    v = c - b
    num = np.sum(u * v, axis=-1)
    den = np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-8)
    return np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0))).astype(np.float32)


def joints_to_14_angles(pts: np.ndarray) -> np.ndarray | None:
    """Converts (T, 26, 3) 3D mocap joints into the exact 14 scale-invariant MediaPipe angles."""
    if len(pts) < 6 or not np.isfinite(pts).all():
        return None
    shoulders = 0.5 * (pts[:, J_LS] + pts[:, J_RS])
    hips = 0.5 * (pts[:, J_LH] + pts[:, J_RH])
    ankles = 0.5 * (pts[:, J_LA] + pts[:, J_RA])

    # Auto-detect world UP axis from upright standing spine vector mean(shoulders - hips)
    spine_mean = np.median(shoulders - hips, axis=0)
    up_axis = int(np.argmax(np.abs(spine_mean)))
    up_sign = 1.0 if spine_mean[up_axis] >= 0 else -1.0
    up_vec = np.zeros(3, dtype=np.float32)
    up_vec[up_axis] = up_sign
    horiz_axes = [ax for ax in (0, 1, 2) if ax != up_axis]

    scale = np.maximum(np.linalg.norm(shoulders - hips, axis=-1), 1e-5)

    def line_tilt(i: int, j: int) -> np.ndarray:
        v = pts[:, i] - pts[:, j]
        horiz = np.hypot(v[:, horiz_axes[0]], v[:, horiz_axes[1]])
        return np.degrees(np.arctan2(np.abs(v[:, up_axis]), np.maximum(horiz, 1e-8))).astype(np.float32)

    torso_target = hips + up_vec[None, :]
    # MediaPipe wrist_height = positive when wrist is ABOVE shoulder along UP axis
    l_wh = ((pts[:, J_LW, up_axis] - pts[:, J_LS, up_axis]) * up_sign) / scale
    r_wh = ((pts[:, J_RW, up_axis] - pts[:, J_RS, up_axis]) * up_sign) / scale

    angles = np.stack([
        calc_angle_vec(pts[:, J_LS], pts[:, J_LE], pts[:, J_LW]),
        calc_angle_vec(pts[:, J_RS], pts[:, J_RE], pts[:, J_RW]),
        calc_angle_vec(pts[:, J_LH], pts[:, J_LS], pts[:, J_LE]),
        calc_angle_vec(pts[:, J_RH], pts[:, J_RS], pts[:, J_RE]),
        calc_angle_vec(pts[:, J_LS], pts[:, J_LH], pts[:, J_LK]),
        calc_angle_vec(pts[:, J_RS], pts[:, J_RH], pts[:, J_RK]),
        calc_angle_vec(shoulders, hips, torso_target),
        line_tilt(J_LS, J_RS),
        line_tilt(J_LH, J_RH),
        l_wh.astype(np.float32),
        r_wh.astype(np.float32),
        calc_angle_vec(shoulders, hips, ankles),
        calc_angle_vec(pts[:, J_LH], pts[:, J_LK], pts[:, J_LA]),
        calc_angle_vec(pts[:, J_RH], pts[:, J_RK], pts[:, J_RA]),
    ], axis=-1).astype(np.float32)
    return angles if np.isfinite(angles).all() else None


def resample_32(angles_sub: np.ndarray) -> np.ndarray:
    t_old = np.linspace(0.0, 1.0, len(angles_sub))
    t_new = np.linspace(0.0, 1.0, SEQUENCE_LENGTH)
    return np.stack([np.interp(t_new, t_old, angles_sub[:, c]) for c in range(14)], axis=-1).astype(np.float32)


def train_and_save(ex_name: str, seqs: list[np.ndarray], labels: list[int], groups: list[str]) -> None:
    X_ang = np.stack(seqs).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    g = np.asarray(groups, dtype=object)
    X_tab = build_tabular_features(X_ang)
    n_subj = len(set(g))
    print(f"\n=== [{ex_name.upper()}] Reps: {len(y)} | Subjects: {n_subj} | Correct={(y==0).sum()} / Incorrect={(y==1).sum()} ===", flush=True)

    cv = StratifiedGroupKFold(n_splits=min(5, n_subj))
    rng = np.random.default_rng(SEED)
    names = ["Random Forest", "SVM (RBF)", "HistGradientBoosting"]
    oof = {n: np.zeros(len(y), dtype=np.float32) for n in names}
    lat = {n: [] for n in names}

    for fold_i, (tr, te) in enumerate(cv.split(X_tab, y, g), 1):
        ang_tr = np.concatenate([
            X_ang[tr],
            X_ang[tr][:, :, ANGLE_LR_SWAP],
            X_ang[tr] + rng.normal(0.0, 1.2, size=X_ang[tr].shape).astype(np.float32),
        ], axis=0)
        y_tr = np.concatenate([y[tr], y[tr], y[tr]], axis=0)
        tab_tr = build_tabular_features(ang_tr)

        clfs = {
            "Random Forest": RandomForestClassifier(
                n_estimators=250, max_depth=12, min_samples_leaf=2,
                class_weight="balanced", random_state=SEED + fold_i, n_jobs=-1,
            ),
            "SVM (RBF)": make_pipeline(
                StandardScaler(),
                SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED + fold_i),
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=200, max_depth=5, learning_rate=0.05, class_weight="balanced", random_state=SEED + fold_i,
            ),
        }
        for name, clf in clfs.items():
            clf.fit(tab_tr, y_tr)
            oof[name][te] = clf.predict_proba(X_tab[te])[:, 1]
            t0 = time.perf_counter()
            for _ in range(10):
                clf.predict_proba(X_tab[te[:1]])
            lat[name].append(1000.0 * (time.perf_counter() - t0) / 10.0)

    rows = []
    for name in names:
        preds = (oof[name] >= 0.5).astype(int)
        rows.append({
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
    board = pd.DataFrame(rows).sort_values("OOF Macro F1", ascending=False).reset_index(drop=True)
    print(board.to_string(index=False), flush=True)

    # Fit winning Random Forest on 100% mirror-augmented data
    ang_full = np.concatenate([X_ang, X_ang[:, :, ANGLE_LR_SWAP]], axis=0)
    y_full = np.concatenate([y, y], axis=0)
    rf_final = RandomForestClassifier(
        n_estimators=250, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    rf_final.fit(build_tabular_features(ang_full), y_full)

    out_dir = MODELS_ROOT / ex_name
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_final, out_dir / "best_tabular_model.joblib", compress=3)
    board.to_csv(out_dir / "model_leaderboard.csv", index=False)
    meta = {
        "exercise": ex_name,
        "winner_model": str(board.iloc[0]["Model"]),
        "best_tabular_model": "Random Forest",
        "threshold": 0.5,
        "sequence_length": SEQUENCE_LENGTH,
        "features": FEATURES,
        "core_joints": CORE_JOINTS,
        "dataset": "REHAB24-6 (3D Mocap Joints)",
        "total_reps": int(len(y)),
        "unique_subjects": int(n_subj),
        "leaderboard": board.to_dict(orient="records"),
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"SAVED {ex_name} Random Forest bundle -> {out_dir}", flush=True)


def main() -> None:
    seg_csv = download_with_progress(
        "https://zenodo.org/records/13305826/files/Segmentation.csv?download=1",
        WORK_DIR / "Segmentation.csv",
        "REHAB24-6 Segmentation.csv",
    )
    joints_zip = download_with_progress(
        "https://zenodo.org/records/13305826/files/3d_joints.zip?download=1",
        WORK_DIR / "3d_joints.zip",
        "REHAB24-6 3d_joints.zip",
    )

    df = pd.read_csv(seg_csv, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.lower()

    with zipfile.ZipFile(joints_zip) as zf:
        all_files = [n for n in zf.namelist() if not n.endswith("/")]
        print(f"3d_joints.zip entries: {len(all_files)} | sample: {all_files[:4]}", flush=True)

        for ex_name, ex_id in REHAB_EXERCISES.items():
            sub = df[df.exercise_id.astype(int).eq(ex_id)].copy()
            seqs, labels, groups = [], [], []
            for vid_id, reps in sub.groupby("video_id"):
                key = str(vid_id).lower()
                matched = [n for n in all_files if re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", n.lower())]
                if not matched:
                    continue
                pts = load_joints_array(zf, matched[0])
                if pts is None or len(pts) < 10:
                    continue
                angles_all = joints_to_14_angles(pts)
                if angles_all is None:
                    continue
                # Auto-align mocap frame index if 3d_joints is 120Hz while Segmentation.csv is 30Hz (or 1:1)
                max_end = int(reps.last_frame.max())
                ratio = 1.0
                if len(angles_all) > max_end * 2.5:
                    ratio = round(len(angles_all) / max(max_end, 1))
                    ratio = 4.0 if 3.2 <= ratio <= 4.8 else (2.0 if 1.6 <= ratio <= 2.4 else 1.0)
                person = f"rehab_{reps.person_id.iloc[0]}"
                for s, e, corr in zip(reps.first_frame.to_numpy(int), reps.last_frame.to_numpy(int), reps.correctness):
                    s_i = max(0, int(round(s * ratio)))
                    e_i = min(len(angles_all), int(round((e + 1) * ratio)))
                    if e_i - s_i < 6:
                        continue
                    seqs.append(resample_32(angles_all[s_i:e_i]))
                    labels.append(int(int(corr) == 0))
                    groups.append(person)
            if not seqs:
                raise RuntimeError(f"Extracted 0 reps for {ex_name} from 3d_joints.zip!")
            train_and_save(ex_name, seqs, labels, groups)

    # Free 551 MB zip after successful training
    joints_zip.unlink(missing_ok=True)
    print("\nREHAB24-6 LOCAL TRAINING COMPLETE (lunge + arm_abduction)!", flush=True)


if __name__ == "__main__":
    main()
