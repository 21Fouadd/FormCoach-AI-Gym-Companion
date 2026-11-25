"""Locally trains real Random Forest (+ SVM & HistGB benchmark) models for Curl, Shoulder Press & Row from Mendeley kgbb3yn47p."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.pose_engine import CORE_JOINTS, FEATURES, POSE_TASK_PATH, SEQUENCE_LENGTH, build_tabular_features  # noqa: E402

WORK_DIR = Path(r"C:\Users\riadk\.gemini\antigravity\brain\dd804d97-60b5-4387-bd3d-b2732aa1bdf8\scratch\local_train\mendeley")
WORK_DIR.mkdir(parents=True, exist_ok=True)
MODELS_ROOT = ROOT / "backend" / "models"

SEED = 42
FRAME_STEP = 4
ANGLE_LR_SWAP = [1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 11, 13, 12]
MENDELEY_EXERCISES = {
    "curl": ["curl", "bicep"],
    "shoulder": ["shoulder", "press"],
    "row": ["row", "bent"],
}

_WORKER_DET = None


def _init_worker(pose_task_str: str) -> None:
    global _WORKER_DET
    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=pose_task_str),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    _WORKER_DET = mp.tasks.vision.PoseLandmarker.create_from_options(opts)


def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-8), -1.0, 1.0))))


def _extract_ang(res) -> np.ndarray | None:
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
        _calc_angle(p[11], p[13], p[15]), _calc_angle(p[12], p[14], p[16]),
        _calc_angle(p[23], p[11], p[13]), _calc_angle(p[24], p[12], p[14]),
        _calc_angle(p[11], p[23], p[25]), _calc_angle(p[12], p[24], p[26]),
        _calc_angle(sh, hp, hp + np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        tilt(11, 12), tilt(23, 24),
        float((p[11, 1] - p[15, 1]) / sc), float((p[12, 1] - p[16, 1]) / sc),
        _calc_angle(sh, hp, 0.5 * (p[27] + p[28])),
        _calc_angle(p[23], p[25], p[27]), _calc_angle(p[24], p[26], p[28]),
    ], dtype=np.float32)
    return ang if np.isfinite(ang).all() else None


def _rom(x: np.ndarray) -> float:
    return float(np.percentile(x, 90) - np.percentile(x, 10)) if len(x) else 0.0


def _process_one_video(args: tuple[str, str, int, str]) -> list[tuple[np.ndarray, int, str]]:
    vpath_str, ex, lbl, subj = args
    cap = cv2.VideoCapture(vpath_str)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frames, angles, idx = [], [], 0
    try:
        while cap.grab():
            if idx % FRAME_STEP == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    h, w = bgr.shape[:2]
                    if max(h, w) > 480:
                        s = 480.0 / max(h, w)
                        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
                    res = _WORKER_DET.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                    ang = _extract_ang(res)
                    if ang is not None:
                        frames.append(idx)
                        angles.append(ang)
            idx += 1
    finally:
        cap.release()

    if len(frames) < 6:
        return []
    f_arr = np.asarray(frames, dtype=np.int32)
    a_arr = np.asarray(angles, dtype=np.float32)

    if ex == "curl":
        sig, inv, pr, dr = a_arr[:, :2].min(1), True, 15.0, 13.0
    elif ex == "shoulder":
        sig, inv, pr, dr = a_arr[:, 2:4].max(1), False, 20.0, 16.0
    else:
        if max(_rom(a_arr[:, 0]), _rom(a_arr[:, 1])) >= max(28.0, 0.7 * _rom(a_arr[:, 4:6].mean(1))):
            sig, inv, pr, dr = a_arr[:, :2].min(1), True, 15.0, 13.0
        else:
            sig, inv, pr, dr = a_arr[:, 4:6].mean(1), True, 18.0, 15.0

    grid = np.arange(f_arr[0], f_arr[-1] + 1, FRAME_STEP)
    curve = np.interp(grid, f_arr, sig)
    if len(curve) >= 5:
        curve = np.convolve(curve, [0.25, 0.5, 0.25], mode="same")
    pks, _ = find_peaks(-curve if inv else curve, prominence=pr, distance=max(2, int(0.65 * fps / FRAME_STEP)))
    segs = []
    for i, pk in enumerate(pks):
        l, r = (0 if i == 0 else pks[i - 1]), (len(curve) - 1 if i == len(pks) - 1 else pks[i + 1])
        s_i = l + int(np.argmax(curve[l:pk + 1]) if inv else np.argmin(curve[l:pk + 1]))
        e_i = pk + int(np.argmax(curve[pk:r + 1]) if inv else np.argmin(curve[pk:r + 1]))
        ok = (curve[s_i] - curve[pk] >= dr and curve[e_i] - curve[pk] >= dr) if inv else (curve[pk] - curve[s_i] >= dr and curve[pk] - curve[e_i] >= dr)
        if ok and e_i > s_i + 2:
            segs.append((int(grid[s_i]), int(grid[e_i]) + 1))
    if not segs and (f_arr[-1] - f_arr[0]) >= int(0.8 * fps):
        segs.append((int(f_arr[0]), int(f_arr[-1]) + 1))

    out = []
    for s, e in segs:
        m = (f_arr >= s) & (f_arr < e)
        if m.sum() < 4:
            continue
        t_grid = np.linspace(float(s), float(max(s + 1, e - 1)), SEQUENCE_LENGTH)
        seq = np.stack([np.interp(t_grid, f_arr[m], a_arr[m, c]) for c in range(14)], axis=-1).astype(np.float32)
        out.append((seq, lbl, subj))
    return out


def ensure_mendeley_zip() -> Path:
    dest = WORK_DIR / "Dataset Exercise-wise.zip"
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"[cached] {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
        return dest

    resp = requests.get(
        "https://data.mendeley.com/public-api/datasets/kgbb3yn47p/files?version=3&folder_id=root",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    resp.raise_for_status()
    items = resp.json()
    items = items if isinstance(items, list) else items.get("data", [])
    chosen = next((x for x in items if "exercise-wise" in x.get("filename", "").lower()), items[0])
    cd = chosen.get("content_details") or {}
    url = cd.get("download_url") or (
        f"https://api.data.mendeley.com/datasets/kgbb3yn47p/files/{chosen['id']}/file_downloaded?version=3"
    )
    total = int(cd.get("size") or 0)
    tmp = dest.with_suffix(".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if 0 < existing < total:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"[resume] Downloading Mendeley {chosen['filename']} from {existing/1e9:.2f}/{total/1e9:.2f} GB...", flush=True)
    else:
        existing = 0
        print(f"[download] Downloading Mendeley {chosen['filename']} ({total/1e9:.2f} GB)...", flush=True)

    with requests.get(url, headers=headers, stream=True, timeout=600, allow_redirects=True) as r:
        r.raise_for_status()
        if mode == "ab" and r.status_code != 206:
            mode = "wb"
            existing = 0
        done = existing
        next_log = ((done // (200 * 1024 * 1024)) + 1) * 200 * 1024 * 1024
        with tmp.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total > 0 and done >= next_log:
                        print(f"  Mendeley ZIP: {done/1e9:.2f}/{total/1e9:.2f} GB ({100*done/total:.0f}%)", flush=True)
                        next_log += 200 * 1024 * 1024
    tmp.replace(dest)
    print(f"[done] Mendeley ZIP ready ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
    return dest


def parse_meta(rel: str) -> tuple[str, int] | None:
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


def train_and_save(ex_name: str, rows: list[tuple[np.ndarray, int, str]]) -> None:
    X_ang = np.stack([r[0] for r in rows]).astype(np.float32)
    y = np.asarray([r[1] for r in rows], dtype=np.int32)
    g = np.asarray([r[2] for r in rows], dtype=object)
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

    b_rows = []
    for name in names:
        preds = (oof[name] >= 0.5).astype(int)
        b_rows.append({
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
    board = pd.DataFrame(b_rows).sort_values("OOF Macro F1", ascending=False).reset_index(drop=True)
    print(board.to_string(index=False), flush=True)

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
        "frame_step": FRAME_STEP,
        "features": FEATURES,
        "core_joints": CORE_JOINTS,
        "dataset": "Mendeley Multi-View Seven Fitness Exercises (kgbb3yn47p v3)",
        "total_reps": int(len(y)),
        "unique_subjects": int(n_subj),
        "leaderboard": board.to_dict(orient="records"),
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"SAVED {ex_name} Random Forest bundle -> {out_dir}", flush=True)


def main() -> None:
    zip_path = ensure_mendeley_zip()
    n_workers = max(2, min(6, (os.cpu_count() or 4) - 2))
    print(f"Starting parallel MediaPipe extraction with {n_workers} CPU workers...", flush=True)

    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        all_mp4s = [m for m in zf.namelist() if m.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]

        for ex, kws in MENDELEY_EXERCISES.items():
            cache_f = WORK_DIR / f"cache_{ex}.npz"
            if cache_f.exists():
                c = np.load(cache_f, allow_pickle=True)
                rows = list(zip(c["X_ang"], c["y"], c["groups"]))
                print(f"[cached poses] {ex}: {len(rows)} reps", flush=True)
                train_and_save(ex, rows)
                continue

            jobs = []
            seen = set()
            ex_tmp = tmp_dir / ex
            ex_tmp.mkdir(parents=True, exist_ok=True)
            for mem in all_mp4s:
                if not any(kw in mem.lower() for kw in kws):
                    continue
                bn = Path(mem).name.lower()
                if bn in seen:
                    continue
                meta = parse_meta(mem)
                if meta is None:
                    continue
                seen.add(bn)
                subj, lbl = meta
                out_v = ex_tmp / f"{len(jobs):04d}_{Path(mem).name}"
                with zf.open(mem) as src, out_v.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                jobs.append((str(out_v), ex, lbl, subj))

            print(f"\nExtracting poses for '{ex}': {len(jobs)} videos across {n_workers} workers...", flush=True)
            rows: list[tuple[np.ndarray, int, str]] = []
            t0 = time.perf_counter()
            done_v = 0
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(str(POSE_TASK_PATH),)) as pool:
                futs = [pool.submit(_process_one_video, job) for job in jobs]
                for fut in as_completed(futs):
                    rows.extend(fut.result())
                    done_v += 1
                    if done_v % 30 == 0 or done_v == len(jobs):
                        print(f"  [{ex}] {done_v}/{len(jobs)} videos -> {len(rows)} reps ({time.perf_counter()-t0:.0f}s)", flush=True)

            shutil.rmtree(ex_tmp, ignore_errors=True)
            if not rows:
                raise RuntimeError(f"Extracted 0 reps for {ex}!")
            np.savez_compressed(
                cache_f,
                X_ang=np.stack([r[0] for r in rows]),
                y=np.asarray([r[1] for r in rows]),
                groups=np.asarray([r[2] for r in rows], dtype=object),
            )
            train_and_save(ex, rows)

    zip_path.unlink(missing_ok=True)
    print("\nMENDELEY LOCAL TRAINING COMPLETE (curl + shoulder + row)!", flush=True)


if __name__ == "__main__":
    main()
