# === AI Personal Gym Companion — 1-Cell Instant Test (All 7 Exercises + Auto-Detect, Zero Retraining) ===
# Paste this entire script into ONE Google Colab cell and run it directly!
# Supports: auto, squat, push, curl, shoulder (OHP), row (Bent-Over Row / Deadlift), lunge, arm_abduction.

import importlib, json, os, shutil, subprocess, sys, urllib.request
from pathlib import Path

if importlib.util.find_spec("mediapipe") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe", "joblib", "scipy"], check=True)

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import Markdown, Video, display
from scipy.signal import find_peaks

try:
    from google.colab import drive, files
except ImportError:
    drive = None
    files = None

EXERCISE_MODE = "auto"  # Change to "squat", "push", "curl", "shoulder", "row", "lunge", "arm_abduction", or keep "auto"
DRIVE_ROOT = Path("/content/drive/MyDrive/GymForm_Graduation_Project")
WORK_DIR = Path("/content/gymform_test")
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = DRIVE_ROOT / "test_outputs" if (DRIVE_ROOT.exists() or os.path.exists("/content/drive/MyDrive")) else WORK_DIR / "outputs"

if drive is not None and not os.path.exists("/content/drive/MyDrive"):
    try:
        drive.mount("/content/drive")
        OUT_DIR = DRIVE_ROOT / "test_outputs"
    except Exception:
        pass
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
JOINT_LINKS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
SEQ_LEN = 32


class TemporalAttentionPool(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        w = tf.nn.softmax(self.score_dense(inputs), axis=1)
        return tf.reduce_sum(inputs * w, axis=1)


_LOADED_BUNDLES: dict[str, dict] = {}


def load_exercise_bundle(ex: str) -> dict:
    if ex in _LOADED_BUNDLES:
        return _LOADED_BUNDLES[ex]
    bundle = {"tab": None, "deep": None, "ang_sc": None, "skel_sc": None, "meta": {}}
    # Reuse in-memory notebook models if Section 12 was already run in this runtime
    mem = globals().get("_DEMO_MODELS_CACHE", {}).get(ex)
    if mem:
        w = str(globals().get("WINNER_MODEL", ""))
        best_t = globals().get("best_tab")
        best_d = globals().get("best_deep")
        bundle["tab"] = mem["tab"].get(best_t)
        bundle["deep"] = mem["deep"].get(best_d)
        bundle["ang_sc"] = globals().get("ang_scaler_full")
        bundle["skel_sc"] = globals().get("skel_scaler_full")
        bundle["meta"] = {"winner_model": w, "best_deep_model": best_d, "best_tabular_model": best_t, "threshold": 0.5}
    else:
        bdir = DRIVE_ROOT / f"exported_model_{ex}"
        if bdir.exists():
            if (bdir / "model_metadata.json").exists():
                bundle["meta"] = json.loads((bdir / "model_metadata.json").read_text(encoding="utf-8"))
            for k, fn in [("tab", "best_tabular_model.joblib"), ("ang_sc", "angle_scaler.joblib"), ("skel_sc", "skel_scaler.joblib")]:
                if (bdir / fn).exists():
                    try:
                        bundle[k] = joblib.load(bdir / fn)
                    except Exception:
                        pass
            if (bdir / "best_deep_model.keras").exists():
                try:
                    bundle["deep"] = tf.keras.models.load_model(
                        bdir / "best_deep_model.keras", custom_objects={"TemporalAttentionPool": TemporalAttentionPool}
                    )
                except Exception:
                    pass
    _LOADED_BUNDLES[ex] = bundle
    return bundle


def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-8), -1.0, 1.0))))


def _rom(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 90) - np.percentile(arr, 10)) if len(arr) else 0.0


def extract_frame_pose(result, img_w: int, img_h: int):
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return None
    l2 = result.pose_landmarks[0]
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
    hips_vis = max(l2[23].visibility, l2[24].visibility) >= 0.35
    lower_vis = hips_vis and max(l2[25].visibility, l2[26].visibility) >= 0.30
    arm_vis = (min(l2[11].visibility, l2[13].visibility, l2[15].visibility), min(l2[12].visibility, l2[14].visibility, l2[16].visibility))
    p = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]], dtype=np.float32)
    sh, hp = 0.5 * (p[11] + p[12]), 0.5 * (p[23] + p[24])
    raw_sc = float(np.linalg.norm(sh - hp))
    scale = raw_sc if (hips_vis and raw_sc >= 0.12) else max(raw_sc, float(np.linalg.norm(p[11] - p[12])) * 1.35, 0.20)
    tilt = lambda a, b: float(np.degrees(np.arctan2(abs(p[a, 1] - p[b, 1]), np.hypot(p[a, 0] - p[b, 0], p[a, 2] - p[b, 2]))))
    ang = np.array([
        _calc_angle(p[11], p[13], p[15]), _calc_angle(p[12], p[14], p[16]),
        _calc_angle(p[23], p[11], p[13]), _calc_angle(p[24], p[12], p[14]),
        _calc_angle(p[11], p[23], p[25]), _calc_angle(p[12], p[24], p[26]),
        _calc_angle(sh, hp, hp + np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        tilt(11, 12), tilt(23, 24), float((p[11, 1] - p[15, 1]) / scale), float((p[12, 1] - p[16, 1]) / scale),
        _calc_angle(sh, hp, 0.5 * (p[27] + p[28])), _calc_angle(p[23], p[25], p[27]), _calc_angle(p[24], p[26], p[28]),
    ], dtype=np.float32)
    p2 = np.array([[lm.x * img_w, lm.y * img_h] for lm in l2], dtype=np.float32)
    e2d = np.array([_calc_angle(p2[11], p2[13], p2[15]), _calc_angle(p2[12], p2[14], p2[16])], dtype=np.float32)
    return ang, ((p[CORE_JOINTS] - hp) / scale).astype(np.float32), e2d, arm_vis, hips_vis, lower_vis


def auto_detect_exercise(angles_arr: np.ndarray, elb_2d: np.ndarray, arm_vis: np.ndarray, lower_vis: bool) -> str:
    if len(angles_arr) == 0:
        return "squat"
    med_torso, p90_torso = float(np.median(angles_arr[:, 6])), float(np.percentile(angles_arr[:, 6], 90))
    avg_hip = 0.5 * (angles_arr[:, 4] + angles_arr[:, 5])
    l_ok, r_ok = float(np.median(arm_vis[:, 0])) >= 0.22, float(np.median(arm_vis[:, 1])) >= 0.22
    sides = [s for s, ok in ((0, l_ok), (1, r_ok)) if ok] or [int(np.argmax(np.median(arm_vis, axis=0)))]
    p90_wh = max(float(np.percentile(angles_arr[:, 9 + s], 90)) for s in sides)
    p90_sho = max(float(np.percentile(angles_arr[:, 2 + s], 90)) for s in sides)
    sho_rom = max(_rom(angles_arr[:, 2 + s]) for s in sides)
    elb_rom3 = max(_rom(angles_arr[:, s]) for s in sides)
    elb_rom = max(elb_rom3, max(_rom(0.5 * angles_arr[:, s] + 0.5 * elb_2d[:, s]) for s in sides))
    min_knee = np.minimum(angles_arr[:, 12], angles_arr[:, 13])
    knee_rom = max(_rom(0.5 * (angles_arr[:, 12] + angles_arr[:, 13])), _rom(min_knee))
    if float(np.mean((angles_arr[:, 6] > 58.0) & (avg_hip > 128.0))) >= 0.30 or (med_torso > 78.0 and float(np.min(angles_arr[:, 6])) > 50.0 and float(np.percentile(avg_hip, 75)) > 120.0):
        return "push"
    if p90_wh > 0.22 and p90_sho > 96.0:
        return "shoulder"
    if lower_vis:
        if p90_torso > 42.0 and 22.0 < med_torso < 86.0 and float(np.percentile(0.5 * (angles_arr[:, 12] + angles_arr[:, 13]), 10)) > 112.0:
            return "row"
        if med_torso < 48.0 and knee_rom >= 26.0 and float(np.percentile(min_knee, 10)) < 128.0 and float(np.percentile(np.minimum(angles_arr[:, 4], angles_arr[:, 5]), 10)) < 132.0:
            return "lunge" if float(np.percentile(np.abs(angles_arr[:, 4] - angles_arr[:, 5]), 85)) >= 32.0 else "squat"
    if med_torso < 35.0 and p90_wh <= 0.20 and sho_rom >= 34.0 and p90_sho >= 62.0 and sho_rom > 0.85 * elb_rom3:
        return "arm_abduction"
    if elb_rom >= 24.0 and p90_sho < 58.0:
        return "curl"
    return "squat" if lower_vis else "curl"


def _tab_features(ang: np.ndarray) -> np.ndarray:
    p1, p2, p3 = ang[:, :11, :], ang[:, 11:22, :], ang[:, 22:, :]
    vel = np.diff(ang, axis=1)
    pairs = [(0, 1), (2, 3), (4, 5), (9, 10), (12, 13)]
    return np.concatenate([
        ang.mean(1), ang.std(1), ang.min(1), ang.max(1), ang.max(1) - ang.min(1),
        p1.mean(1), p2.mean(1), p2.min(1), p2.max(1), p3.mean(1), np.abs(vel).mean(1), np.abs(vel).max(1),
        np.stack([np.abs(ang[:, :, l] - ang[:, :, r]).mean(1) for l, r in pairs], 1),
        np.stack([np.abs(ang[:, :, l] - ang[:, :, r]).max(1) for l, r in pairs], 1),
    ], axis=1).astype(np.float32)


def _find_1d_segs(grid: np.ndarray, sig: np.ndarray, frames: np.ndarray, invert: bool, prom: float, drop: float, dist: int):
    c = np.convolve(np.interp(grid, frames, sig), [0.25, 0.5, 0.25], mode="same") if len(grid) >= 5 else np.interp(grid, frames, sig)
    pks, _ = find_peaks(-c if invert else c, prominence=prom, distance=max(2, dist))
    segs = []
    for i, pk in enumerate(pks):
        l, r = (0 if i == 0 else pks[i - 1]), (len(c) - 1 if i == len(pks) - 1 else pks[i + 1])
        s = l + int(np.argmax(c[l:pk + 1]) if invert else np.argmin(c[l:pk + 1]))
        e = pk + int(np.argmax(c[pk:r + 1]) if invert else np.argmin(c[pk:r + 1]))
        ok = (c[s] - c[pk] >= drop and c[e] - c[pk] >= drop) if invert else (c[pk] - c[s] >= drop and c[pk] - c[e] >= drop)
        if ok and e > s + 2:
            segs.append((int(grid[s]), int(grid[e]) + 1, int(grid[pk])))
    return segs


def segment_reps_auto(frames: np.ndarray, angles: np.ndarray, elb_2d: np.ndarray, fps: float, ex: str):
    if len(frames) < 6 or fps <= 0:
        return []
    grid, dist = np.arange(frames[0], frames[-1] + 1, 2), max(2, int(0.65 * fps / 2))
    if ex == "curl":
        cands = []
        for s in (0, 1):
            sig = 0.5 * angles[:, s] + 0.5 * elb_2d[:, s]
            if _rom(sig) >= 20.0:
                cands.extend(_find_1d_segs(grid, sig, frames, True, 15.0, 13.0, dist))
        cands.sort(key=lambda t: t[2])
        merged = []
        for seg in cands:
            if merged and abs(seg[2] - merged[-1][2]) <= int(0.55 * fps):
                p = merged[-1]
                merged[-1] = (min(p[0], seg[0]), max(p[1], seg[1]), (p[2] + seg[2]) // 2)
            else:
                merged.append(seg)
        segs = [(s, e) for s, e, _ in merged]
    else:
        if ex == "squat":
            sig, inv, pr, dr = angles[:, 12:14].mean(1), True, 20.0, 16.0
        elif ex == "lunge":
            sig, inv, pr, dr = angles[:, 12:14].min(1), True, 22.0, 18.0
        elif ex == "push":
            sig, inv, pr, dr = angles[:, :2].min(1), True, 15.0, 14.0
        elif ex in ("shoulder", "arm_abduction"):
            sig, inv, pr, dr = angles[:, 2:4].max(1), False, 20.0, 16.0
        else:  # row / deadlift
            use_row = max(_rom(angles[:, 0]), _rom(angles[:, 1])) >= max(32.0, 0.7 * _rom(angles[:, 4:6].mean(1)))
            sig, inv, pr, dr = (angles[:, :2].min(1), True, 16.0, 14.0) if use_row else (angles[:, 4:6].mean(1), True, 20.0, 16.0)
        segs = [(s, e) for s, e, _ in _find_1d_segs(grid, sig, frames, inv, pr, dr, dist)]
    if not segs and (frames[-1] - frames[0]) >= int(0.8 * fps):
        segs.append((int(frames[0]), int(frames[-1]) + 1))
    return segs


def diagnose_fixes(ex: str, seq: np.ndarray, ml_bad: bool | None, hips_vis: bool):
    l_e, r_e, l_s, r_s, torso, b_line, l_k, r_k = seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3], seq[:, 6], seq[:, 11], seq[:, 12], seq[:, 13]
    cues = []
    if ex == "squat":
        if hips_vis and np.percentile(torso, 90) > 42: cues.append("Keep chest up & back straight")
        if np.min(0.5 * (l_k + r_k)) > 110: cues.append("Go deeper on your squat")
    elif ex == "push":
        if hips_vis and np.min(b_line) < 150: cues.append("Brace core & keep back straight")
        if np.min(np.minimum(l_e, r_e)) > 106: cues.append("Lower chest closer to floor")
        if np.percentile(0.5 * (l_s + r_s), 85) > 78: cues.append("Tuck elbows in")
    elif ex == "curl":
        ce, cs = (l_e, l_s) if _rom(l_e) >= _rom(r_e) else (r_e, r_s)
        if hips_vis and _rom(torso) > 18: cues.append("Keep back straight, avoid swinging")
        if np.percentile(cs, 85) > 52: cues.append("Pin elbows to your sides")
        if np.min(ce) > 108: cues.append("Curl higher for full ROM")
    elif ex == "shoulder":
        if hips_vis and _rom(torso) > 18: cues.append("Brace core, avoid leaning back")
        if np.max(np.maximum(l_e, r_e)) < 136: cues.append("Press all the way overhead")
    elif ex == "row":
        if max(_rom(l_e), _rom(r_e)) >= 32:
            if hips_vis and np.mean(torso) < 22: cues.append("Hinge forward more with flat back")
            if np.min(np.minimum(l_e, r_e)) > 115: cues.append("Pull elbows higher")
        elif np.max(0.5 * (seq[:, 4] + seq[:, 5])) < 150:
            cues.append("Lock out hips cleanly at the top")
    elif ex == "lunge":
        if hips_vis and np.percentile(torso, 90) > 32: cues.append("Keep torso upright")
        if np.min(np.minimum(l_k, r_k)) > 108: cues.append("Drop back knee lower")
    elif ex == "arm_abduction":
        if np.max(np.maximum(l_s, r_s)) < 70: cues.append("Raise arms to shoulder height")
        if np.median(l_e if _rom(l_s) >= _rom(r_s) else r_e) < 120: cues.append("Keep soft elbow bend")
    is_bad = (len(cues) > 0) if ml_bad is None else ml_bad
    if is_bad and not cues:
        cues.append("Control tempo & alignment")
    return is_bad, (["Great rep! Clean form."] if not is_bad else cues[:2])


def predict_and_render(vpath: Path):
    frames, angles, joints, elb2d, arm_vis, hips_flags, lower_flags = [], [], [], [], [], [], []
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    with mp.tasks.vision.PoseLandmarker.create_from_options(POSE_OPTS) as det:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % 2 == 0:
                h, w = fr.shape[:2]
                s = min(1.0, 720.0 / max(h, w))
                rs = cv2.resize(fr, (int(w * s), int(h * s)))
                res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)))
                out = extract_frame_pose(res, rs.shape[1], rs.shape[0])
                if out is not None:
                    frames.append(idx); angles.append(out[0]); joints.append(out[1]); elb2d.append(out[2])
                    arm_vis.append(out[3]); hips_flags.append(out[4]); lower_flags.append(out[5])
            idx += 1
    cap.release()
    if len(frames) < 6:
        print(f"No valid pose frames in {vpath.name} ({len(frames)} valid). Ensure clear visibility.")
        return

    f_arr, a_arr, j_arr = np.asarray(frames, np.int32), np.asarray(angles, np.float32), np.asarray(joints, np.float32)
    e_arr, av_arr = np.asarray(elb2d, np.float32), np.asarray(arm_vis, np.float32)
    hips_med, lower_med = bool(np.mean(hips_flags) >= 0.5), bool(np.mean(lower_flags) >= 0.5)
    ex = auto_detect_exercise(a_arr, e_arr, av_arr, lower_med) if EXERCISE_MODE == "auto" else EXERCISE_MODE
    bundle = load_exercise_bundle(ex)
    segs = segment_reps_auto(f_arr, a_arr, e_arr, fps, ex)
    model_label = bundle["meta"].get("winner_model") or ("Loaded ML" if (bundle["tab"] or bundle["deep"]) else "Calibrated Biomechanics")

    rows = []
    t_new = np.linspace(0.0, 1.0, SEQ_LEN)
    for rep_i, (st, en) in enumerate(segs, 1):
        m = (f_arr >= st) & (f_arr < en)
        if m.sum() < 3:
            continue
        t_old = np.linspace(0.0, 1.0, int(m.sum()))
        ang_seq = np.stack([np.interp(t_new, t_old, a_arr[m, c]) for c in range(14)], -1).astype(np.float32)
        jf = j_arr[m].reshape(int(m.sum()), -1)
        ji = np.stack([np.interp(t_new, t_old, jf[:, c]) for c in range(jf.shape[1])], -1).reshape(SEQ_LEN, 12, 3)
        skel = np.concatenate([ji, np.diff(ji, axis=0, prepend=ji[:1])], -1).astype(np.float32)

        probs = []
        if bundle["tab"] is not None:
            probs.append((0.45, float(bundle["tab"].predict_proba(_tab_features(ang_seq[None]))[0, 1])))
        if bundle["deep"] is not None:
            ai = bundle["ang_sc"].transform(ang_seq).reshape(1, SEQ_LEN, 14) if bundle["ang_sc"] else ang_seq[None]
            si = bundle["skel_sc"].transform(skel.reshape(-1, 6)).reshape(1, SEQ_LEN, 12, 6) if bundle["skel_sc"] else skel[None]
            bd = str(bundle["meta"].get("best_deep_model", ""))
            inp = [si, ai] if ("2D-CNN + BiGRU" in bd or len(bundle["deep"].inputs) == 2) else (si if "2D CNN" in bd else ai)
            probs.append((0.55, float(np.nan_to_num(bundle["deep"].predict(inp, verbose=0).ravel()[0], nan=0.5))))
        p_bad = (sum(w * p for w, p in probs) / sum(w for w, _ in probs)) if probs else None
        is_bad, cues = diagnose_fixes(ex, ang_seq, (p_bad >= float(bundle["meta"].get("threshold", 0.5))) if p_bad is not None else None, hips_med)
        rows.append({"Rep": rep_i, "Time": f"{st/fps:.1f}s-{en/fps:.1f}s", "Verdict": "INCORRECT" if is_bad else "CORRECT", "Prob": round(p_bad, 2) if p_bad is not None else "--", "Coaching Cues": " | ".join(cues), "_bad": is_bad, "_st": st, "_en": en, "_cues": cues})

    display(Markdown(f"### `{vpath.name}` -> Detected: **{ex.upper()}** | Model: **{model_label}** | Reps: **{len(rows)}**"))
    if rows:
        display(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]))

    raw_mp4, out_mp4 = WORK_DIR / f"{vpath.stem}_raw.mp4", OUT_DIR / f"{vpath.stem}_{ex}_coach.mp4"
    cap, writer, f_i, done, last_lms = cv2.VideoCapture(str(vpath)), None, 0, 0, None
    with mp.tasks.vision.PoseLandmarker.create_from_options(POSE_OPTS) as det:
        while True:
            ok, fr = cap.read()
            if not ok: break
            if writer is None:
                s = min(1.0, 854.0 / max(fr.shape[:2]))
                sz = (max(2, int(fr.shape[1] * s) // 2 * 2), max(2, int(fr.shape[0] * s) // 2 * 2))
                writer = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, sz)
            fr = cv2.resize(fr, sz)
            w, h = sz
            if f_i % 2 == 0:
                res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                last_lms = res.pose_landmarks[0] if res.pose_landmarks else None
            while done < len(rows) and f_i >= rows[done]["_en"] - 1:
                done += 1
            active = next((i for i, r in enumerate(rows) if r["_st"] <= f_i < r["_en"]), None)
            info = rows[active] if active is not None else (rows[done - 1] if done > 0 else None)
            bad = bool(info and info["_bad"])
            acc = (50, 60, 245) if bad else (60, 225, 95)
            if last_lms is not None:
                pts = {i: (int(p.x * w), int(p.y * h)) for i, p in enumerate(last_lms) if p.visibility >= 0.25 and 0 <= p.x <= 1 and 0 <= p.y <= 1}
                for a, b in JOINT_LINKS:
                    if a in pts and b in pts: cv2.line(fr, pts[a], pts[b], acc, 2, cv2.LINE_AA)
                for pt in pts.values(): cv2.circle(fr, pt, 4, (50, 220, 255), -1, cv2.LINE_AA)
            cv2.rectangle(fr, (0, 0), (w, 46), (15, 18, 25), -1)
            cv2.putText(fr, f"{ex.upper()} REPS: {done}/{len(rows)} | {info['Verdict'] if info else 'TRACKING'}", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, acc, 2, cv2.LINE_AA)
            if info:
                cv2.rectangle(fr, (0, h - 42), (w, h), (15, 18, 25), -1)
                cv2.putText(fr, f"Coach: {info['_cues'][0][:75]}", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.56, acc, 2, cv2.LINE_AA)
            writer.write(fr); f_i += 1
    cap.release()
    if writer: writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_mp4), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)], check=True)
    raw_mp4.unlink(missing_ok=True)
    display(Video(str(out_mp4), embed=True, width=700))


if files is not None:
    print(f"Upload 1 or more exercise videos to test (mode={EXERCISE_MODE}):")
    for fname in files.upload():
        predict_and_render(Path(fname))
else:
    print("Run inside Google Colab to upload videos via files.upload(), or call predict_and_render(Path('video.mp4')).")
