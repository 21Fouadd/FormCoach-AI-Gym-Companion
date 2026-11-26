"""MediaPipe Pose Engine, Rep Counter, Trained Model Loader, Voice-Cue Diagnosis & HUD Renderer.

Feature extraction is 1-to-1 compatible with `colab/GymForm_MultiModel_Benchmark_Colab.ipynb`.
Supports all 7 exercises + Stage-1 Auto-Detection in both full-body and chest-up webcam views:
  - squat, push, curl, shoulder (OHP), row (Bent-Over Row / Deadlift), lunge, arm_abduction
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)

# Compatibility shim for scikit-learn 1.6.x HistGradientBoosting pickles loaded on 1.9.x
try:
    import sklearn._loss._loss as _sk_loss
    sys.modules.setdefault("_loss", _sk_loss)
except Exception:
    pass

import cv2
import joblib
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks, savgol_filter

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

POSE_TASK_PATH = MODELS_ROOT / "pose_landmarker_full.task"
POSE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
JOINT_LINKS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]
FEATURES = [
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "torso_tilt", "shoulder_line_tilt",
    "hip_line_tilt", "left_wrist_height", "right_wrist_height",
    "body_line", "left_knee", "right_knee",
]
SUPPORTED_EXERCISES = ["auto", "squat", "push", "curl", "shoulder", "row", "lunge", "arm_abduction"]
SEQUENCE_LENGTH = 32


def _rom_p90_p10(arr_1d: np.ndarray) -> float:
    if len(arr_1d) == 0:
        return 0.0
    return float(np.percentile(arr_1d, 90) - np.percentile(arr_1d, 10))


def auto_detect_exercise(
    angles_arr: np.ndarray,
    lower_body_visible: bool = True,
    elb_2d_arr: np.ndarray | None = None,
    arm_vis_arr: np.ndarray | None = None,
    require_motion: bool = False,
) -> str | None:
    """Stage-1 biomechanical classifier across all 7 supported exercises.

    When `require_motion=True` (used in live streaming mode), returns `None` while the user is
    standing idle so `auto` never locks onto a default exercise before movement begins.
    """
    if len(angles_arr) == 0:
        return None if require_motion else "squat"

    med_torso = float(np.median(angles_arr[:, 6]))
    p75_torso = float(np.percentile(angles_arr[:, 6], 75))
    p90_torso = float(np.percentile(angles_arr[:, 6], 90))
    avg_hip = 0.5 * (angles_arr[:, 4] + angles_arr[:, 5])
    p75_hip = float(np.percentile(avg_hip, 75))
    hip_rom = _rom_p90_p10(avg_hip)

    if arm_vis_arr is not None and len(arm_vis_arr) == len(angles_arr):
        l_ok = float(np.median(arm_vis_arr[:, 0])) >= 0.22
        r_ok = float(np.median(arm_vis_arr[:, 1])) >= 0.22
        if not (l_ok or r_ok):
            l_ok = float(np.median(arm_vis_arr[:, 0])) >= float(np.median(arm_vis_arr[:, 1]))
            r_ok = not l_ok
    else:
        l_ok, r_ok = True, True

    active_sides = [s for s, ok in ((0, l_ok), (1, r_ok)) if ok]
    p90_wh = max(float(np.percentile(angles_arr[:, 9 + s], 90)) for s in active_sides)
    p90_sho = max(float(np.percentile(angles_arr[:, 2 + s], 90)) for s in active_sides)
    sho_rom = max(_rom_p90_p10(angles_arr[:, 2 + s]) for s in active_sides)

    elb_rom_3d = max(_rom_p90_p10(angles_arr[:, s]) for s in active_sides)
    if elb_2d_arr is not None and len(elb_2d_arr) == len(angles_arr):
        elb_rom = max(
            elb_rom_3d,
            max(_rom_p90_p10(0.5 * angles_arr[:, s] + 0.5 * elb_2d_arr[:, s]) for s in active_sides),
        )
    else:
        elb_rom = elb_rom_3d

    min_hip_curve = np.minimum(angles_arr[:, 4], angles_arr[:, 5])
    p10_min_hip = float(np.percentile(min_hip_curve, 10))
    max_hip_rom = max(_rom_p90_p10(angles_arr[:, 4]), _rom_p90_p10(angles_arr[:, 5]))
    min_knee_curve = np.minimum(angles_arr[:, 12], angles_arr[:, 13])
    avg_knee_curve = 0.5 * (angles_arr[:, 12] + angles_arr[:, 13])
    knee_rom = max(_rom_p90_p10(avg_knee_curve), _rom_p90_p10(min_knee_curve))
    p10_min_knee = float(np.percentile(min_knee_curve, 10))
    p10_avg_knee = float(np.percentile(avg_knee_curve, 10))
    hip_split_p85 = float(np.percentile(np.abs(angles_arr[:, 4] - angles_arr[:, 5]), 85))

    # 1. Push-Up: horizontal body plank (torso horizontal AND hips straight on the SAME frames)
    plank_ratio = float(np.mean((angles_arr[:, 6] > 58.0) & (avg_hip > 128.0)))
    min_torso = float(np.min(angles_arr[:, 6]))
    if plank_ratio >= 0.30 or (med_torso > 78.0 and min_torso > 50.0 and p75_hip > 120.0):
        if not require_motion or elb_rom >= 16.0:
            return "push"
        return None

    # If athlete is currently in a horizontal straight-body plank, wait for push-up motion
    if require_motion and float(np.mean((angles_arr[-5:, 6] > 58.0) & (avg_hip[-5:] > 125.0))) >= 0.6:
        return None

    # 2. Shoulder Press / OHP: wrists press clearly above shoulders + high shoulder elevation
    if p90_wh > 0.22 and p90_sho > 96.0:
        if not require_motion or (sho_rom >= 20.0 or elb_rom >= 20.0):
            return "shoulder"

    # 3. Lower-body exercises (only when hips & knees are genuinely visible in camera frame)
    if lower_body_visible:
        # Bent-Over Row / Deadlift: forward hinge (22..86 deg, non-plank) while knees stay relatively high
        if p90_torso > 42.0 and 22.0 < med_torso < 86.0 and p10_avg_knee > 112.0:
            if not require_motion or (hip_rom >= 24.0 or elb_rom >= 24.0):
                return "row"

        # Squat vs. Lunge: genuine knee + hip flexion while torso remains upright/semi-upright (< 48 deg)
        if med_torso < 48.0 and knee_rom >= 26.0 and p10_min_knee < 128.0 and p10_min_hip < 132.0 and max_hip_rom >= 20.0:
            if hip_split_p85 >= 32.0:
                return "lunge"
            return "squat"

    # 4. Arm Abduction (Lateral Raise): upright torso (< 35 deg), wide shoulder abduction to shoulder level
    if med_torso < 35.0 and p90_wh <= 0.20 and sho_rom >= 34.0 and p90_sho >= 62.0 and sho_rom > 0.85 * elb_rom_3d:
        return "arm_abduction"

    # 5. Bicep Curl: elbow flexion/extension with upper arms staying low (< 58 deg shoulder elevation)
    if elb_rom >= (34.0 if require_motion else 24.0) and p90_sho < 58.0:
        return "curl"

    if require_motion:
        return None
    return "squat" if lower_body_visible else "curl"


def ensure_pose_task() -> Path:
    if not POSE_TASK_PATH.exists() or POSE_TASK_PATH.stat().st_size < 100_000:
        tmp = POSE_TASK_PATH.with_suffix(".part")
        urllib.request.urlretrieve(POSE_TASK_URL, tmp)
        tmp.replace(POSE_TASK_PATH)
    return POSE_TASK_PATH


def create_pose_landmarker():
    model_path = ensure_pose_task()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    cosine = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def extract_pose_from_result(result, img_w: int, img_h: int) -> dict[str, Any] | None:
    """Returns angles(14), norm_joints(12,3), person_box, landmarks_norm, 2D arm angles, and visibility metadata."""
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return None
    lms_2d = result.pose_landmarks[0]
    visible_pts = [
        (lm.x, lm.y) for lm in lms_2d
        if lm.visibility >= 0.30 and 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0
    ]
    if len(visible_pts) < 4:
        return None

    xs = [pt[0] * img_w for pt in visible_pts]
    ys = [pt[1] * img_h for pt in visible_pts]
    pad_x = max(16.0, (max(xs) - min(xs)) * 0.14)
    pad_y = max(20.0, (max(ys) - min(ys)) * 0.10)
    person_box = (
        int(max(0, min(xs) - pad_x)),
        int(max(0, min(ys) - pad_y)),
        int(min(img_w - 1, max(xs) + pad_x)),
        int(min(img_h - 1, max(ys) + pad_y)),
    )

    landmarks_norm = {
        str(i): [round(float(lms_2d[i].x), 4), round(float(lms_2d[i].y), 4), round(float(lms_2d[i].visibility), 3)]
        for i in CORE_JOINTS
        if lms_2d[i].visibility >= 0.22 and -0.05 <= lms_2d[i].x <= 1.05 and -0.05 <= lms_2d[i].y <= 1.05
    }

    # Accept full-body, single-side (side-view), AND chest-up webcam arm chains (hips off-screen!)
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

    left_arm_vis = float(min(lms_2d[11].visibility, lms_2d[13].visibility, lms_2d[15].visibility))
    right_arm_vis = float(min(lms_2d[12].visibility, lms_2d[14].visibility, lms_2d[16].visibility))
    hips_vis = float(max(lms_2d[23].visibility, lms_2d[24].visibility))
    knees_vis = float(max(lms_2d[25].visibility, lms_2d[26].visibility))
    hips_visible = hips_vis >= 0.35
    lower_body_visible = hips_visible and (knees_vis >= 0.30)

    if not (left_lower or right_lower or left_upper or right_upper or left_arm_only or right_arm_only):
        return {
            "person_box": person_box,
            "landmarks_norm": landmarks_norm,
            "angles": None,
            "norm_joints": None,
            "elb_2d": None,
            "arm_vis": (left_arm_vis, right_arm_vis),
            "hips_visible": hips_visible,
            "lower_body_visible": lower_body_visible,
        }

    p = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]], dtype=np.float32)
    shoulders = (p[11] + p[12]) / 2.0
    hips = (p[23] + p[24]) / 2.0
    raw_torso_scale = float(np.linalg.norm(shoulders - hips))
    if hips_visible and raw_torso_scale >= 0.12:
        scale = raw_torso_scale
    else:
        shoulder_span = float(np.linalg.norm(p[11] - p[12]))
        scale = max(raw_torso_scale, shoulder_span * 1.35, 0.20)
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

    p2 = np.array([[lm.x * img_w, lm.y * img_h] for lm in lms_2d], dtype=np.float32)
    elb_2d = np.array([
        calc_angle(p2[11], p2[13], p2[15]),
        calc_angle(p2[12], p2[14], p2[16]),
    ], dtype=np.float32)

    norm_joints = ((p[CORE_JOINTS] - hips) / scale).astype(np.float32)
    if not (np.isfinite(angles).all() and np.isfinite(norm_joints).all()):
        return None

    return {
        "person_box": person_box,
        "landmarks_norm": landmarks_norm,
        "angles": angles,
        "norm_joints": norm_joints,
        "elb_2d": elb_2d,
        "arm_vis": (left_arm_vis, right_arm_vis),
        "hips_visible": hips_visible,
        "lower_body_visible": lower_body_visible,
    }


def build_tabular_features(angle_seqs: np.ndarray) -> np.ndarray:
    """Exact match to Colab notebook phase-aware + asymmetry + velocity tabular features."""
    g_mean = angle_seqs.mean(axis=1)
    g_std = angle_seqs.std(axis=1)
    g_min = angle_seqs.min(axis=1)
    g_max = angle_seqs.max(axis=1)
    g_rom = g_max - g_min

    p1 = angle_seqs[:, :11, :]
    p2 = angle_seqs[:, 11:22, :]
    p3 = angle_seqs[:, 22:, :]
    phase_feats = np.concatenate([
        p1.mean(axis=1), p2.mean(axis=1), p2.min(axis=1), p2.max(axis=1), p3.mean(axis=1)
    ], axis=1)

    vel = np.diff(angle_seqs, axis=1)
    v_abs_mean = np.abs(vel).mean(axis=1)
    v_max = np.abs(vel).max(axis=1)

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


def resample_sequence(angles_arr: np.ndarray, joints_arr: np.ndarray, seq_len: int = SEQUENCE_LENGTH):
    n = len(angles_arr)
    t_old = np.linspace(0.0, 1.0, n)
    t_new = np.linspace(0.0, 1.0, seq_len)
    ang_seq = np.stack([np.interp(t_new, t_old, angles_arr[:, c]) for c in range(angles_arr.shape[1])], axis=-1)
    j_flat = joints_arr.reshape(n, -1)
    j_seq = np.stack([np.interp(t_new, t_old, j_flat[:, c]) for c in range(j_flat.shape[1])], axis=-1)
    j_seq = j_seq.reshape(seq_len, len(CORE_JOINTS), 3)
    vel = np.diff(j_seq, axis=0, prepend=j_seq[:1])
    skel_grid = np.concatenate([j_seq, vel], axis=-1).astype(np.float32)
    return ang_seq.astype(np.float32), skel_grid


def diagnose_voice_cues(
    exercise: str,
    angle_seq: np.ndarray,
    ml_incorrect: bool | None = None,
    hips_visible: bool = True,
    active_arm: int | None = None,
    prob_incorrect: float | None = None,
    threshold: float = 0.5,
) -> tuple[bool, list[str]]:
    """Produces detailed, angle-specific coaching cues + hybrid ML/biomechanical verdict."""
    # Smooth raw angle sequence with Savitzky-Golay (matches training preprocessing & removes webcam jitter)
    seq = angle_seq.astype(np.float32)
    if len(seq) >= 7:
        try:
            seq = savgol_filter(seq, window_length=7, polyorder=2, axis=0).astype(np.float32)
        except Exception:
            pass

    cues: list[str] = []
    severe_fault = False

    l_elb, r_elb = seq[:, 0], seq[:, 1]
    l_sho, r_sho = seq[:, 2], seq[:, 3]
    l_hip, r_hip = seq[:, 4], seq[:, 5]
    torso_tilt = seq[:, 6]
    sho_tilt = seq[:, 7]
    hip_tilt = seq[:, 8]
    body_line = seq[:, 11]
    l_knee, r_knee = seq[:, 12], seq[:, 13]
    avg_knee = 0.5 * (l_knee + r_knee)
    avg_elb = 0.5 * (l_elb + r_elb)
    avg_sho = 0.5 * (l_sho + r_sho)
    avg_hip = 0.5 * (l_hip + r_hip)

    positive_msg = "Great rep! Clean form and controlled tempo."

    if exercise == "squat":
        p90_torso = float(np.percentile(torso_tilt, 90))
        min_knee = float(np.min(avg_knee))
        p85_hip_tilt = float(np.percentile(hip_tilt, 85))
        knee_asym = float(np.mean(np.abs(l_knee - r_knee)))
        peak_vel = float(np.max(np.abs(np.diff(avg_knee)))) if len(avg_knee) > 2 else 0.0

        if hips_visible and p90_torso > 42.0:
            cues.append(f"Chest leaning too far forward ({p90_torso:.0f}° torso tilt) — brace your core and keep your chest proud")
            if p90_torso > 52.0:
                severe_fault = True
        if min_knee > 110.0:
            cues.append(f"Squat depth is shallow (min knee {min_knee:.0f}°) — sit hips lower toward parallel (~95°)")
            if min_knee > 125.0:
                severe_fault = True
        if hips_visible and p85_hip_tilt > 12.0:
            cues.append(f"Hips shifting unevenly ({p85_hip_tilt:.0f}° tilt) — drive evenly through both heels")
        if knee_asym > 18.0:
            side = "left" if float(np.min(l_knee)) > float(np.min(r_knee)) else "right"
            cues.append(f"Uneven knee tracking ({knee_asym:.0f}° gap) — align your {side} knee evenly over toes")
        if peak_vel > 22.0 and not cues:
            cues.append("Control your descent speed — lower smoothly for ~2 seconds instead of dropping")
        positive_msg = f"Great squat! Solid depth ({min_knee:.0f}° knee) and upright torso."

    elif exercise == "push":
        min_body = float(np.min(body_line))
        min_elb = float(np.min(np.minimum(l_elb, r_elb)))
        p85_sho = float(np.percentile(avg_sho, 85))
        p85_sho_tilt = float(np.percentile(sho_tilt, 85))

        if hips_visible and min_body < 150.0:
            cues.append(f"Hips sagging or back arching ({min_body:.0f}° body line) — squeeze glutes and brace your core straight")
            if min_body < 138.0:
                severe_fault = True
        if min_elb > 106.0:
            cues.append(f"Partial push-up depth (min elbow {min_elb:.0f}°) — lower your chest closer to the floor (~90°)")
            if min_elb > 122.0:
                severe_fault = True
        if p85_sho > 78.0:
            cues.append(f"Elbows flaring wide ({p85_sho:.0f}° shoulder angle) — tuck elbows ~45° toward your ribs")
        if p85_sho_tilt > 14.0:
            cues.append(f"Uneven push-up press ({p85_sho_tilt:.0f}° shoulder tilt) — push evenly through both hands")
        positive_msg = f"Clean push-up! Strong depth ({min_elb:.0f}° elbow) and straight plank line."

    elif exercise == "curl":
        if active_arm == 0:
            cur_elb, cur_sho, arm_lbl = l_elb, l_sho, "left"
        elif active_arm == 1:
            cur_elb, cur_sho, arm_lbl = r_elb, r_sho, "right"
        else:
            if _rom_p90_p10(l_elb) >= _rom_p90_p10(r_elb):
                cur_elb, cur_sho, arm_lbl = l_elb, l_sho, "left"
            else:
                cur_elb, cur_sho, arm_lbl = r_elb, r_sho, "right"

        torso_swing = float(np.percentile(torso_tilt, 90) - np.percentile(torso_tilt, 10))
        p90_torso = float(np.percentile(torso_tilt, 90))
        p85_sho = float(np.percentile(cur_sho, 85))
        min_elb = float(np.min(cur_elb))
        max_elb = float(np.max(cur_elb))
        peak_vel = float(np.max(np.abs(np.diff(cur_elb)))) if len(cur_elb) > 2 else 0.0

        if hips_visible and (torso_swing > 18.0 or p90_torso > 28.0):
            cues.append(f"Torso swinging for momentum ({max(torso_swing, p90_torso):.0f}° lean) — brace abs and keep back still")
            if torso_swing > 26.0 or p90_torso > 35.0:
                severe_fault = True
        if p85_sho > 52.0:
            cues.append(f"{arm_lbl.capitalize()} elbow drifting forward ({p85_sho:.0f}° shoulder) — pin your upper arm to your ribs")
            if p85_sho > 68.0:
                severe_fault = True
        if min_elb > 108.0:
            cues.append(f"Incomplete peak squeeze on {arm_lbl} arm (min elbow {min_elb:.0f}°) — curl higher toward your shoulder")
            if min_elb > 122.0:
                severe_fault = True
        if max_elb < 132.0:
            cues.append(f"Incomplete bottom extension ({max_elb:.0f}° max elbow) — lower weight smoothly to full stretch")
        if peak_vel > 28.0 and not cues:
            cues.append("Rep tempo is too fast — pause 1s at the top and lower with control")
        positive_msg = f"Clean {arm_lbl} curl! Full squeeze ({min_elb:.0f}° elbow) with steady upper arm."

    elif exercise == "shoulder":
        torso_swing = float(np.percentile(torso_tilt, 90) - np.percentile(torso_tilt, 10))
        p90_torso = float(np.percentile(torso_tilt, 90))
        max_elb = float(np.max(np.maximum(l_elb, r_elb)))
        p85_sho_tilt = float(np.percentile(sho_tilt, 85))

        if hips_visible and (torso_swing > 18.0 or p90_torso > 24.0):
            cues.append(f"Lower back arching / leaning back ({max(torso_swing, p90_torso):.0f}° torso tilt) — squeeze glutes and ribs down")
            if p90_torso > 32.0:
                severe_fault = True
        if max_elb < 136.0:
            cues.append(f"Incomplete overhead lockout (max elbow {max_elb:.0f}°) — press straight up to full extension (~165°)")
            if max_elb < 120.0:
                severe_fault = True
        if p85_sho_tilt > 15.0 and min(_rom_p90_p10(l_sho), _rom_p90_p10(r_sho)) > 35.0:
            cues.append(f"Uneven overhead press ({p85_sho_tilt:.0f}° shoulder tilt) — drive both arms up evenly")
        positive_msg = f"Strong overhead press! Full lockout ({max_elb:.0f}° elbow) and braced core."

    elif exercise == "row":
        elb_rom = max(_rom_p90_p10(l_elb), _rom_p90_p10(r_elb))
        hip_rom = _rom_p90_p10(avg_hip)
        if elb_rom >= 32.0 and elb_rom >= 0.70 * hip_rom:
            mean_torso = float(np.mean(torso_tilt))
            min_elb = float(np.min(np.minimum(l_elb, r_elb)))
            p85_sho = float(np.percentile(np.maximum(l_sho, r_sho), 85))
            if hips_visible and mean_torso < 22.0:
                cues.append(f"Torso too upright ({mean_torso:.0f}° hinge) — hinge forward ~45° with a flat back")
            if min_elb > 115.0:
                cues.append(f"Short pull range (min elbow {min_elb:.0f}°) — drive elbows higher past your ribs and squeeze lats")
            if p85_sho > 78.0:
                cues.append(f"Elbows flaring wide ({p85_sho:.0f}°) — keep elbows tucked closer to your sides")
            positive_msg = f"Great row! Strong back pull ({min_elb:.0f}° elbow) and solid hip hinge."
        else:
            max_hip = float(np.max(avg_hip))
            p85_hip_tilt = float(np.percentile(hip_tilt, 85))
            if max_hip < 150.0:
                cues.append(f"Incomplete hip lockout (max hip {max_hip:.0f}°) — squeeze glutes and stand tall at the top")
            if hips_visible and p85_hip_tilt > 14.0:
                cues.append(f"Hips tilting sideways ({p85_hip_tilt:.0f}°) — keep pelvis square throughout the pull")
            positive_msg = f"Clean hinge/deadlift! Full hip lockout ({max_hip:.0f}°) and balanced stance."

    elif exercise == "lunge":
        p90_torso = float(np.percentile(torso_tilt, 90))
        min_knee = float(np.min(np.minimum(l_knee, r_knee)))
        p85_hip_tilt = float(np.percentile(hip_tilt, 85))
        if hips_visible and p90_torso > 32.0:
            cues.append(f"Torso leaning forward ({p90_torso:.0f}° tilt) — keep shoulders stacked tall over your hips")
            if p90_torso > 42.0:
                severe_fault = True
        if min_knee > 108.0:
            cues.append(f"Shallow lunge depth (min knee {min_knee:.0f}°) — drop your back knee lower toward ~90°")
            if min_knee > 122.0:
                severe_fault = True
        if hips_visible and p85_hip_tilt > 14.0:
            cues.append(f"Pelvis tilting sideways ({p85_hip_tilt:.0f}°) — stabilize core and keep hips level")
        positive_msg = f"Great lunge! Deep knee bend ({min_knee:.0f}°) and tall upright torso."

    elif exercise == "arm_abduction":
        torso_swing = float(np.percentile(torso_tilt, 90) - np.percentile(torso_tilt, 10))
        p90_torso = float(np.percentile(torso_tilt, 90))
        max_sho = float(np.max(np.maximum(l_sho, r_sho)))
        active_elb = l_elb if _rom_p90_p10(l_sho) >= _rom_p90_p10(r_sho) else r_elb
        med_elb = float(np.median(active_elb))

        if hips_visible and (torso_swing > 18.0 or p90_torso > 24.0):
            cues.append(f"Torso leaning or swinging ({max(torso_swing, p90_torso):.0f}° tilt) — keep core still and lower slowly")
            if torso_swing > 26.0:
                severe_fault = True
        if max_sho < 70.0:
            cues.append(f"Arms raised too low (peak shoulder {max_sho:.0f}°) — lift smoothly to 85–95° shoulder height")
            if max_sho < 55.0:
                severe_fault = True
        if med_elb < 120.0:
            cues.append(f"Elbows bent too much ({med_elb:.0f}°) — keep arms long with only a slight ~15° soft bend")
        positive_msg = f"Clean lateral raise! Smooth height ({max_sho:.0f}° shoulder) and controlled torso."

    # Hybrid ML + Biomechanical Confidence Fusion:
    # - High-confidence ML (>= thr+0.12) always flags Incorrect
    # - Borderline ML (within ±0.12 of thr) uses biomechanical fault cues as tiebreaker (cuts false alarms!)
    # - Severe biomechanical fault always flags Incorrect
    if ml_incorrect is None:
        is_incorrect = len(cues) > 0
    elif prob_incorrect is not None:
        if severe_fault or prob_incorrect >= (threshold + 0.12):
            is_incorrect = True
        elif prob_incorrect <= (threshold - 0.12):
            is_incorrect = False
        else:
            is_incorrect = len(cues) > 0
    else:
        is_incorrect = ml_incorrect or severe_fault

    if is_incorrect and not cues:
        # Give a specific kinematic cue even when ML caught a subtle timing/asymmetry fault
        asym_elb = float(np.mean(np.abs(l_elb - r_elb)))
        asym_knee = float(np.mean(np.abs(l_knee - r_knee)))
        if exercise in ("squat", "lunge") and asym_knee > 12.0:
            cues.append(f"Subtle leg asymmetry detected ({asym_knee:.0f}° L/R knee gap) — balance weight evenly")
        elif exercise in ("push", "curl", "shoulder", "arm_abduction") and asym_elb > 14.0:
            cues.append(f"Subtle arm asymmetry detected ({asym_elb:.0f}° L/R gap) — move both sides smoothly together")
        else:
            cues.append("Subtle form or tempo drift detected — slow down the lowering phase and brace your core")

    if not is_incorrect:
        cues = [positive_msg]
    return is_incorrect, cues[:3]


class LoadedExerciseModel:
    """Loads exported Colab model bundle (`backend/models/<exercise>/`) with graceful fallback."""

    def __init__(self, exercise: str) -> None:
        self.exercise = exercise
        self.bundle_dir = MODELS_ROOT / exercise
        self.tab_model = None
        self.deep_model = None
        self.ang_scaler = None
        self.skel_scaler = None
        self.metadata: dict[str, Any] = {}
        self.mode = "biomechanical_fallback"
        self._try_load()

    def _try_load(self) -> None:
        if not self.bundle_dir.exists():
            return
        meta_file = self.bundle_dir / "model_metadata.json"
        if meta_file.exists():
            self.metadata = json.loads(meta_file.read_text(encoding="utf-8"))

        tab_path = self.bundle_dir / "best_tabular_model.joblib"
        if tab_path.exists():
            try:
                self.tab_model = joblib.load(tab_path)
            except Exception:
                self.tab_model = None

        ang_sc = self.bundle_dir / "angle_scaler.joblib"
        skel_sc = self.bundle_dir / "skel_scaler.joblib"
        if ang_sc.exists():
            self.ang_scaler = joblib.load(ang_sc)
        if skel_sc.exists():
            self.skel_scaler = joblib.load(skel_sc)

        keras_path = self.bundle_dir / "best_deep_model.keras"
        if keras_path.exists():
            try:
                import tensorflow as tf

                class TemporalAttentionPool(tf.keras.layers.Layer):
                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)
                        self.score_dense = tf.keras.layers.Dense(1)

                    def call(self, inputs):
                        w = tf.nn.softmax(self.score_dense(inputs), axis=1)
                        return tf.reduce_sum(inputs * w, axis=1)

                self.deep_model = tf.keras.models.load_model(
                    keras_path,
                    custom_objects={"TemporalAttentionPool": TemporalAttentionPool},
                )
            except Exception:
                self.deep_model = None

        if self.deep_model is not None and self.tab_model is not None:
            self.mode = f"ensemble ({self.metadata.get('best_deep_model', 'Deep')} + {self.metadata.get('best_tabular_model', 'Tabular')})"
        elif self.deep_model is not None:
            self.mode = self.metadata.get("best_deep_model", "deep_keras")
        elif self.tab_model is not None:
            self.mode = self.metadata.get("best_tabular_model", "tabular_joblib")

    def predict_rep(
        self,
        angle_seq: np.ndarray,
        skel_grid: np.ndarray,
        hips_visible: bool = True,
        active_arm: int | None = None,
    ) -> dict[str, Any]:
        probs: list[tuple[float, float]] = []

        # Smooth angles with Savitzky-Golay before tabular inference (matches Colab training!)
        smooth_seq = angle_seq.astype(np.float32)
        if len(smooth_seq) >= 7:
            try:
                smooth_seq = savgol_filter(smooth_seq, window_length=7, polyorder=2, axis=0).astype(np.float32)
            except Exception:
                pass

        if self.tab_model is not None:
            x_tab = build_tabular_features(smooth_seq[None])
            p_tab = float(self.tab_model.predict_proba(x_tab)[0, 1])
            probs.append((0.45, p_tab))

        if self.deep_model is not None:
            best_deep = self.metadata.get("best_deep_model", "")
            ang_in = (
                self.ang_scaler.transform(angle_seq).reshape(1, *angle_seq.shape)
                if self.ang_scaler is not None else angle_seq[None]
            )
            skel_in = (
                self.skel_scaler.transform(skel_grid.reshape(-1, skel_grid.shape[-1])).reshape(1, *skel_grid.shape)
                if self.skel_scaler is not None else skel_grid[None]
            )
            if "2D-CNN + BiGRU" in best_deep or len(self.deep_model.inputs) == 2:
                p_deep = float(self.deep_model.predict([skel_in, ang_in], verbose=0).ravel()[0])
            elif "2D CNN" in best_deep:
                p_deep = float(self.deep_model.predict(skel_in, verbose=0).ravel()[0])
            else:
                p_deep = float(self.deep_model.predict(ang_in, verbose=0).ravel()[0])
            probs.append((0.55, p_deep))

        threshold = float(self.metadata.get("threshold", 0.5))
        if probs:
            total_w = sum(w for w, _ in probs)
            prob_incorrect = sum(w * p for w, p in probs) / total_w
            ml_incorrect: bool | None = prob_incorrect >= threshold
        else:
            prob_incorrect = None
            ml_incorrect = None

        is_incorrect, cues = diagnose_voice_cues(
            self.exercise,
            smooth_seq,
            ml_incorrect=ml_incorrect,
            hips_visible=hips_visible,
            active_arm=active_arm,
            prob_incorrect=prob_incorrect,
            threshold=threshold,
        )
        if prob_incorrect is None:
            prob_incorrect = 0.82 if is_incorrect else 0.14

        return {
            "verdict": "Incorrect" if is_incorrect else "Correct",
            "is_incorrect": is_incorrect,
            "incorrect_probability": round(float(prob_incorrect), 3),
            "cues": cues,
            "model_used": self.mode,
        }


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, LoadedExerciseModel] = {}

    def get(self, exercise: str) -> LoadedExerciseModel:
        ex = exercise.lower().strip()
        if ex not in self._models:
            self._models[ex] = LoadedExerciseModel(ex)
        return self._models[ex]

    def summary(self) -> dict[str, str]:
        return {ex: self.get(ex).mode for ex in SUPPORTED_EXERCISES if ex != "auto"}


class LiveRepTracker:
    """Real-time streaming rep counter + live form cue generator for all 7 exercises + Auto-Detect."""

    def __init__(self, exercise: str) -> None:
        self.exercise = exercise.lower().strip()
        self.auto_mode = self.exercise == "auto"
        self.stage = "up"
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_angles: list[np.ndarray] = []
        self.current_joints: list[np.ndarray] = []
        self.current_hips_vis: list[bool] = []

        # Per-arm state for curls (handles simultaneous two-arm, alternating, and one-arm curls)
        self._arm_stage = ["up", "up"]
        self._arm_angles: list[list[np.ndarray]] = [[], []]
        self._arm_joints: list[list[np.ndarray]] = [[], []]
        self._arm_hips_vis: list[list[bool]] = [[], []]
        self._arm_last_done_ts = [-10.0, -10.0]

        # Dual sub-tracker state for row vs deadlift under `exercise == "row"`
        self._row_pull_stage = "up"
        self._row_pull_angles: list[np.ndarray] = []
        self._row_pull_joints: list[np.ndarray] = []
        self._deadlift_stage = "up"
        self._deadlift_angles: list[np.ndarray] = []
        self._deadlift_joints: list[np.ndarray] = []

        # Rolling auto-detection buffers
        self._auto_buf: list[np.ndarray] = []
        self._auto_joints: list[np.ndarray] = []
        self._auto_elb2d: list[np.ndarray] = []
        self._auto_meta: list[dict[str, Any]] = []

        self.last_verdict = "Ready"
        self.last_cues: list[str] = ["Step into frame to begin"]
        self.all_cues: list[str] = []
        self._last_cue_ts = 0.0

    def _record_completed_rep(
        self,
        ang_list: list[np.ndarray],
        joint_list: list[np.ndarray],
        model: LoadedExerciseModel,
        hips_visible: bool = True,
        active_arm: int | None = None,
    ) -> str:
        ang_seq, skel_grid = resample_sequence(
            np.asarray(ang_list, dtype=np.float32),
            np.asarray(joint_list, dtype=np.float32),
        )
        res = model.predict_rep(ang_seq, skel_grid, hips_visible=hips_visible, active_arm=active_arm)
        self.rep_count += 1
        if res["is_incorrect"]:
            self.incorrect_reps += 1
        else:
            self.correct_reps += 1
        self.last_verdict = res["verdict"]
        self.last_cues = res["cues"]
        self.all_cues.extend(res["cues"])
        self._last_cue_ts = time.monotonic()
        return res["cues"][0]

    def _step_single(
        self,
        angles: np.ndarray,
        norm_joints: np.ndarray,
        model: LoadedExerciseModel,
        elb_2d: np.ndarray | None,
        arm_vis: tuple[float, float],
        hips_visible: bool,
    ) -> tuple[float, bool, str | None]:
        newly_completed = False
        spoken_cue: str | None = None
        now = time.monotonic()

        # --- 1. BICEP CURL: independent Left/Right arm tracking with simultaneous merge ---
        if self.exercise == "curl":
            eff = [0.0, 0.0]
            for s in (0, 1):
                e3 = float(angles[s])
                e2 = float(elb_2d[s]) if elb_2d is not None else e3
                eff[s] = 0.5 * e3 + 0.5 * e2

            vis_ok = [arm_vis[0] >= 0.14, arm_vis[1] >= 0.14]
            if not any(vis_ok):
                vis_ok = [arm_vis[0] >= arm_vis[1], arm_vis[1] > arm_vis[0]]

            down_th, up_th = 104.0, 134.0
            for s in (0, 1):
                if not vis_ok[s]:
                    continue
                a_val = eff[s]
                if a_val < up_th:
                    self._arm_angles[s].append(angles)
                    self._arm_joints[s].append(norm_joints)
                    self._arm_hips_vis[s].append(hips_visible)
                    if a_val < down_th:
                        self._arm_stage[s] = "down"
                elif self._arm_stage[s] == "down" and len(self._arm_angles[s]) >= 3:
                    other = 1 - s
                    # Merge simultaneous two-arm curls finished within 0.65s of each other
                    if (now - self._arm_last_done_ts[other]) > 0.65:
                        hv = bool(np.mean(self._arm_hips_vis[s]) >= 0.5) if self._arm_hips_vis[s] else hips_visible
                        spoken_cue = self._record_completed_rep(
                            self._arm_angles[s], self._arm_joints[s], model, hips_visible=hv, active_arm=s
                        )
                        newly_completed = True
                    self._arm_last_done_ts[s] = now
                    self._arm_stage[s] = "up"
                    self._arm_angles[s].clear()
                    self._arm_joints[s].clear()
                    self._arm_hips_vis[s].clear()
                else:
                    if len(self._arm_angles[s]) > 120:
                        self._arm_angles[s] = self._arm_angles[s][-60:]
                        self._arm_joints[s] = self._arm_joints[s][-60:]
                        self._arm_hips_vis[s] = self._arm_hips_vis[s][-60:]

            self.stage = "down" if ("down" in self._arm_stage) else "up"
            visible_effs = [eff[s] for s in (0, 1) if vis_ok[s]]
            primary = float(min(visible_effs)) if visible_effs else float(min(eff))

            if not newly_completed and self.stage == "down" and (now - self._last_cue_ts) > 3.5:
                active_sho = max(
                    float(angles[2]) if vis_ok[0] else 0.0,
                    float(angles[3]) if vis_ok[1] else 0.0,
                )
                if active_sho > 55.0:
                    spoken_cue = "Keep your elbows pinned to your sides"
                    self.last_cues = [spoken_cue]
                    self._last_cue_ts = now
            return primary, newly_completed, spoken_cue

        # --- 2. BENT-OVER ROW / DEADLIFT: supports both elbow-pull rows and hip-hinge deadlifts ---
        if self.exercise == "row":
            vis_elbs = [float(angles[s]) for s in (0, 1) if arm_vis[s] >= 0.16]
            min_elb = min(vis_elbs) if vis_elbs else float(min(angles[0], angles[1]))
            avg_hip = float(0.5 * (angles[4] + angles[5]))
            torso = float(angles[6])

            # Sub-tracker A: Bent-Over Row (elbow pull while hinged forward)
            if torso > 20.0 and min_elb < 148.0:
                self._row_pull_angles.append(angles)
                self._row_pull_joints.append(norm_joints)
                if min_elb < 124.0:
                    self._row_pull_stage = "down"
            elif self._row_pull_stage == "down" and len(self._row_pull_angles) >= 2:
                spoken_cue = self._record_completed_rep(
                    self._row_pull_angles, self._row_pull_joints, model, hips_visible=hips_visible
                )
                newly_completed = True
                self._row_pull_stage = "up"
                self._row_pull_angles.clear()
                self._row_pull_joints.clear()
                self._deadlift_stage = "up"
                self._deadlift_angles.clear()
                self._deadlift_joints.clear()
            elif len(self._row_pull_angles) > 120:
                self._row_pull_angles = self._row_pull_angles[-60:]
                self._row_pull_joints = self._row_pull_joints[-60:]

            # Sub-tracker B: Deadlift / RDL (hip hinge down and stand tall with straight arms)
            if not newly_completed and min_elb > 125.0:
                if avg_hip < 148.0:
                    self._deadlift_angles.append(angles)
                    self._deadlift_joints.append(norm_joints)
                    if avg_hip < 122.0:
                        self._deadlift_stage = "down"
                elif self._deadlift_stage == "down" and len(self._deadlift_angles) >= 5:
                    spoken_cue = self._record_completed_rep(
                        self._deadlift_angles, self._deadlift_joints, model, hips_visible=hips_visible
                    )
                    newly_completed = True
                    self._deadlift_stage = "up"
                    self._deadlift_angles.clear()
                    self._deadlift_joints.clear()
                elif len(self._deadlift_angles) > 120:
                    self._deadlift_angles = self._deadlift_angles[-60:]
                    self._deadlift_joints = self._deadlift_joints[-60:]

            self.stage = "down" if (self._row_pull_stage == "down" or self._deadlift_stage == "down") else "up"
            primary = min_elb if self._row_pull_stage == "down" else avg_hip
            return primary, newly_completed, spoken_cue

        # --- 3. SQUAT, LUNGE, PUSH-UP, SHOULDER PRESS, ARM ABDUCTION ---
        if self.exercise == "squat":
            primary = float(0.5 * (angles[12] + angles[13]))
            in_active = primary < 155.0
            hit_peak = primary < 128.0
        elif self.exercise == "lunge":
            primary = float(min(angles[12], angles[13]))
            in_active = primary < 148.0
            hit_peak = primary < 118.0
        elif self.exercise == "push":
            if abs(arm_vis[0] - arm_vis[1]) > 0.25:
                dom = 0 if arm_vis[0] > arm_vis[1] else 1
                primary = float(angles[dom])
            else:
                primary = float(0.5 * (angles[0] + angles[1]))
            in_active = primary < 146.0
            hit_peak = primary < 115.0
        elif self.exercise == "shoulder":
            vis_shos = [float(angles[2 + s]) for s in (0, 1) if arm_vis[s] >= 0.15]
            primary = float(max(vis_shos)) if vis_shos else float(max(angles[2], angles[3]))
            max_wh = float(max(angles[9], angles[10]))
            in_active = primary > 104.0
            hit_peak = (primary > 124.0) and (max_wh > 0.22)
        elif self.exercise == "arm_abduction":
            vis_shos = [float(angles[2 + s]) for s in (0, 1) if arm_vis[s] >= 0.15]
            primary = float(max(vis_shos)) if vis_shos else float(max(angles[2], angles[3]))
            in_active = primary > 44.0
            hit_peak = primary > 68.0
        else:
            primary = float(0.5 * (angles[12] + angles[13]))
            in_active = primary < 150.0
            hit_peak = primary < 122.0

        if in_active:
            self.current_angles.append(angles)
            self.current_joints.append(norm_joints)
            self.current_hips_vis.append(hips_visible)
            if hit_peak:
                self.stage = "down"
        elif self.stage == "down" and len(self.current_angles) >= 3:
            hv = bool(np.mean(self.current_hips_vis) >= 0.5) if self.current_hips_vis else hips_visible
            spoken_cue = self._record_completed_rep(
                self.current_angles, self.current_joints, model, hips_visible=hv
            )
            newly_completed = True
            self.stage = "up"
            self.current_angles.clear()
            self.current_joints.clear()
            self.current_hips_vis.clear()
        else:
            if len(self.current_angles) > 120:
                self.current_angles = self.current_angles[-60:]
                self.current_joints = self.current_joints[-60:]
                self.current_hips_vis = self.current_hips_vis[-60:]

        # Instant mid-rep safety cue (throttled every 3.5s)
        if not newly_completed and self.stage == "down" and (now - self._last_cue_ts) > 3.5:
            if self.exercise == "squat" and hips_visible and float(angles[6]) > 46.0:
                spoken_cue = "Keep your back straight"
                self.last_cues = [spoken_cue]
                self._last_cue_ts = now
            elif self.exercise == "push" and hips_visible and float(angles[11]) < 148.0:
                spoken_cue = "Keep your back straight and core tight"
                self.last_cues = [spoken_cue]
                self._last_cue_ts = now

        return primary, newly_completed, spoken_cue

    def update(
        self,
        angles: np.ndarray,
        norm_joints: np.ndarray,
        model: LoadedExerciseModel,
        pose_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        elb_2d = pose_meta.get("elb_2d") if pose_meta else None
        arm_vis = pose_meta.get("arm_vis", (1.0, 1.0)) if pose_meta else (1.0, 1.0)
        hips_visible = bool(pose_meta.get("hips_visible", True)) if pose_meta else True
        lower_body_visible = bool(pose_meta.get("lower_body_visible", True)) if pose_meta else True

        if self.exercise == "auto":
            self._auto_buf.append(angles)
            self._auto_joints.append(norm_joints)
            self._auto_elb2d.append(elb_2d if elb_2d is not None else angles[:2])
            self._auto_meta.append({
                "elb_2d": elb_2d,
                "arm_vis": arm_vis,
                "hips_visible": hips_visible,
                "lower_body_visible": lower_body_visible,
            })
            if len(self._auto_buf) > 36:
                self._auto_buf = self._auto_buf[-36:]
                self._auto_joints = self._auto_joints[-36:]
                self._auto_elb2d = self._auto_elb2d[-36:]
                self._auto_meta = self._auto_meta[-36:]

            detected = None
            if len(self._auto_buf) >= 18:
                lb_vis_med = bool(np.mean([m["lower_body_visible"] for m in self._auto_meta]) >= 0.5)
                arm_vis_arr = np.asarray([m["arm_vis"] for m in self._auto_meta], dtype=np.float32)
                detected = auto_detect_exercise(
                    np.asarray(self._auto_buf, dtype=np.float32),
                    lower_body_visible=lb_vis_med,
                    elb_2d_arr=np.asarray(self._auto_elb2d, dtype=np.float32),
                    arm_vis_arr=arm_vis_arr,
                    require_motion=True,
                )

            if detected is None:
                return {
                    "rep_count": self.rep_count,
                    "correct_reps": self.correct_reps,
                    "incorrect_reps": self.incorrect_reps,
                    "stage": self.stage,
                    "primary_angle": None,
                    "verdict": "Detecting",
                    "cues": ["Start moving — AI is auto-detecting your exercise..."],
                    "spoken_cue": None,
                    "rep_completed": False,
                    "form_score": 100.0,
                }

            self.exercise = detected
            label_map = {
                "squat": "Squat",
                "push": "Push-Up",
                "curl": "Bicep Curl",
                "shoulder": "Shoulder Press",
                "row": "Row / Deadlift",
                "lunge": "Lunge",
                "arm_abduction": "Arm Abduction",
            }
            self.last_verdict = "Tracking"
            self.last_cues = [f"Detected {label_map.get(detected, detected.title())} — keep going!"]
            if model.exercise == "auto":
                model = LoadedExerciseModel(self.exercise)

            # Replay buffered motion frames so even the very first rep is counted cleanly
            for b_ang, b_jnt, b_meta in zip(self._auto_buf[:-1], self._auto_joints[:-1], self._auto_meta[:-1]):
                self._step_single(
                    b_ang, b_jnt, model, b_meta["elb_2d"], b_meta["arm_vis"], b_meta["hips_visible"]
                )
            self._auto_buf.clear()
            self._auto_joints.clear()
            self._auto_elb2d.clear()
            self._auto_meta.clear()

        primary, newly_completed, spoken_cue = self._step_single(
            angles, norm_joints, model, elb_2d, arm_vis, hips_visible
        )
        form_score = (100.0 * self.correct_reps / self.rep_count) if self.rep_count > 0 else 100.0
        return {
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "stage": self.stage,
            "primary_angle": round(primary, 1),
            "verdict": self.last_verdict,
            "cues": self.last_cues,
            "spoken_cue": spoken_cue,
            "rep_completed": newly_completed,
            "form_score": round(form_score, 1),
        }


def draw_trainer_hud(
    frame: np.ndarray,
    pose_info: dict[str, Any] | None,
    user: dict[str, Any] | None,
    user_similarity: float,
    rep_count: int,
    verdict: str,
    cues: list[str],
) -> np.ndarray:
    """Draws person bounding box, floating registered user HUD card, skeleton, and voice cue banner."""
    out = frame.copy()
    h, w = out.shape[:2]
    is_bad = verdict.lower() == "incorrect"
    accent = (40, 60, 240) if is_bad else (60, 220, 90)

    if pose_info is not None:
        lms = pose_info["landmarks_norm"]
        pts = {int(k): (int(v[0] * w), int(v[1] * h)) for k, v in lms.items()}
        for a, b in JOINT_LINKS:
            if a in pts and b in pts:
                cv2.line(out, pts[a], pts[b], accent, 2, cv2.LINE_AA)
        for pt in pts.values():
            cv2.circle(out, pt, 4, (50, 220, 255), -1, cv2.LINE_AA)

        x1, y1, x2, y2 = pose_info["person_box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), accent, 2)

        if user is not None:
            hud_lines = [
                f"{user['name']} ({int(user_similarity * 100)}% ID)" if user_similarity > 0 else user["name"],
                f"Age: {user['age']} | Height: {int(user['height_cm'])} cm",
                f"Level: {user['fitness_level']}",
            ]
        else:
            hud_lines = ["Unregistered Athlete", "Face camera to identify"]

        card_w = min(w - 16, 265)
        card_h = 20 * len(hud_lines) + 14
        cx1 = max(8, min(x1, w - card_w - 8))
        cy1 = max(8, y1 - card_h - 6) if (y1 - card_h - 6) >= 8 else min(h - card_h - 8, y1 + 8)

        overlay = out.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (18, 22, 30), -1)
        cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)
        cv2.rectangle(out, (cx1, cy1), (cx1 + card_w, cy1 + card_h), accent, 1)

        for idx_l, txt in enumerate(hud_lines):
            color = (255, 255, 255) if idx_l == 0 else (200, 220, 235)
            cv2.putText(
                out, txt, (cx1 + 8, cy1 + 18 + idx_l * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )

    badge = f"Reps: {rep_count}  |  {verdict.upper()}"
    cv2.rectangle(out, (w - 255, 10), (w - 10, 44), (18, 22, 30), -1)
    cv2.rectangle(out, (w - 255, 10), (w - 10, 44), accent, 1)
    cv2.putText(out, badge, (w - 245, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.54, accent, 2, cv2.LINE_AA)

    if cues:
        msg = cues[0]
        cv2.rectangle(out, (0, h - 46), (w, h), (15, 18, 26), -1)
        cv2.putText(out, f"Coach: {msg}", (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.62, accent, 2, cv2.LINE_AA)

    return out


def _extract_segments_from_1d(
    grid: np.ndarray,
    curve: np.ndarray,
    invert: bool,
    prom: float,
    min_drop: float,
    min_dist: int,
) -> list[tuple[int, int, int]]:
    """Returns list of (start_frame, stop_frame, peak_frame) from a 1D angle curve."""
    if len(curve) >= 5:
        curve = np.convolve(curve, [0.25, 0.5, 0.25], mode="same")
    search = -curve if invert else curve
    peaks, _ = find_peaks(search, prominence=prom, distance=max(2, min_dist))
    found: list[tuple[int, int, int]] = []
    for i, pk in enumerate(peaks):
        left = 0 if i == 0 else peaks[i - 1]
        right = len(curve) - 1 if i == len(peaks) - 1 else peaks[i + 1]
        if invert:
            s_i = left + int(np.argmax(curve[left:pk + 1]))
            e_i = pk + int(np.argmax(curve[pk:right + 1]))
            ok = (curve[s_i] - curve[pk] >= min_drop) and (curve[e_i] - curve[pk] >= min_drop)
        else:
            s_i = left + int(np.argmin(curve[left:pk + 1]))
            e_i = pk + int(np.argmin(curve[pk:right + 1]))
            ok = (curve[pk] - curve[s_i] >= min_drop) and (curve[pk] - curve[e_i] >= min_drop)
        if ok and e_i > s_i + 2:
            found.append((int(grid[s_i]), int(grid[e_i]) + 1, int(grid[pk])))
    return found


def segment_reps_postset(
    frames: np.ndarray,
    angles: np.ndarray,
    fps: float,
    exercise: str,
    step: int = 3,
    elb_2d: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    if len(frames) < 6 or fps <= 0:
        return []

    grid = np.arange(frames[0], frames[-1] + 1, step)
    min_dist = max(2, int(0.65 * fps / step))

    # Special per-arm curl segmentation (handles simultaneous two-arm, alternating, and incline curls)
    if exercise == "curl":
        if elb_2d is not None and len(elb_2d) == len(angles):
            l_sig = 0.5 * angles[:, 0] + 0.5 * elb_2d[:, 0]
            r_sig = 0.5 * angles[:, 1] + 0.5 * elb_2d[:, 1]
        else:
            l_sig, r_sig = angles[:, 0], angles[:, 1]

        candidates: list[tuple[int, int, int]] = []
        for arm_sig in (l_sig, r_sig):
            if _rom_p90_p10(arm_sig) >= 20.0:
                curve = np.interp(grid, frames, arm_sig)
                candidates.extend(
                    _extract_segments_from_1d(grid, curve, invert=True, prom=15.0, min_drop=13.0, min_dist=min_dist)
                )
        candidates.sort(key=lambda t: t[2])

        merged: list[tuple[int, int, int]] = []
        merge_frames = int(0.55 * fps)
        for seg in candidates:
            if merged and abs(seg[2] - merged[-1][2]) <= merge_frames:
                prev = merged[-1]
                merged[-1] = (min(prev[0], seg[0]), max(prev[1], seg[1]), (prev[2] + seg[2]) // 2)
            else:
                merged.append(seg)
        segments = [(s, e) for s, e, _ in merged]
        if not segments and (frames[-1] - frames[0]) >= int(0.8 * fps):
            segments.append((int(frames[0]), int(frames[-1]) + 1))
        return segments

    if exercise == "squat":
        sig, invert, prom, min_drop = angles[:, 12:14].mean(axis=1), True, 20.0, 16.0
    elif exercise == "lunge":
        sig, invert, prom, min_drop = angles[:, 12:14].min(axis=1), True, 22.0, 18.0
    elif exercise == "push":
        sig, invert, prom, min_drop = angles[:, 0:2].min(axis=1), True, 15.0, 14.0
    elif exercise in ("shoulder", "arm_abduction"):
        sig, invert, prom, min_drop = angles[:, 2:4].max(axis=1), False, 20.0, 16.0
    elif exercise == "row":
        elb_rom = max(_rom_p90_p10(angles[:, 0]), _rom_p90_p10(angles[:, 1]))
        hip_rom = _rom_p90_p10(angles[:, 4:6].mean(axis=1))
        if elb_rom >= 32.0 and elb_rom >= 0.70 * hip_rom:
            sig, invert, prom, min_drop = angles[:, 0:2].min(axis=1), True, 16.0, 14.0
        else:
            sig, invert, prom, min_drop = angles[:, 4:6].mean(axis=1), True, 20.0, 16.0
    else:
        sig, invert, prom, min_drop = angles[:, 4:6].mean(axis=1), True, 18.0, 14.0

    curve = np.interp(grid, frames, sig)
    found = _extract_segments_from_1d(grid, curve, invert=invert, prom=prom, min_drop=min_drop, min_dist=min_dist)
    segments = [(s, e) for s, e, _ in found]

    if not segments and (frames[-1] - frames[0]) >= int(0.8 * fps):
        segments.append((int(frames[0]), int(frames[-1]) + 1))
    return segments


def analyze_postset_video(
    video_path: Path,
    output_video_path: Path,
    exercise: str,
    model: LoadedExerciseModel,
    face_engine,
    registered_users: list[dict[str, Any]],
    override_user: dict[str, Any] | None = None,
    step: int = 3,
    model_registry: ModelRegistry | None = None,
) -> dict[str, Any]:
    """Full post-set video analysis: Face ID + Pose + Rep Segmentation + Model Prediction + Annotated MP4."""
    from .face_id import ActiveUserTracker

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    tracker = ActiveUserTracker(lock_ttl_seconds=30.0, check_every_n_frames=6)
    if override_user is not None:
        tracker.locked_user = override_user
        tracker.locked_similarity = 1.0

    sampled_frames, sampled_angles, sampled_joints, sampled_elb2d, sampled_arm_vis = [], [], [], [], []
    hips_vis_flags: list[bool] = []
    lower_vis_flags: list[bool] = []
    pose_cache: dict[int, dict[str, Any] | None] = {}
    user_cache: dict[int, tuple[dict[str, Any] | None, float]] = {}

    frame_idx = 0
    width, height = 640, 480
    with create_pose_landmarker() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = min(1.0, 854.0 / max(h, w))
            rw, rh = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
            width, height = rw, rh
            if frame_idx % step == 0:
                resized = cv2.resize(frame, (rw, rh))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                p_info = extract_pose_from_result(res, rw, rh)
                pose_cache[frame_idx] = p_info
                if p_info is not None:
                    if override_user is None:
                        u, sim = tracker.update(resized, p_info["person_box"], face_engine, registered_users)
                    else:
                        u, sim = override_user, 1.0
                    user_cache[frame_idx] = (u, sim)
                    if p_info["angles"] is not None:
                        sampled_frames.append(frame_idx)
                        sampled_angles.append(p_info["angles"])
                        sampled_joints.append(p_info["norm_joints"])
                        sampled_elb2d.append(p_info["elb_2d"])
                        sampled_arm_vis.append(p_info.get("arm_vis", (1.0, 1.0)))
                        hips_vis_flags.append(bool(p_info.get("hips_visible", True)))
                        lower_vis_flags.append(bool(p_info.get("lower_body_visible", True)))
            frame_idx += 1
    cap.release()

    frames_arr = np.asarray(sampled_frames, dtype=np.int32)
    angles_arr = np.asarray(sampled_angles, dtype=np.float32) if sampled_angles else np.empty((0, 14), dtype=np.float32)
    joints_arr = np.asarray(sampled_joints, dtype=np.float32) if sampled_joints else np.empty((0, 12, 3), dtype=np.float32)
    elb2d_arr = np.asarray(sampled_elb2d, dtype=np.float32) if sampled_elb2d else None
    arm_vis_arr = np.asarray(sampled_arm_vis, dtype=np.float32) if sampled_arm_vis else None
    hips_vis_med = bool(np.mean(hips_vis_flags) >= 0.5) if hips_vis_flags else True
    lower_vis_med = bool(np.mean(lower_vis_flags) >= 0.5) if lower_vis_flags else True

    auto_detected = False
    if exercise == "auto":
        auto_detected = True
        detected = auto_detect_exercise(
            angles_arr,
            lower_body_visible=lower_vis_med,
            elb_2d_arr=elb2d_arr,
            arm_vis_arr=arm_vis_arr,
            require_motion=False,
        )
        exercise = detected or "squat"
        model = model_registry.get(exercise) if model_registry is not None else LoadedExerciseModel(exercise)

    segments = segment_reps_postset(frames_arr, angles_arr, fps, exercise, step=step, elb_2d=elb2d_arr)
    rep_reports: list[dict[str, Any]] = []
    for rep_num, (start, stop) in enumerate(segments, 1):
        mask = (frames_arr >= start) & (frames_arr < stop)
        if mask.sum() < 4:
            continue
        ang_seq, skel_grid = resample_sequence(angles_arr[mask], joints_arr[mask])
        pred = model.predict_rep(ang_seq, skel_grid, hips_visible=hips_vis_med)
        rep_reports.append({
            "rep": rep_num,
            "start_sec": round(start / fps, 2),
            "end_sec": round(stop / fps, 2),
            "verdict": pred["verdict"],
            "is_incorrect": pred["is_incorrect"],
            "incorrect_probability": pred["incorrect_probability"],
            "cues": pred["cues"],
            "model_used": pred["model_used"],
        })

    raw_out = output_video_path.with_suffix(".raw.mp4")
    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    last_pose = None
    last_user, last_sim = tracker.locked_user, tracker.locked_similarity
    idx = 0
    completed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height))
        if idx in pose_cache:
            last_pose = pose_cache[idx]
        if idx in user_cache and user_cache[idx][0] is not None:
            last_user, last_sim = user_cache[idx]

        while completed < len(segments) and idx >= segments[completed][1] - 1:
            completed += 1

        active_idx = next((i for i, (s, e) in enumerate(segments) if s <= idx < e), None)
        if active_idx is not None and active_idx < len(rep_reports):
            cur_verdict = rep_reports[active_idx]["verdict"]
            cur_cues = rep_reports[active_idx]["cues"]
        elif completed > 0 and completed - 1 < len(rep_reports):
            cur_verdict = rep_reports[completed - 1]["verdict"]
            cur_cues = rep_reports[completed - 1]["cues"]
        else:
            cur_verdict = "Tracking"
            cur_cues = ["Analyzing exercise form..."]

        annotated = draw_trainer_hud(frame, last_pose, last_user, last_sim, completed, cur_verdict, cur_cues)
        writer.write(annotated)
        idx += 1
    cap.release()
    writer.release()

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None

    encoded_ok = False
    if ffmpeg_bin:
        try:
            subprocess.run(
                [
                    ffmpeg_bin, "-y", "-loglevel", "error", "-i", str(raw_out),
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(output_video_path),
                ],
                check=True,
            )
            raw_out.unlink(missing_ok=True)
            encoded_ok = output_video_path.exists() and output_video_path.stat().st_size > 1024
        except Exception:
            encoded_ok = False

    if not encoded_ok:
        # Fallback: Windows Media Foundation native H.264 ('avc1') so browsers can still play it
        cap_re = cv2.VideoCapture(str(raw_out))
        w_h264 = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
        if w_h264.isOpened():
            while True:
                ok_f, fr = cap_re.read()
                if not ok_f:
                    break
                w_h264.write(fr)
            w_h264.release()
            cap_re.release()
            raw_out.unlink(missing_ok=True)
        else:
            cap_re.release()
            raw_out.replace(output_video_path)

    total_reps = len(rep_reports)
    incorrect_reps = sum(1 for r in rep_reports if r["is_incorrect"])
    correct_reps = total_reps - incorrect_reps
    form_score = (100.0 * correct_reps / total_reps) if total_reps > 0 else 0.0
    unique_cues = list(dict.fromkeys(c for r in rep_reports for c in r["cues"] if "Great rep" not in c))
    if not unique_cues and total_reps > 0:
        unique_cues = ["Awesome set! Clean form throughout."]

    return {
        "exercise": exercise,
        "auto_detected": auto_detected,
        "recognized_user": tracker.locked_user,
        "recognition_confidence": tracker.locked_similarity,
        "total_reps": total_reps,
        "correct_reps": correct_reps,
        "incorrect_reps": incorrect_reps,
        "form_score": round(form_score, 1),
        "summary_cues": unique_cues,
        "reps": rep_reports,
        "model_used": model.mode,
    }
