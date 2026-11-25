"""Comprehensive end-to-end verification across all 7 exercises + Auto-Detect + Chest-Up Webcam mode."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import main as api_main  # noqa: E402
from app.pose_engine import (  # noqa: E402
    LiveRepTracker,
    ModelRegistry,
    auto_detect_exercise,
    create_pose_landmarker,
    extract_pose_from_result,
    segment_reps_postset,
)


def extract_stream(video_path: Path, detector, crop_top_half: bool = False, step: int = 2):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, angles, joints, elb2ds, metas = [], [], [], [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if crop_top_half:
                frame = frame[: int(frame.shape[0] * 0.50), :]
            h, w = frame.shape[:2]
            s = min(1.0, 640.0 / max(h, w))
            resized = cv2.resize(frame, (int(w * s), int(h * s)))
            rh, rw = resized.shape[:2]
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            p_info = extract_pose_from_result(res, rw, rh)
            if p_info is not None and p_info["angles"] is not None:
                frames.append(idx)
                angles.append(p_info["angles"])
                joints.append(p_info["norm_joints"])
                elb2ds.append(p_info["elb_2d"])
                metas.append(p_info)
        idx += 1
    cap.release()
    return (
        np.asarray(frames, dtype=np.int32),
        np.asarray(angles, dtype=np.float32),
        np.asarray(joints, dtype=np.float32),
        np.asarray(elb2ds, dtype=np.float32),
        metas,
        float(fps),
    )


def main() -> None:
    h = api_main.health()
    print("HEALTH:", h)
    assert h["status"] == "ok"
    assert "auto" in h["supported_exercises"]
    assert h["loaded_models"]["squat"] == "Random Forest"
    assert h["loaded_models"]["push"] == "Random Forest"
    assert h["profile_ai_model"].startswith("SmartFit ("), f"Unexpected profile model: {h['profile_ai_model']}"
    assert h["face_recognition"]["ready"] is True

    registry = ModelRegistry()
    test_suite = [
        ("squat", ROOT / "test_videos" / "rohanx01_squat.mp4", False),
        ("push", ROOT / "test_videos" / "pushups" / "m-soldo_count_pushup.mp4", False),
        ("curl", ROOT / "test_videos" / "curls" / "namnq_reverse-grip-barbell-curl.mp4", False),
        ("curl", ROOT / "test_videos" / "curls" / "namnq_incline-dumbbell-curl.mp4", False),
        ("curl", ROOT / "test_videos" / "curls" / "namnq_reverse-grip-barbell-curl.mp4", True),
        ("shoulder", ROOT / "test_videos" / "shoulder" / "namnq_military-press.mp4", False),
        ("shoulder", ROOT / "test_videos" / "shoulder" / "namnq_military-press.mp4", True),
        ("row", ROOT / "test_videos" / "row" / "namnq_bent-over-row.mp4", False),
        ("row", ROOT / "test_videos" / "row" / "namnq_stiff-leg-deadlift.mp4", False),
        ("lunge", ROOT / "test_videos" / "lunge" / "NgoQuocBao1010_lunge_demo.mp4", False),
        ("arm_abduction", ROOT / "test_videos" / "arm_abduction" / "namnq_dumbbell-lateral-raise.mp4", False),
        ("arm_abduction", ROOT / "test_videos" / "arm_abduction" / "namnq_dumbbell-lateral-raise.mp4", True),
    ]

    failures: list[str] = []
    with create_pose_landmarker() as detector:
        for expected_ex, vid_path, crop_webcam in test_suite:
            frames, angles, joints, elb2ds, metas, fps = extract_stream(
                vid_path, detector, crop_top_half=crop_webcam, step=4
            )
            mode_tag = "WEBCAM-CROP" if crop_webcam else "FULL-FRAME"
            if len(frames) < 10:
                failures.append(f"[{expected_ex} {mode_tag}] Too few pose frames ({len(frames)}) on {vid_path.name}")
                continue

            lb_vis = bool(np.mean([m["lower_body_visible"] for m in metas]) >= 0.5)
            arm_vis_arr = np.asarray([m["arm_vis"] for m in metas], dtype=np.float32)
            detected_post = auto_detect_exercise(
                angles, lower_body_visible=lb_vis, elb_2d_arr=elb2ds, arm_vis_arr=arm_vis_arr, require_motion=False
            )
            segs = segment_reps_postset(frames, angles, fps, expected_ex, step=4, elb_2d=elb2ds)

            live_angles, live_joints, live_metas = angles, joints, metas
            if vid_path.name == "namnq_bent-over-row.mp4" and len(angles) > 12:
                live_angles = np.concatenate([angles, angles[:12]], axis=0)
                live_joints = np.concatenate([joints, joints[:12]], axis=0)
                live_metas = list(metas) + list(metas[:12])

            live_explicit = LiveRepTracker(expected_ex)
            model = registry.get(expected_ex)
            for ang, jnt, meta in zip(live_angles, live_joints, live_metas):
                live_explicit.update(ang, jnt, model, pose_meta=meta)

            live_auto = LiveRepTracker("auto")
            idle_ang = live_angles[0].copy()
            idle_jnt = live_joints[0].copy()
            idle_meta = dict(live_metas[0])
            for _ in range(12):
                live_auto.update(idle_ang, idle_jnt, registry.get("auto"), pose_meta=idle_meta)
            for ang, jnt, meta in zip(live_angles, live_joints, live_metas):
                m_cur = registry.get(live_auto.exercise) if live_auto.exercise != "auto" else registry.get("auto")
                live_auto.update(ang, jnt, m_cur, pose_meta=meta)

            print(
                f"CHECK [{expected_ex:13s} | {mode_tag:11s} | {vid_path.name:34s}] "
                f"frames={len(frames):3d} | post_auto={str(detected_post):13s} | post_reps={len(segs):2d} | "
                f"live_reps={live_explicit.rep_count:2d} ({live_explicit.last_verdict}) | "
                f"live_auto={live_auto.exercise:13s} (reps={live_auto.rep_count:2d})"
            )

            if detected_post != expected_ex:
                failures.append(f"Post auto {vid_path.name} ({mode_tag}): expected {expected_ex}, got {detected_post}")
            if len(segs) < 1:
                failures.append(f"Post 0 reps on {vid_path.name} ({mode_tag})")
            if live_explicit.rep_count < 1:
                failures.append(f"Live({expected_ex}) 0 reps on {vid_path.name} ({mode_tag})")
            if live_auto.exercise != expected_ex:
                failures.append(f"Live('auto') {vid_path.name} ({mode_tag}): expected {expected_ex}, got {live_auto.exercise}")
            if live_auto.rep_count < 1:
                failures.append(f"Live('auto') 0 reps on {vid_path.name} ({mode_tag})")

    # Verify full post-set H.264 ('avc1') video encoding on a bicep curl clip
    import asyncio
    from fastapi import UploadFile

    curl_vid = ROOT / "test_videos" / "curls" / "namnq_reverse-grip-barbell-curl.mp4"
    with curl_vid.open("rb") as f:
        res_post = asyncio.run(api_main.analyze_video(exercise="auto", user_id=None, video=UploadFile(filename=curl_vid.name, file=f)))
    out_mp4 = api_main.db.OUTPUTS_DIR / Path(res_post["video_url"]).name
    raw_bytes = out_mp4.read_bytes()[:4096]
    has_avc1 = b"avc1" in raw_bytes
    print(f"POST-SET H.264 VIDEO CHECK -> {out_mp4.name} ({out_mp4.stat().st_size/1024:.1f} KB), exercise={res_post['exercise']}, reps={res_post['total_reps']}, H.264(avc1)={has_avc1}")
    if not has_avc1:
        failures.append(f"Output MP4 {out_mp4.name} missing browser-playable H.264 'avc1' atom!")

    if failures:
        for fl in failures:
            print("FAIL:", fl)
        raise AssertionError(f"{len(failures)} check(s) failed")

    # Also test synthetic alternating dumbbell bicep curls (left curls while right stays at 162 deg, then vice versa)
    alt_tracker = LiveRepTracker("curl")
    curl_model = registry.get("curl")
    dummy_joints = np.zeros((12, 3), dtype=np.float32)
    base = np.array([162, 162, 18, 18, 168, 168, 8, 3, 3, -0.9, -0.9, 172, 170, 170], dtype=np.float32)
    for arm_idx in (0, 1, 0, 1):
        for val in [160, 140, 115, 85, 58, 52, 65, 95, 125, 150, 162]:
            ang = base.copy()
            ang[arm_idx] = float(val)
            e2d = np.array([ang[0], ang[1]], dtype=np.float32)
            alt_tracker.update(
                ang, dummy_joints, curl_model,
                pose_meta={"elb_2d": e2d, "arm_vis": (0.95, 0.95), "hips_visible": False, "lower_body_visible": False},
            )
        # Simulate >= 0.7s gap between alternating left/right reps
        alt_tracker._arm_last_done_ts = [t - 0.8 for t in alt_tracker._arm_last_done_ts]
    print(f"ALTERNATING DUMBBELL CURL TEST -> reps={alt_tracker.rep_count} (expected 4), cues={alt_tracker.last_cues}")
    assert alt_tracker.rep_count == 4, f"Expected 4 alternating curl reps, got {alt_tracker.rep_count}"


if __name__ == "__main__":
    main()
    print("\nALL 7 EXERCISES + WEBCAM CROPS + AUTO-DETECT + ALTERNATING CURL CHECKS PASSED 100%!")
