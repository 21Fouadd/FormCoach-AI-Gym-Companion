# %% [markdown]
# # FormCoach — Unified Multi-Model Ensemble Benchmark (All 6 Exercises)
#
# Tests whether mixing multiple diverse models (**Random Forest + HistGradientBoosting + SVM-RBF**) via **Soft-Voting** beats every single model, AND compares:
# 1. **Unified Ensemble Architecture (Per-Exercise):** Same `RF + HistGB + SVM-RBF` Soft-Voting Ensemble trained per exercise (`squat`, `push`, `curl`, `shoulder`, `lunge`, `arm_abduction`).
# 2. **Single Universal Model (All-In-One):** One single `RF + HistGB + SVM-RBF` model trained across ALL 6 exercises combined (with exercise one-hot features).
#
# Uses **100% pure `scikit-learn`** (`VotingClassifier(voting="soft")`), so the exported `.joblib` loads directly into your local FastAPI backend with **zero extra dependencies** and **zero code changes**!

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
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
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
            n_estimators=300, max_depth=14, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=220, max_depth=6, learning_rate=0.05,
            min_samples_leaf=15, l2_regularization=1.0,
            class_weight="balanced", random_state=seed,
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        ),
    }


def make_unified_ensemble(seed: int) -> VotingClassifier:
    est = make_base_estimators(seed)
    return VotingClassifier(
        estimators=[
            ("rf", est["Random Forest"]),
            ("hgb", est["HistGradientBoosting"]),
            ("svm", est["SVM (RBF)"]),
        ],
        voting="soft",
        weights=[2.0, 1.5, 1.5],
        n_jobs=-1,
    )


# -------- Dataset Loaders (Reuse Drive Caches + Fast UI-PRMD / NgoQuocBao1010) --------
def load_drive_cache(ex: str) -> list[tuple[np.ndarray, int, str]]:
    cands = [
        f"pose_angles_cache_{ex}.npz",
        f"pose_cache_{ex}_rehab1.npz",
        f"pose_cache_{ex}_rehab0.npz",
    ]
    for root in DRIVE_ROOTS:
        for fn in cands:
            p = root / fn
            if p.exists():
                c = np.load(p, allow_pickle=True)
                xk = "X_angles" if "X_angles" in c else "X_ang"
                rows = list(zip(c[xk], c["y"], c["groups"]))
                print(f"  [Drive Cache] {ex}: {len(rows)} reps from {p.name}")
                return rows
    return []


def calc_a3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-8), -1.0, 1.0))).astype(np.float32)


def calc_a2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, v = a[:, :2] - b[:, :2], c[:, :2] - b[:, :2]
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-8), -1.0, 1.0))).astype(np.float32)


def load_ngo_curl() -> list[tuple[np.ndarray, int, str]]:
    base = "https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/core/bicep_model"
    rows = []
    for sp in ("train.csv", "test.csv"):
        raw = urllib.request.urlopen(urllib.request.Request(f"{base}/{sp}", headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
        df = pd.read_csv(io.BytesIO(raw))
        pt = lambda n: df[[f"{n}_x", f"{n}_y", f"{n}_z"]].to_numpy(np.float32)
        l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hp, r_hp = (
            pt("left_shoulder"), pt("right_shoulder"), pt("left_elbow"), pt("right_elbow"),
            pt("left_wrist"), pt("right_wrist"), pt("left_hip"), pt("right_hip"),
        )
        m_sh, m_hp = 0.5 * (l_sh + r_sh), 0.5 * (l_hp + r_hp)
        sc = np.maximum(np.linalg.norm(m_sh - m_hp, axis=-1), 0.12)
        stand = np.full(len(df), 175.0, np.float32)
        ang14 = np.stack([
            0.5 * (calc_a3(l_sh, l_el, l_wr) + calc_a2(l_sh, l_el, l_wr)),
            0.5 * (calc_a3(r_sh, r_el, r_wr) + calc_a2(r_sh, r_el, r_wr)),
            calc_a3(l_hp, l_sh, l_el), calc_a3(r_hp, r_sh, r_el),
            stand, stand,
            calc_a3(m_sh, m_hp, m_hp + np.array([0.0, -1.0, 0.0], np.float32)),
            np.degrees(np.arctan2(np.abs(l_sh[:, 1] - r_sh[:, 1]), np.hypot(l_sh[:, 0] - r_sh[:, 0], l_sh[:, 2] - r_sh[:, 2]))).astype(np.float32),
            np.degrees(np.arctan2(np.abs(l_hp[:, 1] - r_hp[:, 1]), np.hypot(l_hp[:, 0] - r_hp[:, 0], l_hp[:, 2] - r_hp[:, 2]))).astype(np.float32),
            ((l_sh[:, 1] - l_wr[:, 1]) / sc).astype(np.float32),
            ((r_sh[:, 1] - r_wr[:, 1]) / sc).astype(np.float32),
            stand, stand, stand,
        ], axis=1)
        labels = (df["label"].astype(str).str.upper() != "C").astype(np.int32).to_numpy()
        chg = np.where(np.diff(labels) != 0)[0] + 1
        cid = 0
        for s_i, e_i in zip(np.concatenate([[0], chg]), np.concatenate([chg, [len(labels)]])):
            seg, lbl = ang14[s_i:e_i], int(labels[s_i])
            if len(seg) < 16 or not np.isfinite(seg).all():
                continue
            for sub in np.array_split(seg, max(1, len(seg) // 320)):
                cid += 1
                grp = f"ngo_{sp.split('.')[0]}_G{cid:03d}"
                if len(sub) < 16:
                    continue
                sig = np.minimum(sub[:, 0], sub[:, 1])
                pks, _ = find_peaks(-sig, prominence=18.0, distance=18)
                added = 0
                for i, pk in enumerate(pks):
                    l, r = (0 if i == 0 else pks[i - 1]), (len(sig) - 1 if i == len(pks) - 1 else pks[i + 1])
                    st, en = l + int(np.argmax(sig[l:pk + 1])), pk + int(np.argmax(sig[pk:r + 1]))
                    if en - st >= 12 and (sig[st] - sig[pk] >= 15.0 or sig[en] - sig[pk] >= 15.0):
                        rows.append((resample_32(sub[st:en + 1]), lbl, grp))
                        added += 1
                if added == 0:
                    for w0 in range(0, len(sub) - 24, 45):
                        rows.append((resample_32(sub[w0:w0 + 45]), lbl, grp))
    print(f"  [NgoQuocBao1010] curl: {len(rows)} reps")
    return rows


# UI-PRMD Forward Kinematics (matches Colab_Upgrade_UIPRMD_And_Row.py)
UI_PARENTS = [None, 0, 1, 2, 1, 4, 5, 6, 1, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 7, 11]


def _rot_zyx(ang_deg: np.ndarray) -> np.ndarray:
    a, b, c = np.radians(ang_deg)
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], np.float32)
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]], np.float32)
    return Rz @ Ry @ Rx


def uiprmd_fk_to_14(pos_arr: np.ndarray, ang_arr: np.ndarray) -> np.ndarray | None:
    T = min(len(pos_arr), len(ang_arr))
    if T < 12 or pos_arr.shape[1] < 66 or ang_arr.shape[1] < 66:
        return None
    pos, ang = pos_arr[:T, :66].reshape(T, 22, 3), ang_arr[:T, :66].reshape(T, 22, 3)
    out = []
    for t in range(T):
        w_pos, w_rot = np.zeros((22, 3), np.float32), [np.eye(3, dtype=np.float32)] * 22
        w_pos[0], w_rot[0] = pos[t, 0], _rot_zyx(ang[t, 0])
        for j in range(1, 22):
            p = UI_PARENTS[j]
            w_pos[j] = w_pos[p] + w_rot[p] @ pos[t, j]
            w_rot[j] = w_rot[p] @ _rot_zyx(ang[t, j])
        ls, le, lw, rs, re, rw = w_pos[4], w_pos[5], w_pos[6], w_pos[8], w_pos[9], w_pos[10]
        lh, lk, la, rh, rk, ra = w_pos[12], w_pos[13], w_pos[14], w_pos[16], w_pos[17], w_pos[18]
        sh, hp, an = 0.5 * (ls + rs), 0.5 * (lh + rh), 0.5 * (la + ra)
        sc = max(float(np.linalg.norm(sh - hp)), 1e-3)
        ca = lambda a, b, c: float(np.degrees(np.arccos(np.clip(np.dot(a - b, c - b) / max(np.linalg.norm(a - b) * np.linalg.norm(c - b), 1e-8), -1.0, 1.0))))
        tl = lambda p1, p2: float(np.degrees(np.arctan2(abs(p1[1] - p2[1]), np.hypot(p1[0] - p2[0], p1[2] - p2[2]))))
        row = np.array([
            ca(ls, le, lw), ca(rs, re, rw), ca(lh, ls, le), ca(rh, rs, re),
            ca(ls, lh, lk), ca(rs, rh, rk), ca(sh, hp, hp + np.array([0.0, 1.0, 0.0], np.float32)),
            tl(ls, rs), tl(lh, rh), float((lw[1] - ls[1]) / sc), float((rw[1] - rs[1]) / sc),
            ca(sh, hp, an), ca(lh, lk, la), ca(rh, rk, ra),
        ], np.float32)
        if np.isfinite(row).all():
            out.append(row)
    return np.asarray(out, np.float32) if len(out) >= 12 else None


def load_uiprmd() -> dict[str, list[tuple[np.ndarray, int, str]]]:
    def _dl(url: str, dest: Path) -> Path:
        if not (dest.exists() and zipfile.is_zipfile(dest)):
            print(f"Downloading {dest.name}...")
            urllib.request.urlretrieve(url, dest)
        return dest

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

    z_c = _dl("https://zenodo.org/records/23104965/files/Segmented%20Movements.zip?download=1", WORK_DIR / "uiprmd_correct.zip")
    z_i = _dl("https://zenodo.org/records/23104965/files/Incorrect%20Movements.zip?download=1", WORK_DIR / "uiprmd_incorrect.zip")
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


# === Load All Available Exercise Datasets ===
print("Loading datasets...")
uiprmd = load_uiprmd()
ngo_curl = load_ngo_curl()

datasets: dict[str, list[tuple[np.ndarray, int, str]]] = {}
for ex in EXERCISES:
    rows = load_drive_cache(ex)
    if ex in uiprmd:
        rows = rows + uiprmd[ex]
    if ex == "curl":
        rows = ngo_curl + rows
    if rows:
        datasets[ex] = rows

# === PART 1: Benchmark Individual Models vs. Unified Soft-Voting Ensemble (Per Exercise) ===
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
    model_names = ["Random Forest", "HistGradientBoosting", "SVM (RBF)", "Unified Ensemble (RF+HGB+SVM)"]
    oof = {n: np.zeros(len(y), np.float32) for n in model_names}

    for fi, (tr, te) in enumerate(cv.split(X_tab, y, g), 1):
        ang_tr = np.concatenate([
            X_ang[tr], X_ang[tr][:, :, ANGLE_LR_SWAP],
            X_ang[tr] + rng.normal(0, 1.2, X_ang[tr].shape).astype(np.float32),
        ], axis=0)
        y_tr = np.concatenate([y[tr], y[tr], y[tr]], axis=0)
        tab_tr = build_tabular_features(ang_tr)

        clfs = make_base_estimators(SEED + fi)
        clfs["Unified Ensemble (RF+HGB+SVM)"] = make_unified_ensemble(SEED + fi)
        for name, clf in clfs.items():
            clf.fit(tab_tr, y_tr)
            oof[name][te] = clf.predict_proba(X_tab[te])[:, 1]

    # Tune threshold for Unified Ensemble on OOF
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.38, 0.62, 25):
        f1_v = float(f1_score(y, (oof["Unified Ensemble (RF+HGB+SVM)"] >= thr).astype(int), average="macro"))
        if f1_v > best_f1:
            best_f1, best_thr = f1_v, float(round(thr, 3))

    b_rows = []
    for name in model_names:
        thr = best_thr if name.startswith("Unified") else 0.5
        preds = (oof[name] >= thr).astype(int)
        b_rows.append({
            "Model": name,
            "Family": "Soft-Voting Ensemble" if name.startswith("Unified") else "Tabular",
            "OOF Macro F1": round(float(f1_score(y, preds, average="macro")), 4),
            "Accuracy": round(float(accuracy_score(y, preds)), 4),
            "Balanced Acc": round(float(balanced_accuracy_score(y, preds)), 4),
            "ROC-AUC": round(float(roc_auc_score(y, oof[name])), 4),
        })
    board = pd.DataFrame(b_rows).sort_values("OOF Macro F1", ascending=False).reset_index(drop=True)
    print(board.to_string(index=False))

    rf_f1 = float(board[board["Model"] == "Random Forest"].iloc[0]["OOF Macro F1"])
    ens_row = board[board["Model"] == "Unified Ensemble (RF+HGB+SVM)"].iloc[0]
    summary_rows.append({
        "Exercise": ex, "Reps": len(y), "Groups": n_subj,
        "RF F1": rf_f1,
        "Ensemble F1": float(ens_row["OOF Macro F1"]),
        "Delta": round(float(ens_row["OOF Macro F1"]) - rf_f1, 4),
        "Ensemble AUC": float(ens_row["ROC-AUC"]),
        "Winner": str(board.iloc[0]["Model"]),
    })

    # Fit final Unified Soft-Voting Ensemble on full dataset + mirror flip
    ens_final = make_unified_ensemble(SEED)
    ens_final.fit(
        build_tabular_features(np.concatenate([X_ang, X_ang[:, :, ANGLE_LR_SWAP]], axis=0)),
        np.concatenate([y, y], axis=0),
    )
    ex_dir = EXPORT_DIR / ex
    ex_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ens_final, ex_dir / "best_tabular_model.joblib", compress=3)
    board.to_csv(ex_dir / "model_leaderboard.csv", index=False)
    (ex_dir / "model_metadata.json").write_text(json.dumps({
        "exercise": ex,
        "winner_model": str(board.iloc[0]["Model"]),
        "best_tabular_model": "Unified Ensemble (RF+HGB+SVM)",
        "threshold": best_thr,
        "sequence_length": SEQ_LEN,
        "features": FEATURES,
        "core_joints": CORE_JOINTS,
        "total_reps": int(len(y)),
        "unique_subjects": int(n_subj),
        "leaderboard": board.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")


# === PART 2: Test ONE Single Universal Model Across ALL Exercises Combined ===
print("\n================ [UNIVERSAL ALL-IN-ONE MODEL TEST] ================")
ex_list = list(datasets.keys())
X_all_list, y_all_list, g_all_list, ex_idx_list = [], [], [], []
for idx, ex in enumerate(ex_list):
    rows = datasets[ex]
    Xa = np.stack([r[0] for r in rows]).astype(np.float32)
    Xt = build_tabular_features(Xa)
    oh = np.zeros((len(rows), len(ex_list)), dtype=np.float32)
    oh[:, idx] = 1.0
    X_all_list.append(np.concatenate([Xt, oh], axis=1))
    y_all_list.append(np.asarray([r[1] for r in rows], dtype=np.int32))
    g_all_list.append(np.asarray([f"{ex}_{r[2]}" for r in rows], dtype=object))
    ex_idx_list.append(np.full(len(rows), idx, dtype=np.int32))

X_uni = np.concatenate(X_all_list, axis=0)
y_uni = np.concatenate(y_all_list, axis=0)
g_uni = np.concatenate(g_all_list, axis=0)
ex_uni = np.concatenate(ex_idx_list, axis=0)

oof_uni = np.zeros(len(y_uni), np.float32)
cv_u = StratifiedGroupKFold(n_splits=5)
for fi, (tr, te) in enumerate(cv_u.split(X_uni, y_uni, g_uni), 1):
    clf_u = make_unified_ensemble(SEED + fi)
    clf_u.fit(X_uni[tr], y_uni[tr])
    oof_uni[te] = clf_u.predict_proba(X_uni[te])[:, 1]

uni_overall_f1 = round(float(f1_score(y_uni, (oof_uni >= 0.5).astype(int), average="macro")), 4)
print(f"Single Universal Model Overall Macro F1: {uni_overall_f1}")
for r in summary_rows:
    idx = ex_list.index(r["Exercise"])
    m = ex_uni == idx
    r["Universal 1-Model F1"] = round(float(f1_score(y_uni[m], (oof_uni[m] >= 0.5).astype(int), average="macro")), 4)

print("\n================ FINAL COMPARISON SUMMARY ================")
print(pd.DataFrame(summary_rows).to_string(index=False))

zip_out = SAVE_ROOT / "gym_model_bundle_unified_ensemble.zip"
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in EXPORT_DIR.rglob("*"):
        if p.is_file():
            zf.write(p, arcname=p.relative_to(EXPORT_DIR).as_posix())
print(f"\nPACKAGED UNIFIED ENSEMBLE BUNDLE: {zip_out} ({zip_out.stat().st_size / 1e6:.2f} MB)")
if files is not None:
    files.download(str(zip_out))
