# FormCoach — AI Personal Gym Companion

**Graduation Project · Team AI25** (`Mohamed Shawky · Ahmed El Kholy · Fouad Khaled`)

Real-time computer-vision gym trainer combining **MediaPipe Pose 3D+2D kinematics**, **6 trained Random Forest exercise form models** (+ Stage-1 auto-detection across all 7 exercises), **3-Angle Biometric Face ID HUD (`OpenCV YuNet + SFace`)**, **SmartFit Fitness-Level AI**, **TTS voice coaching**, **H.264 post-set video analysis**, and a connected **Flutter mobile app (`formcoach/`)**.

---

## Production Model Benchmarks (5-Fold Subject-Grouped CV)

| Exercise | Production Model | Datasets | Reps / Groups | OOF Macro F1 | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Bicep Curl** | **Unified Ensemble (`RF+HGB+SVM`)** (#1) | NgoQuocBao1010 | 439 / 77 | **0.9771** | **0.9958** |
| **Squat** | **Unified Ensemble (`RF+HGB+SVM`)** (#1) | UI-PRMD | 178 / 10 | **0.9267** | **0.9204** |
| **Arm Abduction** | **Unified Ensemble (`RF+HGB+SVM`)** | UI-PRMD | 144 / 10 | **0.8894** | **0.9739** |
| **Lunge** | **Unified Ensemble (`RF+HGB+SVM`)** (#1) | UI-PRMD | 255 / 10 | **0.8283** | **0.9255** |
| **Push-Up** | Random Forest *(Deep+RF Ensemble #1: `0.6606`)* | Mendeley + REHAB24-6 | 35 subj | **0.6444** | **0.7015** |
| **Shoulder Press** | Random Forest *(SVM-RBF #1: `0.6205`)* | Mendeley | 1,961 / 25 | **0.5873** | **0.6535** |
| **Bent-Over Row** | Calibrated Biomechanical Analyzer | Live Kinematics | — | Real-Time | — |
| **Athlete Profile AI** | SmartFit Random Forest (#1) | SmartFit (1,000 athletes) | 1,000 | **0.4589** | — |

---

## Repository Structure

```text
GRADPROJECT/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI REST + /ws/live/{exercise} WebSocket server
│   │   ├── pose_engine.py    # 7-exercise + Stage-1 Auto-Detect + H.264 video annotator
│   │   ├── face_id.py        # 3-Angle YuNet + SFace 128-D biometric HUD & IoU lock-on
│   │   ├── profile_ai.py     # SmartFit Random Forest fitness-level predictor
│   │   └── database.py       # SQLite users, multi-angle Face ID templates & sessions
│   ├── models/               # Deployed Random Forest .joblib, ONNX Face ID & Pose Landmarker
│   ├── static/index.html     # 3-Tab Web Demo UI (Live Camera, Post-Set Video, 3-Angle Face ID)
│   └── requirements.txt
├── colab/
│   ├── Colab_Unified_Ensemble_Benchmark.ipynb   # Latest all-updates RF+HGB+SVM ensemble & universal test
│   ├── GymForm_MultiModel_Benchmark_Colab.ipynb # Full 10-model Deep Learning + Tabular benchmark
│   ├── Colab_Upgrade_Bicep_Curl.ipynb           # NgoQuocBao1010 Bicep Curl upgrade
│   └── Colab_Upgrade_UIPRMD_And_Row.ipynb       # UI-PRMD Squat/Lunge/Arm Abduction upgrade
├── formcoach/                # Connected Flutter mobile app
├── scripts/                  # Model installer, E2E verifier & PowerPoint deck generator
├── docs/                     # Graduation Defense Presentation (.pptx)
└── START_SERVER.bat          # One-click Windows backend launcher
```

---

## Quick Start (Local Windows / Linux / macOS)

### 1. Create Virtual Environment & Install Dependencies
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Backend & Web Demo UI
```bash
# Windows one-click:
START_SERVER.bat

# Or manually from backend/:
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000** for the 3-Tab Web Demo UI (or **http://localhost:8000/docs** for Swagger API Docs).

---

## Google Colab Training & Reproducibility
All large multi-dataset training runs are executed on **Google Colab** via the notebooks in [`colab/`](colab/):
- **`colab/Colab_Unified_Ensemble_Benchmark.ipynb`** — Benchmarks `Random Forest`, `HistGradientBoosting`, `SVM (RBF)`, `Soft-Voting Ensemble`, and a single Universal All-In-One model across all 6 exercises.
- **`colab/GymForm_MultiModel_Benchmark_Colab.ipynb`** — Full 10-model subject-grouped benchmark (`RF`, `LightGBM`, `XGBoost`, `SVM-RBF`, `1D-CNN`, `Dilated TCN`, `BiLSTM+Attn`, `2D Skeleton CNN`, `2D-CNN+BiGRU`, and Cross-Family Ensemble).

Install downloaded Colab bundles locally with:
```bash
backend\.venv\Scripts\python.exe scripts\install_model_bundles.py
```
