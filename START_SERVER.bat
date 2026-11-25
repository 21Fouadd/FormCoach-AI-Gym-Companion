@echo off
title AI Gym Trainer Backend (Port 8000)
cd /d "%~dp0"
if not exist "backend\models\profile\smartfit_fitness_model.joblib" (
    echo [1/2] Training native SmartFit Fitness-Level AI model...
    "backend\.venv\Scripts\python.exe" "scripts\train_smartfit_local.py"
)
echo [2/2] Starting FastAPI AI Server on http://localhost:8000 ...
"backend\.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir "backend" --host 0.0.0.0 --port 8000 --reload
pause
