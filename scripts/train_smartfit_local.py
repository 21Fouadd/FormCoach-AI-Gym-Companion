"""Trains the native scikit-learn SmartFit User Profile Fitness-Level Predictor locally."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "backend" / "models" / "profile" / "smartfit_fitness_model.joblib"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SEED = 42


def main() -> None:
    sf_rows = []
    for offset in range(0, 1000, 100):
        resp = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": "Tomertg/SmartFit_Dataset",
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            break
        batch = resp.json().get("rows", [])
        if not batch:
            break
        sf_rows.extend(item["row"] for item in batch)

    sf_df = pd.DataFrame(sf_rows)
    if sf_df.empty:
        raise RuntimeError("No rows fetched from Tomertg/SmartFit_Dataset")

    target_col = next((c for c in sf_df.columns if c.lower() in ("experience", "fitness_level", "level")), None)
    if target_col is None:
        raise RuntimeError(f"Experience target column missing in {sf_df.columns.tolist()}")

    sf_df = sf_df.dropna(subset=[target_col]).copy()
    sf_df[target_col] = sf_df[target_col].astype(str).str.strip().str.title()
    sf_df = sf_df[sf_df[target_col].isin(["Beginner", "Intermediate", "Advanced"])].copy()

    candidate_num = [c for c in sf_df.columns if c.lower() in ("age", "height_cm", "weight_kg", "bmi", "height", "weight")]
    candidate_cat = [c for c in sf_df.columns if c.lower() in ("gender", "sex")]

    for c in candidate_num:
        sf_df[c] = pd.to_numeric(sf_df[c], errors="coerce")

    h_col = next((c for c in candidate_num if "height" in c.lower()), None)
    w_col = next((c for c in candidate_num if "weight" in c.lower()), None)
    if h_col and w_col and not any(c.lower() == "bmi" for c in candidate_num):
        sf_df["BMI"] = sf_df[w_col] / ((sf_df[h_col] / 100.0).clip(lower=0.5) ** 2)
        candidate_num.append("BMI")

    sf_features = candidate_num + candidate_cat
    sf_df = sf_df.dropna(subset=sf_features).reset_index(drop=True)
    print(f"SmartFit rows: {len(sf_df)} | Features: {sf_features}")

    le = LabelEncoder()
    y_sf = le.fit_transform(sf_df[target_col])
    X_sf = sf_df[sf_features]

    transformers = [("num", StandardScaler(), candidate_num)]
    if candidate_cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), candidate_cat))
    preprocessor = ColumnTransformer(transformers=transformers)

    models = {
        "Random Forest": make_pipeline(
            preprocessor, RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=SEED)
        ),
        "SVM (RBF)": make_pipeline(
            preprocessor, SVC(C=2.0, probability=True, class_weight="balanced", random_state=SEED)
        ),
        "MLP Neural Net": make_pipeline(
            preprocessor, MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED)
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    for name, pipe in models.items():
        oof_pred = np.zeros(len(y_sf), dtype=int)
        for tr_i, te_i in skf.split(X_sf, y_sf):
            pipe.fit(X_sf.iloc[tr_i], y_sf[tr_i])
            oof_pred[te_i] = pipe.predict(X_sf.iloc[te_i])
        results.append({
            "Model": name,
            "OOF Macro F1": round(float(f1_score(y_sf, oof_pred, average="macro")), 4),
            "Accuracy": round(float(accuracy_score(y_sf, oof_pred)), 4),
            "Balanced Acc": round(float(balanced_accuracy_score(y_sf, oof_pred)), 4),
        })

    board = pd.DataFrame(results).sort_values("OOF Macro F1", ascending=False).reset_index(drop=True)
    print("SmartFit 5-Fold CV Leaderboard:\n", board.to_string(index=False))

    best_name = str(board.iloc[0]["Model"])
    best_pipe = models[best_name]
    best_pipe.fit(X_sf, y_sf)

    joblib.dump(
        {
            "pipeline": best_pipe,
            "classes": le.classes_.tolist(),
            "features": sf_features,
            "numeric_features": candidate_num,
            "categorical_features": candidate_cat,
            "model_name": best_name,
            "leaderboard": board.to_dict(orient="records"),
        },
        OUT_PATH,
    )
    print("Saved native SmartFit bundle to:", OUT_PATH)

    sys.path.insert(0, str(ROOT / "backend"))
    from app.profile_ai import SmartFitProfilePredictor

    pred = SmartFitProfilePredictor()
    print("Verified SmartFitProfilePredictor().mode:", pred.mode)
    assert pred.mode.startswith("SmartFit ("), f"Unexpected mode: {pred.mode}"
    print("Sample prediction:", pred.predict(age=22, height_cm=178, weight_kg=75, gender="Male", requested_level="Auto"))


if __name__ == "__main__":
    main()
