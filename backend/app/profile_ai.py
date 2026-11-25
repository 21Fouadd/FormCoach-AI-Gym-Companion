"""AI Module #3 — SmartFit User-Profile Fitness Level Predictor & Personalized Plan Engine."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"


def build_training_plan(fitness_level: str) -> dict[str, Any]:
    lvl = fitness_level.strip().title()
    if lvl == "Advanced":
        return {"sets": 4, "target_reps": 12, "rest_sec": 60, "focus": "Full depth, strict tempo & progressive overload"}
    if lvl == "Intermediate":
        return {"sets": 3, "target_reps": 10, "rest_sec": 75, "focus": "Consistent range of motion & symmetry"}
    return {"sets": 3, "target_reps": 8, "rest_sec": 90, "focus": "Form mastery, controlled descent & posture"}


class SmartFitProfilePredictor:
    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None
        self.mode = "bmi_age_fallback"
        self._try_load()

    def _try_load(self) -> None:
        candidates = [
            MODELS_ROOT / "profile" / "smartfit_fitness_model.joblib",
            MODELS_ROOT / "squat" / "smartfit_fitness_model.joblib",
            MODELS_ROOT / "smartfit_fitness_model.joblib",
        ]
        for path in candidates:
            if path.exists():
                try:
                    self._bundle = joblib.load(path)
                    self.mode = f"SmartFit ({self._bundle.get('model_name', 'ML')})"
                    return
                except Exception:
                    continue

    def predict(
        self,
        age: int,
        height_cm: float,
        weight_kg: float,
        gender: str = "Other",
        requested_level: str = "Auto",
    ) -> dict[str, Any]:
        req = (requested_level or "Auto").strip().title()
        if req in ("Beginner", "Intermediate", "Advanced"):
            return {
                "fitness_level": req,
                "predicted_by": "user_selected",
                "plan": build_training_plan(req),
            }

        bmi = float(weight_kg) / ((max(float(height_cm), 50.0) / 100.0) ** 2)

        if self._bundle is not None:
            try:
                pipe = self._bundle["pipeline"]
                classes = self._bundle["classes"]
                features = self._bundle["features"]
                row: dict[str, Any] = {}
                for col in features:
                    low = col.lower()
                    if "age" in low:
                        row[col] = float(age)
                    elif "height" in low:
                        row[col] = float(height_cm)
                    elif "weight" in low:
                        row[col] = float(weight_kg)
                    elif low == "bmi":
                        row[col] = bmi
                    elif low in ("gender", "sex"):
                        row[col] = gender
                    else:
                        row[col] = 0
                pred_idx = int(pipe.predict(pd.DataFrame([row]))[0])
                level = str(classes[pred_idx])
                return {
                    "fitness_level": level,
                    "predicted_by": self.mode,
                    "bmi": round(bmi, 1),
                    "plan": build_training_plan(level),
                }
            except Exception:
                pass

        # Clean fallback heuristic when SmartFit .joblib bundle hasn't been dropped in yet
        if 19.0 <= bmi <= 25.5 and 18 <= age <= 35:
            level = "Intermediate"
        elif bmi > 29.0 or age > 50 or age < 16:
            level = "Beginner"
        else:
            level = "Intermediate"

        return {
            "fitness_level": level,
            "predicted_by": self.mode,
            "bmi": round(bmi, 1),
            "plan": build_training_plan(level),
        }
