"""
predict.py
Prediction helper — loads model artifacts and returns prediction + probability.
"""

import numpy as np
import joblib
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load once at import time
rf         = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
scaler     = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
le_target  = joblib.load(os.path.join(MODELS_DIR, "le_target.pkl"))
professions= joblib.load(os.path.join(MODELS_DIR, "professions.pkl"))
FEATURES   = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))

EXERCISE_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
SUGAR_ORDER    = {"low": 0, "medium": 1, "high": 2}


def predict_risk(data: dict) -> dict:
    """
    data keys:
        first_name, last_name (str)
        age (int), weight (float, kg), height (float, cm)
        exercise (str: none/low/medium/high)
        sleep (float, hours)
        sugar_intake (str: low/medium/high)
        smoking (str: yes/no)
        alcohol (str: yes/no)
        married (str: yes/no)
        profession (str)
    Returns dict with: label, probability, risk_score, feature_values
    """
    age    = float(data["age"])
    weight = float(data["weight"])
    height = float(data["height"])
    sleep  = float(data["sleep"])
    bmi    = weight / ((height / 100) ** 2)

    exercise = data["exercise"].lower().strip()
    sugar    = data["sugar_intake"].lower().strip()
    smoking  = 1 if data["smoking"].lower().strip() == "yes" else 0
    alcohol  = 1 if data["alcohol"].lower().strip() == "yes" else 0
    married  = 1 if data["married"].lower().strip() == "yes" else 0
    profession = data["profession"].lower().strip()

    bmi_sleep   = bmi * sleep
    exercise_enc = EXERCISE_ORDER.get(exercise, 1)
    sugar_enc    = SUGAR_ORDER.get(sugar, 1)

    prof_features = {f"prof_{p}": int(p == profession) for p in professions}

    row = {
        "age": age, "weight": weight, "height": height,
        "bmi": bmi, "sleep": sleep, "bmi_sleep": bmi_sleep,
        "exercise_enc": exercise_enc, "sugar_enc": sugar_enc,
        "smoking_enc": smoking, "alcohol_enc": alcohol, "married_enc": married,
        **prof_features
    }

    import pandas as pd
    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    X_sc = scaler.transform(X)

    pred_idx = rf.predict(X_sc)[0]
    proba    = rf.predict_proba(X_sc)[0]

    label = le_target.inverse_transform([pred_idx])[0]  # 'high' or 'low'
    high_idx = list(le_target.classes_).index("high")
    risk_prob = float(proba[high_idx]) * 100

    return {
        "label":      label,
        "risk_score": round(risk_prob, 1),
        "bmi":        round(bmi, 1),
        "feature_values": {
            "bmi":          round(bmi, 1),
            "age":          int(age),
            "sleep":        float(sleep),
            "exercise":     exercise,
            "sugar_intake": sugar,
            "smoking":      data["smoking"],
            "alcohol":      data["alcohol"],
        }
    }
