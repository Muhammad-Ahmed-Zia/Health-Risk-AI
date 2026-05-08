"""
train_model.py
Trains a Random Forest classifier on the Lifestyle & Health Risk dataset
and saves the model + preprocessor artifacts to disk.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ── 1. Load data ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset.csv"))

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
df = df.dropna(subset=["health_risk"])

for col in ["age", "weight", "height", "bmi"]:
    df[col] = df[col].fillna(df[col].mean())

df["sleep"] = df["sleep"].fillna(df["sleep"].median())

# Label-encode binary categoricals
le_map = {}
for col in ["smoking", "alcohol", "married"]:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col].str.lower().str.strip())
    le_map[col] = le

# Ordinal-encode exercise
exercise_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
df["exercise_enc"] = df["exercise"].str.lower().str.strip().map(exercise_order)

# Ordinal-encode sugar_intake
sugar_order = {"low": 0, "medium": 1, "high": 2}
df["sugar_enc"] = df["sugar_intake"].str.lower().str.strip().map(sugar_order)

# One-hot encode profession
professions = sorted(df["profession"].str.lower().str.strip().unique())
for p in professions:
    df[f"prof_{p}"] = (df["profession"].str.lower().str.strip() == p).astype(int)

# BMI-sleep interaction
df["bmi_sleep"] = df["bmi"] * df["sleep"]

# Target
le_target = LabelEncoder()
df["target"] = le_target.fit_transform(df["health_risk"].str.lower().str.strip())

# Feature list (must match predict.py)
FEATURES = (
    ["age", "weight", "height", "bmi", "sleep", "bmi_sleep",
     "exercise_enc", "sugar_enc",
     "smoking_enc", "alcohol_enc", "married_enc"]
    + [f"prof_{p}" for p in professions]
)

X = df[FEATURES]
y = df["target"]

# ── 3. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. Train Random Forest ────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_sc, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"\n[SUCCESS] Random Forest Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# ── 7. Save artifacts ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(rf,         os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(scaler,     os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(le_target,  os.path.join(MODELS_DIR, "le_target.pkl"))
joblib.dump(professions,os.path.join(MODELS_DIR, "professions.pkl"))
joblib.dump(FEATURES,   os.path.join(MODELS_DIR, "features.pkl"))

print(f"\n[SUCCESS] Artifacts saved to {MODELS_DIR}/")
print(f"   Features ({len(FEATURES)}): {FEATURES}")
