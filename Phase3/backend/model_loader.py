import os
import json
import joblib

# ============================================================
# MODEL DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models"
)

# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load(
    os.path.join(MODEL_PATH, "ids_model.pkl")
)

label_encoder = joblib.load(
    os.path.join(MODEL_PATH, "label_encoder.pkl")
)

with open(
    os.path.join(MODEL_PATH, "feature_columns.json"),
    "r"
) as f:

    feature_columns = json.load(f)

print("Model Loaded Successfully")