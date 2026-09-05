#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# =====================================================
# Load Processed Dataset
# =====================================================

df = pd.read_csv("processed_dataset.csv")

print("Dataset Shape :", df.shape)

# =====================================================
# Number of Samples Required
# =====================================================

sample_counts = {
    0: 80,   # BENIGN
    1: 10,   # Botnet
    2: 15,   # BruteForce
    3: 25,   # DDoS
    4: 30,   # DoS
    5: 5,    # Infiltration
    6: 25,   # PortScan
    7: 10    # WebAttack
}

# =====================================================
# Create Testing Dataset
# =====================================================

test_data = []

for label, count in sample_counts.items():

    class_df = df[df["Label"] == label]

    # Take all rows if available rows are less
    count = min(count, len(class_df))

    sampled = class_df.sample(
        n=count,
        random_state=42
    )

    test_data.append(sampled)

# =====================================================
# Merge All Classes
# =====================================================

test_df = pd.concat(
    test_data,
    ignore_index=True
)

# Shuffle Dataset
test_df = test_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# =====================================================
# Save Dataset
# =====================================================

test_df.to_csv(
    "test_dataset.csv",
    index=False
)

# =====================================================
# Label Mapping
# =====================================================

label_map = {
    0: "BENIGN",
    1: "Botnet",
    2: "BruteForce",
    3: "DDoS",
    4: "DoS",
    5: "Infiltration",
    6: "PortScan",
    7: "WebAttack"
}

# =====================================================
# Display Summary
# =====================================================

print("\n" + "="*60)
print("TEST DATASET CREATED SUCCESSFULLY")
print("="*60)

print("Shape :", test_df.shape)

print("\nClass Distribution\n")

distribution = test_df["Label"].value_counts().sort_index()

for label, count in distribution.items():
    print(f"{label_map[label]:15s} : {count}")

print("\nSaved File : test_dataset0.csv")


# In[ ]:





# # Import Libraries

# # With Label

# In[36]:


import pandas as pd
import joblib
import json

from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("ids_model.pkl")

label_encoder = joblib.load("label_encoder.pkl")

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# ============================================================
# LOAD TEST DATASET
# ============================================================

test_df = pd.read_csv("test_dataset.csv")

# ============================================================
# FEATURES & LABEL
# ============================================================

X_test = test_df[feature_columns]

y_test = test_df["Label"]

# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(X_test)

confidence = model.predict_proba(X_test).max(axis=1) * 100

# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame({

    "Actual": label_encoder.inverse_transform(y_test),

    "Prediction": label_encoder.inverse_transform(predictions),

    "Confidence (%)": confidence.round(2)

})

print(results)

print("\nAccuracy :", accuracy_score(y_test, predictions))

print("\nPrediction Count\n")

print(results["Prediction"].value_counts())


# # Without Label

# In[37]:


import pandas as pd
import joblib
import json

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("ids_model.pkl")

label_encoder = joblib.load("label_encoder.pkl")

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# ============================================================
# LOAD TEST DATASET
# ============================================================

test_df = pd.read_csv("test_dataset01.csv")

# ============================================================
# FEATURES
# ============================================================

X_test = test_df[feature_columns]

# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(X_test)

confidence = model.predict_proba(X_test).max(axis=1) * 100

# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame({

    "Prediction": label_encoder.inverse_transform(predictions),

    "Confidence (%)": confidence.round(2)

})

print(results)

print("\nPrediction Count\n")

print(results["Prediction"].value_counts())


# In[ ]:




