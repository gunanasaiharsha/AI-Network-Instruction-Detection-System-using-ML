#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries

# In[45]:


import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import joblib

warnings.filterwarnings("ignore")

print("Libraries Imported Successfully")


# # Set Dataset Path

# In[64]:


dataset_path = r"C:\Users\gunan\OneDrive\Documents\Desktop\IDS(dataset)"

print("Dataset Path")

print(dataset_path)


# # Load All Dataset Files

# In[65]:


dos = pd.read_csv(os.path.join(dataset_path, "DoS.csv"))

ddos = pd.read_csv(os.path.join(dataset_path, "DDoS.csv"))

portscan = pd.read_csv(os.path.join(dataset_path, "PortScan.csv"))

bruteforce = pd.read_csv(os.path.join(dataset_path, "BruteForce.csv"))

webattacks = pd.read_csv(os.path.join(dataset_path, "WebAttacks.csv"))

botnet = pd.read_csv(os.path.join(dataset_path, "Botnet-traffic.csv"))

infiltration = pd.read_csv(os.path.join(dataset_path, "Infilteration-attack.csv"))

benign = pd.read_csv(os.path.join(dataset_path, "Benign-traffic.csv"))

print("All Dataset Files Loaded Successfully")


# # Verify Dataset Shapes

# In[66]:


print("="*60)

print("DATASET SHAPES")

print("="*60)

print("DoS Dataset          :", dos.shape)

print("DDoS Dataset         :", ddos.shape)

print("PortScan Dataset     :", portscan.shape)

print("BruteForce Dataset   :", bruteforce.shape)

print("WebAttacks Dataset   :", webattacks.shape)

print("Botnet Dataset       :", botnet.shape)

print("Infiltration Dataset :", infiltration.shape)

print("Benign Dataset       :", benign.shape)


# # Merge All Datasets

# In[67]:


merged_df = pd.concat(

    [

        benign,

        dos,

        ddos,

        portscan,

        bruteforce,

        webattacks,

        botnet,

        infiltration

    ],

    ignore_index=True

)

print("Datasets Merged Successfully")

print()

print("Dataset Shape :", merged_df.shape)


# # Clean Column Names

# In[68]:


merged_df.columns = merged_df.columns.str.strip()

print("Column Names Cleaned Successfully")


# # Select 55 Features Before Saving

# In[72]:


# ============================================================
# SELECT 55 FEATURES + LABEL
# ============================================================

selected_features = [

    "ACK Flag Count",
    "Flow IAT Min",
    "Min Packet Length",
    "Flow Duration",
    "URG Flag Count",
    "Bwd Header Length",
    "Total Fwd Packets",
    "Bwd IAT Std",
    "Bwd IAT Mean",
    "RST Flag Count",
    "CWE Flag Count",
    "Flow IAT Std",
    "Bwd Packet Length Min",
    "Fwd URG Flags",
    "Fwd IAT Total",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd URG Flags",
    "Flow IAT Max",
    "Fwd Packet Length Min",
    "Fwd IAT Max",
    "Flow Bytes/s",
    "Bwd Packet Length Mean",
    "Bwd IAT Max",
    "Packet Length Variance",
    "Packet Length Mean",
    "Fwd IAT Min",
    "Bwd Packets/s",
    "Fwd Packets/s",
    "Total Length of Bwd Packets",
    "Bwd IAT Total",
    "Fwd Packet Length Max",
    "Packet Length Std",
    "Avg Fwd Segment Size",
    "Average Packet Size",
    "FIN Flag Count",
    "Fwd IAT Std",
    "PSH Flag Count",
    "Fwd Header Length",
    "SYN Flag Count",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Flow Packets/s",
    "Destination Port",
    "Fwd Packet Length Mean",
    "Bwd PSH Flags",
    "Fwd Packet Length Std",
    "Down/Up Ratio",
    "Bwd Packet Length Max",
    "Avg Bwd Segment Size",
    "Fwd IAT Mean",
    "Max Packet Length",
    "ECE Flag Count",
    "Flow IAT Mean",
    "Bwd Packet Length Std",

    # Target Column
    "Label"

]

# Create dataset with only required columns
merged_df = merged_df[selected_features].copy()

print("=" * 60)
print("55 FEATURES SELECTED SUCCESSFULLY")
print("=" * 60)

print("Dataset Shape :", merged_df.shape)
print("Input Features :", merged_df.shape[1] - 1)
print("Target Column :", merged_df.columns[-1])

print("\nFirst 5 Rows")
print(merged_df.head())


# # Dataset Information

# In[73]:


print("="*70)

print("CICIDS2017 DATASET INFORMATION")

print("="*70)

print("Shape :", merged_df.shape)

print()

print("Number of Columns :", len(merged_df.columns))

print()

merged_df.info()


# # First Five Rows

# In[74]:


print("First Five Rows")

print(merged_df.head())


# # Last Five Rows

# In[75]:


print("Last Five Rows")

print(merged_df.tail())


# # Summary Statistics

# In[76]:


merged_df.describe().T


# # Check Missing Values

# In[77]:


missing_values = merged_df.isnull().sum()

missing_values = missing_values[missing_values > 0]

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("Columns with Missing Values :", len(missing_values))

if len(missing_values) > 0:
    print(missing_values.sort_values(ascending=False))
else:
    print("No Missing Values Found")


# # Check Duplicate Records

# In[78]:


duplicates = merged_df.duplicated().sum()

print("=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

print("Duplicate Rows :", duplicates)


# # Check Infinite Values

# In[79]:


numeric_df = merged_df.select_dtypes(include=np.number)

infinite_values = np.isinf(numeric_df).sum().sum()

print("=" * 60)
print("INFINITE VALUES")
print("=" * 60)

print("Infinite Values :", infinite_values)


# # Original Class Distribution

# In[80]:


print("=" * 60)
print("ORIGINAL CLASS DISTRIBUTION")
print("=" * 60)

print(merged_df["Label"].value_counts())


# # Plot Original Class Distribution

# In[81]:


plt.figure(figsize=(12,6))

counts = merged_df["Label"].value_counts()

plt.bar(counts.index, counts.values)

plt.xticks(rotation=45, ha="right")

plt.xlabel("Attack Type")

plt.ylabel("Number of Samples")

plt.title("Original Dataset Distribution")

plt.tight_layout()

plt.show()


# # Remove Heartbleed

# In[82]:


merged_df = merged_df[
    merged_df["Label"] != "Heartbleed"
]

print("=" * 60)
print("HEARTBLEED REMOVED")
print("=" * 60)

print(merged_df["Label"].value_counts())


# # Convert 15 Classes into 8 Classes

# In[83]:


label_mapping = {

    "BENIGN": "BENIGN",

    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",

    "DDoS": "DDoS",

    "PortScan": "PortScan",

    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",

    "Web Attack � Brute Force": "WebAttack",
    "Web Attack � Sql Injection": "WebAttack",
    "Web Attack � XSS": "WebAttack",

    "Bot": "Botnet",

    "Infiltration": "Infiltration"

}

merged_df["Label"] = merged_df["Label"].replace(label_mapping)

print("Attack Categories Grouped Successfully")


# # Verify New 8 Classes

# In[84]:


print("=" * 60)
print("NEW CLASS DISTRIBUTION")
print("=" * 60)

print(merged_df["Label"].value_counts())


# # Balance Dataset (30,000 Samples)

# In[85]:


MAX_SAMPLES = 30000

balanced_data = []

for label in merged_df["Label"].unique():

    class_df = merged_df[
        merged_df["Label"] == label
    ]

    if len(class_df) > MAX_SAMPLES:

        class_df = resample(
            class_df,
            replace=False,
            n_samples=MAX_SAMPLES,
            random_state=42
        )

    balanced_data.append(class_df)

balanced_df = pd.concat(
    balanced_data,
    ignore_index=True
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# # Verify Balanced Dataset

# In[86]:


print("=" * 60)
print("BALANCED DATASET")
print("=" * 60)

print("Dataset Shape :", balanced_df.shape)

print()

print(balanced_df["Label"].value_counts())


# # Plot Balanced Dataset

# In[87]:


plt.figure(figsize=(10,5))

counts = balanced_df["Label"].value_counts()

plt.bar(
    counts.index,
    counts.values
)

plt.xticks(rotation=30)

plt.xlabel("Attack Category")

plt.ylabel("Number of Samples")

plt.title("Balanced Dataset")

plt.tight_layout()

plt.show()


# # Remove Duplicate Records

# In[88]:


print("=" * 60)
print("REMOVING DUPLICATE RECORDS")
print("=" * 60)

print("Shape Before :", balanced_df.shape)

balanced_df.drop_duplicates(inplace=True)

print("Shape After  :", balanced_df.shape)

print("\nDuplicate Records Removed Successfully")


# # Replace Infinite Values

# In[89]:


print("=" * 60)
print("REPLACING INFINITE VALUES")
print("=" * 60)

balanced_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

print("Infinite Values Replaced Successfully")


# # Fill Missing Values

# In[90]:


print("=" * 60)
print("FILLING MISSING VALUES")
print("=" * 60)

print("Missing Values Before :")

print(
    balanced_df.isnull().sum().sum()
)

balanced_df.fillna(
    0,
    inplace=True
)

print()

print("Missing Values After :")

print(
    balanced_df.isnull().sum().sum()
)


# # Label Encoding

# In[91]:


print("=" * 60)
print("LABEL ENCODING")
print("=" * 60)

encoder = LabelEncoder()

balanced_df["Label"] = encoder.fit_transform(
    balanced_df["Label"]
)

print("Encoding Completed\n")

for i, label in enumerate(encoder.classes_):

    print(f"{i} : {label}")


# # Separate Features and Target

# In[92]:


print("=" * 60)
print("SEPARATING FEATURES & TARGET")
print("=" * 60)

X = balanced_df.drop(
    "Label",
    axis=1
)

y = balanced_df["Label"]

print("Feature Shape :", X.shape)

print("Target Shape  :", y.shape)


# # Feature Scaling

# In[93]:


print("=" * 60)
print("FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature Scaling Completed Successfully")


# # Create Processed Dataset

# In[94]:


processed_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

processed_df["Label"] = y.values

print(processed_df.head())


# # Save Dataset

# In[95]:


processed_df.to_csv(
    "processed_dataset.csv",
    index=False
)

print("Processed Dataset Saved Successfully")


# # Save Scaler

# In[96]:


joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Scaler Saved Successfully")


# # Save Label Encoder

# In[97]:


joblib.dump(
    encoder,
    "label_encoder.pkl"
)

print("Label Encoder Saved Successfully")


# # Save Feature Names

# In[98]:


import json

feature_columns = list(X.columns)

with open(
    "feature_columns.json",
    "w"
) as f:

    json.dump(
        feature_columns,
        f,
        indent=4
    )

print("Feature Columns Saved Successfully")


# # Final Summary

# In[99]:


print("=" * 70)

print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")

print("=" * 70)

print("Dataset Shape :", processed_df.shape)

print("Total Features :", len(feature_columns))

print("Total Classes :", len(encoder.classes_))

print()

print("Classes")

for i, label in enumerate(encoder.classes_):

    print(f"{i} : {label}")

print()

print("Files Generated")

print("processed_dataset.csv")

print("scaler.pkl")

print("label_encoder.pkl")

print("feature_columns.json")

print("=" * 70)


# In[100]:


import matplotlib.pyplot as plt

# ==========================================
# Final 8-Class Distribution
# ==========================================

class_counts = balanced_df["Label"].value_counts().sort_values(ascending=False)

print("=" * 60)
print("FINAL DATASET CLASS DISTRIBUTION")
print("=" * 60)

print(class_counts)

# ==========================================
# Bar Graph
# ==========================================

plt.figure(figsize=(10,6))

bars = plt.bar(
    class_counts.index,
    class_counts.values,
    edgecolor="black"
)

# Display values on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

plt.title("Final Balanced IDS Dataset (8 Classes)", fontsize=15)

plt.xlabel("Attack Category", fontsize=12)

plt.ylabel("Number of Samples", fontsize=12)

plt.xticks(rotation=20)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:
