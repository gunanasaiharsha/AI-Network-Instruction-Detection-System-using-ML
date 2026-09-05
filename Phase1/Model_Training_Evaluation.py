#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Libraries Imported Successfully")


# # Load Processed Dataset

# In[2]:


print("="*60)
print("LOADING PROCESSED DATASET")
print("="*60)

df = pd.read_csv("processed_dataset.csv")

print("Dataset Loaded Successfully")

print()

print("Dataset Shape :", df.shape)


# # Dataset Information

# In[3]:


print("="*60)
print("DATASET INFORMATION")
print("="*60)

print("Rows :", df.shape[0])

print("Columns :", df.shape[1])

print()

df.head()


# # Separate Features and Target

# In[4]:


print("="*60)
print("SEPARATING FEATURES & TARGET")
print("="*60)

X = df.drop("Label", axis=1)

y = df["Label"]

print("Feature Shape :", X.shape)

print("Target Shape :", y.shape)


# # Train-Test Split

# In[5]:


print("="*60)
print("TRAIN TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("Training Samples :", X_train.shape)

print("Testing Samples :", X_test.shape)


# # Check Class Distribution

# In[6]:


print("="*60)
print("TRAINING CLASS DISTRIBUTION")
print("="*60)

print(y_train.value_counts())


# # Display Train/Test Percentage

# In[7]:


print("="*60)
print("DATA SPLIT")
print("="*60)

print("Training :", round(len(X_train)/len(df)*100,2),"%")

print("Testing  :", round(len(X_test)/len(df)*100,2),"%")


# # Display Feature Names

# In[8]:


print("="*60)
print("TOTAL FEATURES")
print("="*60)

print("Number of Features :", len(X.columns))

print()

for i,col in enumerate(X.columns,1):

    print(f"{i}. {col}")


# # Train Random Forest Model

# In[9]:


print("=" * 60)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 60)

rf_model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    n_jobs=-1

)

rf_model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully")


# # Predict Test Data

# In[10]:


print("=" * 60)
print("PREDICTING TEST DATA")
print("=" * 60)

y_pred = rf_model.predict(X_test)

print("Prediction Completed Successfully")


# # Model Accuracy

# In[11]:


print("=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest Accuracy : {accuracy * 100:.2f}%")


# # Classification Report

# In[12]:


print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(y_test, y_pred))


# # Confusion Matrix

# In[13]:


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")

plt.show()


# # Feature Importance

# In[14]:


importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf_model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("=" * 60)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 60)

print(importance.head(20))


# # Save Model

# In[15]:


joblib.dump(
    rf_model,
    "ids_model.pkl"
)

print("Random Forest Model Saved Successfully")


# # Save Model Metadata

# In[16]:


import json

metadata = {

    "Model": "Random Forest",

    "Accuracy": round(accuracy * 100, 2),

    "Features": len(X.columns),

    "Classes": 8,

    "Training Samples": len(X_train),

    "Testing Samples": len(X_test)

}

with open(
    "model_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

print("Model Metadata Saved Successfully")


# # Final Summary

# In[17]:


print("=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)

print("Model Name        : Random Forest")
print(f"Accuracy          : {accuracy * 100:.2f}%")
print("Number of Features:", len(X.columns))
print("Number of Classes :", 8)
print("Training Samples  :", len(X_train))
print("Testing Samples   :", len(X_test))

print("\nGenerated Files")

print("1. ids_model.pkl")
print("2. model_metadata.json")

print("=" * 70)


# In[ ]:




