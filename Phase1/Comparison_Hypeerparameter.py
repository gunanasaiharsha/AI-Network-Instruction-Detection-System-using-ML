#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# # Load Dataset

# In[2]:


dataset = pd.read_csv("processed_dataset.csv")

X = dataset.drop("Label", axis=1)

y = dataset["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)


# # Train Multiple Algorithms

# In[3]:


models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(random_state=42),

    "KNN":
        KNeighborsClassifier(),

    "SVM":
        SVC(),

    "Naive Bayes":
        GaussianNB()

}

results = []

for name, model in models.items():

    print("="*60)
    print(name)
    print("="*60)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    print("Accuracy :", accuracy)

    results.append([name, accuracy])


# # Accuracy Comparison

# In[4]:


result_df = pd.DataFrame(

    results,

    columns=[
        "Model",
        "Accuracy"
    ]

)

result_df = result_df.sort_values(

    by="Accuracy",

    ascending=False

)

print(result_df)


# # Random Forest Hyperparameter Tuning

# In[6]:


from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier(random_state=42)

param_grid = {

    "n_estimators":[100,200,300,500],

    "max_depth":[10,20,30,None],

    "min_samples_split":[2,5,10],

    "min_samples_leaf":[1,2,4],

    "max_features":["sqrt","log2"]

}

random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_grid,

    n_iter=10,

    cv=5,

    scoring="accuracy",

    random_state=42,

    n_jobs=-1

)

random_search.fit(

    X_train,

    y_train

)

print(random_search.best_params_)

print(random_search.best_score_)


# # Train Best Random Forest

# In[8]:


best_rf = random_search.best_estimator_

best_rf.fit(

    X_train,

    y_train

)

prediction = best_rf.predict(

    X_test

)

accuracy = accuracy_score(

    y_test,

    prediction

)

print("="*60)

print("FINAL RANDOM FOREST")

print("="*60)

print("Accuracy :", accuracy)

print()

print(classification_report(

    y_test,

    prediction

))


# In[ ]:




