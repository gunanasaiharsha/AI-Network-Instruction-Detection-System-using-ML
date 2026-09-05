import pandas as pd

from backend.model_loader import (

    model,

    label_encoder,

    feature_columns

)

# ============================================================
# PREDICT DATAFRAME
# ============================================================

def predict_dataframe(df):

    X = df[feature_columns]

    predictions = model.predict(X)

    labels = label_encoder.inverse_transform(
        predictions
    )

    df["Prediction"] = labels

    return df


# ============================================================
# PREDICTION SUMMARY
# ============================================================

def prediction_summary(df):

    summary = (

        df["Prediction"]

        .value_counts()

        .reset_index()

    )

    summary.columns = [

        "Prediction",

        "Count"

    ]

    return summary