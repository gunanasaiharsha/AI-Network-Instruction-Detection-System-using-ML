import os
import pandas as pd
from datetime import datetime

from flask import (
    request,
    render_template
)

from backend.predictor import (
    predict_dataframe,
    prediction_summary
)

# ============================================================
# LOG FILE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

CSV_LOG = os.path.join(
    LOG_DIR,
    "csv_prediction_logs.csv"
)

# ============================================================
# UPLOAD DATASET
# ============================================================

def upload_dataset():

    predictions = None

    if request.method == "POST":

        file = request.files.get("file")

        if file and file.filename != "":

            df = pd.read_csv(file)

            # Predict
            result = predict_dataframe(df)

            # Summary for webpage
            predictions = prediction_summary(result)

            # Create log dataframe
            log_df = pd.DataFrame({
                "Time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(result),
                "Prediction": result["Prediction"]
            })


            if os.path.exists(CSV_LOG):

                try:

                    old = pd.read_csv(CSV_LOG)

                    log_df = pd.concat(
                        [old, log_df],
                        ignore_index=True
                    )

                except pd.errors.EmptyDataError:

        # File exists but has no data yet.
                    pass

            log_df.to_csv(
                CSV_LOG,
                index=False
            )

    return render_template(
        "upload.html",
        predictions=predictions
    )