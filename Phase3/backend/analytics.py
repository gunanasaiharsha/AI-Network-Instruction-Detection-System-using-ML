import os
import pandas as pd

from flask import (
    render_template,
    request
)

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ============================================================
# LOG FILES
# ============================================================

LIVE_LOG = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "Phase2",
        "logs",
        "nids_attack_logs.csv"
    )
)

CSV_LOG = os.path.join(
    BASE_DIR,
    "logs",
    "csv_prediction_logs.csv"
)

# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

def analytics_dashboard():

    analytics_type = request.args.get(
        "type",
        "live"
    )

    # ========================================================
    # CSV ANALYTICS
    # ========================================================

    if analytics_type == "csv":

        try:

            df = pd.read_csv(CSV_LOG)

        except (FileNotFoundError, pd.errors.EmptyDataError):

            df = pd.DataFrame(
                columns=[
                    "Time",
                    "Prediction"
                ]
            )

        attack_counts = df["Prediction"].value_counts()

        labels = attack_counts.index.tolist()

        values = attack_counts.values.tolist()

        total_records = len(df)

        benign = len(
            df[df["Prediction"] == "BENIGN"]
        )

        total_attacks = total_records - benign

        return render_template(

            "analytics.html",

            analytics_type=analytics_type,

            labels=labels,

            values=values,

            total_records=total_records,

            total_attacks=total_attacks,

            benign=benign

        )

    # ========================================================
    # LIVE ANALYTICS
    # ========================================================

    else:

        try:

            df = pd.read_csv(LIVE_LOG)

        except (FileNotFoundError, pd.errors.EmptyDataError):

            df = pd.DataFrame(
                columns=[
                    "Time",
                    "Flow_ID",
                    "Status",
                    "Attack",
                    "Confidence (%)"
                ]
            )

        attack_counts = df["Attack"].value_counts()

        labels = attack_counts.index.tolist()

        values = attack_counts.values.tolist()

        total_records = len(df)

        benign = len(
            df[df["Attack"] == "BENIGN"]
        )

        total_attacks = total_records - benign

        return render_template(

            "analytics.html",

            analytics_type=analytics_type,

            labels=labels,

            values=values,

            total_records=total_records,

            total_attacks=total_attacks,

            benign=benign

        )