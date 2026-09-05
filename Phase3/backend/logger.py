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
# HISTORY
# ============================================================

def attack_history():

    history_type = request.args.get(
        "type",
        "live"
    )

    if history_type == "csv":

        log_file = CSV_LOG

        columns = [
            "Time",
            "Prediction"
        ]

    else:

        log_file = LIVE_LOG

        columns = [
            "Time",
            "Flow_ID",
            "Status",
            "Attack",
            "Confidence (%)"
        ]

    try:

        history_df = pd.read_csv(log_file)

    except (FileNotFoundError, pd.errors.EmptyDataError):

        history_df = pd.DataFrame(
            columns=columns
        )

    return render_template(

        "history.html",

        history=history_df.iloc[::-1],

        total_records=len(history_df),

        history_type=history_type

    )