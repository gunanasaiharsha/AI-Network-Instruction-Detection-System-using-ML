import os
import pandas as pd

from flask import render_template

from backend.model_loader import BASE_DIR

# ============================================================
# LIVE MONITOR
# ============================================================

def live_monitor():


    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    LOG_FILE = os.path.join(
        BASE_DIR,
        "..",
        "Phase2",
        "logs",
        "nids_attack_logs.csv"
    )

    try:

        df = pd.read_csv(LOG_FILE)

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

    live_data = df.tail(20).iloc[::-1]

    return render_template(

        "live.html",

        live_data=live_data

    )