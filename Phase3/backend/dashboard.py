import os
import pandas as pd


import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

LOG_FILE = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "Phase2",
        "logs",
        "nids_attack_logs.csv"
    )
)


def get_dashboard_data():

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

    total_flows = len(df)

    total_attacks = len(
        df[df["Status"] == "ALERT"]
    )

    benign = len(
        df[df["Attack"] == "BENIGN"]
    )

    if total_flows > 0:

        accuracy = round(
            (benign / total_flows) * 100,
            2
        )

    else:

        accuracy = 0

    recent_logs = df.tail(10).iloc[::-1]

    return {

        "total_flows": total_flows,

        "total_attacks": total_attacks,

        "benign": benign,

        "accuracy": accuracy,

        "recent_logs": recent_logs

    }