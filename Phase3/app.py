from flask import Flask, render_template

from backend.dashboard import get_dashboard_data
from backend.upload_handler import upload_dataset
from backend.live_monitor import live_monitor
from backend.logger import attack_history
from backend.analytics import analytics_dashboard

app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    dashboard = get_dashboard_data()

    return render_template(

        "index.html",

        total_flows=dashboard["total_flows"],

        total_attacks=dashboard["total_attacks"],

        benign=dashboard["benign"],

        accuracy=dashboard["accuracy"],

        recent_logs=dashboard["recent_logs"]

    )


# ============================================================
# DATASET UPLOAD
# ============================================================

@app.route("/upload", methods=["GET", "POST"])
def upload():

    return upload_dataset()


# ============================================================
# LIVE MONITORING
# ============================================================

@app.route("/live")
def live():

    return live_monitor()


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    return attack_history()


# ============================================================
# ANALYTICS
# ============================================================

@app.route("/analytics")
def analytics():

    return analytics_dashboard()


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template("about.html")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )