import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

st.set_page_config(
    page_title="Credit Risk — MLOps Dashboard",
    page_icon="📊",
    layout="wide"
)

PREDICTION_LOG = os.path.join(os.path.dirname(__file__), "prediction_log.csv")
DRIFT_SUMMARY  = os.path.join(os.path.dirname(__file__), "reports", "latest_drift.json")
METRICS_PATH   = os.path.join(os.path.dirname(__file__), "..", "artifacts", "latest_metrics.json")
PROMOTION_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "last_promotion.json")


# ─────────────────────────────────────────────
#  Load data helpers
# ─────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_prediction_log():
    if not os.path.exists(PREDICTION_LOG):
        return pd.DataFrame()
    df = pd.read_csv(PREDICTION_LOG, parse_dates=["timestamp"])
    return df


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
#  Dashboard layout
# ─────────────────────────────────────────────

st.title("Credit Risk Model — Operations Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

logs      = load_prediction_log()
metrics   = load_json(METRICS_PATH)
promotion = load_json(PROMOTION_PATH)
drift     = load_json(DRIFT_SUMMARY)

# ── Section 1: Model Info ─────────────────────
st.header("Model Status")

col1, col2, col3, col4 = st.columns(4)

if metrics:
    col1.metric("Test AUC",       f"{metrics.get('test_auc', 'N/A')}")
    col2.metric("Test F1",        f"{metrics.get('test_f1', 'N/A')}")
    col3.metric("CV AUC (mean)",  f"{metrics.get('cv_auc_mean', 'N/A')} ± {metrics.get('cv_auc_std', 'N/A')}")
else:
    col1.metric("Test AUC", "N/A — run train.py")

if promotion:
    col4.metric("Production Version", f"v{promotion.get('promoted_version', '?')}")

# ── Section 2: Prediction Stats ───────────────
st.header("Prediction Activity")

if logs.empty:
    st.info("No predictions logged yet. Start the API with: `python src/predict.py` and send some requests.")
else:
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # top metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Predictions",  f"{len(logs):,}")
        m2.metric("Approved",           f"{(logs.decision=='APPROVE').sum():,}")
        m3.metric("Under Review",       f"{(logs.decision=='REVIEW').sum():,}")
        m4.metric("Rejected",           f"{(logs.decision=='REJECT').sum():,}")
        m5.metric("Avg Risk Score",     f"{logs.probability.mean():.3f}")

        # decision breakdown
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Decision breakdown")
            decision_counts = logs["decision"].value_counts().reset_index()
            decision_counts.columns = ["decision", "count"]
            color_map = {"APPROVE": "#2ecc71", "REVIEW": "#f39c12", "REJECT": "#e74c3c"}
            fig = px.pie(
                decision_counts, names="decision", values="count",
                color="decision", color_discrete_map=color_map
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Risk score distribution")
            fig2 = px.histogram(
                logs, x="probability", nbins=40,
                color="decision", color_discrete_map=color_map,
                labels={"probability": "Default Probability"}
            )
            st.plotly_chart(fig2, use_container_width=True)

        # time series if enough data
        if "timestamp" in logs.columns and len(logs) > 10:
            st.subheader("Risk score over time")
            logs_sorted = logs.sort_values("timestamp")
            fig3 = px.scatter(
                logs_sorted, x="timestamp", y="probability",
                color="decision", color_discrete_map=color_map,
                opacity=0.6
            )
            fig3.add_hline(y=0.75, line_dash="dash", line_color="red",
                           annotation_text="Reject threshold")
            fig3.add_hline(y=0.45, line_dash="dash", line_color="orange",
                           annotation_text="Review threshold")
            st.plotly_chart(fig3, use_container_width=True)

    except ImportError:
        st.warning("Install plotly for charts: pip install plotly")
        st.dataframe(logs.tail(50))

# ── Section 3: Drift Status ───────────────────
st.header("Data Drift Status")

if drift is None:
    st.info("No drift report found. Run: `python monitoring/drift_report.py`")
else:
    if drift.get("alert"):
        st.error(
            f"Data drift detected! "
            f"{drift.get('drifted_features', '?')} of {drift.get('total_features', '?')} features drifted. "
            f"Consider retraining the model."
        )
    else:
        st.success(
            f"No significant drift detected. "
            f"{drift.get('drifted_features', 0)} of {drift.get('total_features', '?')} features show drift."
        )

    d1, d2, d3 = st.columns(3)
    d1.metric("Features Checked",  drift.get("total_features", "N/A"))
    d2.metric("Drifted Features",  drift.get("drifted_features", "N/A"))
    d3.metric("Share Drifted",     f"{drift.get('share_drifted', 0):.1%}")

    if drift.get("top_drifted"):
        st.subheader("Top drifted features")
        drift_df = pd.DataFrame(drift["top_drifted"])
        st.dataframe(drift_df, use_container_width=True)

# ── Section 4: Raw log preview ────────────────
if not logs.empty:
    st.header("Recent Predictions")
    st.dataframe(logs.tail(20), use_container_width=True)

# ── Refresh button ─────────────────────────────
if st.button("Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()