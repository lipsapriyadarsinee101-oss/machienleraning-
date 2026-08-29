"""Streamlit dashboard for customer churn prioritization."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
SCORES_PATH = ROOT / "data" / "processed" / "customer_risk_scores.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.set_page_config(page_title="SpringPod Customer Intelligence", page_icon="📊", layout="wide")
st.title("SpringPod Customer Intelligence")
st.caption("Portfolio demonstration using synthetic data - not actual SpringPod customer information")

if not SCORES_PATH.exists() or not METRICS_PATH.exists():
    st.error("Run `python src/data_generation.py` and `python src/train.py` before starting the dashboard.")
    st.stop()

scores = pd.read_csv(SCORES_PATH)
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

regions = st.sidebar.multiselect("Region", sorted(scores["region"].unique()), default=sorted(scores["region"].unique()))
risk_levels = st.sidebar.multiselect(
    "Risk band", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"]
)
filtered = scores[scores["region"].isin(regions) & scores["risk_band"].isin(risk_levels)]

selected_metrics = metrics["models"][metrics["selected_model"]]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers shown", f"{len(filtered):,}")
c2.metric("Average risk", f"{filtered['churn_probability'].mean():.1%}" if len(filtered) else "-")
c3.metric("High/Critical", int(filtered["risk_band"].isin(["High", "Critical"]).sum()))
c4.metric("Model ROC-AUC", f"{selected_metrics['roc_auc']:.3f}")

left, right = st.columns(2)
with left:
    counts = filtered["risk_band"].value_counts().reindex(["Critical", "High", "Medium", "Low"], fill_value=0).reset_index()
    counts.columns = ["risk_band", "customers"]
    st.plotly_chart(
        px.bar(counts, x="risk_band", y="customers", color="risk_band", title="Customers by risk band"),
        use_container_width=True,
    )
with right:
    region_risk = filtered.groupby("region", as_index=False)["churn_probability"].mean()
    st.plotly_chart(
        px.bar(region_risk, x="region", y="churn_probability", title="Average predicted risk by region"),
        use_container_width=True,
    )

st.subheader("Prioritized retention queue")
display_columns = [
    "customer_id", "region", "plan_type", "churn_probability", "risk_band",
    "days_since_last_activity", "support_tickets_90d", "recommended_action"
]
st.dataframe(
    filtered[display_columns].sort_values("churn_probability", ascending=False),
    use_container_width=True,
    hide_index=True,
    column_config={"churn_probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="%.1%%")},
)

