
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Retention & Predictive LTV Engine", layout="wide")
st.title("📊 Customer Retention & Predictive LTV Engine")
st.sidebar.header("📁 Ingestion Control Panel")

profile_file = st.sidebar.file_uploader("Upload Customer Profiles (JSON)", type=["json"])
order_file = st.sidebar.file_uploader("Upload Order History (CSV)", type=["csv"])
traffic_file = st.sidebar.file_uploader("Upload Web Traffic Logs (CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Need sample files? Run `python generate_sample_data.py` locally — "
    "it produces messy versions of all three files for testing."
)


@st.cache_data(show_spinner=False)
def load_profiles(file_bytes: bytes) -> pd.DataFrame:
    data = json.loads(file_bytes.decode("utf-8"))
    return pd.json_normalize(data)


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def clean_and_merge(profiles_df: pd.DataFrame, orders_df: pd.DataFrame, traffic_df: pd.DataFrame):
    logs = []
    anomalies_purged = 0

    # ---- Profiles ----
    before = len(profiles_df)
    profiles_df = profiles_df.dropna(subset=["customer_id"]).drop_duplicates(subset="customer_id")
    anomalies_purged += before - len(profiles_df)
    logs.append(f"[CLEAN] customer_profiles: {before} -> {len(profiles_df)} rows "
                f"(dropped nulls/duplicate customer_id)")

    if "signup_date" in profiles_df.columns:
        profiles_df["signup_date"] = pd.to_datetime(profiles_df["signup_date"], errors="coerce")
        profiles_df["cohort"] = profiles_df["signup_date"].dt.to_period("M").astype(str)
        logs.append("[TRANSFORM] Derived monthly 'cohort' label from signup_date")

    # ---- Orders ----
    before = len(orders_df)
    orders_df = orders_df.dropna(subset=["customer_id"])
    anomalies_purged += before - len(orders_df)  # null customer_id rows

    if "Amount_INR" in orders_df.columns:
        orders_df["Amount_INR"] = pd.to_numeric(orders_df["Amount_INR"], errors="coerce")
        bad_amount = orders_df["Amount_INR"].isna() | (orders_df["Amount_INR"] < 0)
        anomalies_purged += int(bad_amount.sum())
        orders_df = orders_df[~bad_amount]
    if "Purchase_Date" in orders_df.columns:
        orders_df["Purchase_Date"] = pd.to_datetime(orders_df["Purchase_Date"], errors="coerce")
        future_mask = orders_df["Purchase_Date"] > pd.Timestamp.now()
        anomalies_purged += int(future_mask.sum())
        orders_df = orders_df[~future_mask].dropna(subset=["Purchase_Date"])
    logs.append(f"[CLEAN] order_history: {before} -> {len(orders_df)} rows "
                f"(dropped null customer_id, negative/NaN amounts, future dates)")

    # ---- Web traffic ----
    before = len(traffic_df)
    traffic_df = traffic_df.dropna(subset=["customer_id"])
    anomalies_purged += before - len(traffic_df)
    logs.append(f"[CLEAN] web_traffic_logs: {before} -> {len(traffic_df)} rows (dropped null customer_id)")

    # ---- Merge ----
    logs.append("[MERGE] Joining order_history <- customer_profiles on customer_id (left join)")
    merged = orders_df.merge(profiles_df, on="customer_id", how="left")
    logs.append(f"[MERGE] Final merged dataset shape: {merged.shape[0]} rows x {merged.shape[1]} cols")
    logs.append("[DONE] Pipeline completed successfully.")

    return profiles_df, orders_df, traffic_df, merged, logs, anomalies_purged


@st.cache_data(show_spinner=False)
def compute_cohort_retention(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Retention = fraction of customers in a cohort who purchased in at
    least 2 distinct calendar months (a simple, explainable proxy for
    'came back after their first purchase').
    """
    df = merged.dropna(subset=["cohort", "Purchase_Date"]).copy()
    df["order_month"] = df["Purchase_Date"].dt.to_period("M").astype(str)

    months_per_customer = df.groupby(["cohort", "customer_id"])["order_month"].nunique()
    retained = (months_per_customer >= 2).groupby("cohort").mean() * 100
    total_customers = df.groupby("cohort")["customer_id"].nunique()

    result = pd.DataFrame({
        "cohort": retained.index,
        "retention_rate_pct": retained.values,
    }).merge(total_customers.rename("customers"), on="cohort")
    return result.sort_values("cohort")


@st.cache_data(show_spinner=False)
def compute_predicted_spend(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Per cohort: fit spend-per-month with a linear regression on the
    monthly order totals, then extrapolate one quarter (3 months) ahead.
    """
    df = merged.dropna(subset=["cohort", "Purchase_Date", "Amount_INR"]).copy()
    df["order_month"] = df["Purchase_Date"].dt.to_period("M")

    predictions = []
    for cohort, g in df.groupby("cohort"):
        monthly = g.groupby("order_month")["Amount_INR"].sum().sort_index()
        if len(monthly) < 2:
            # not enough history to fit a trend line -> use the flat average
            predicted = monthly.mean() if len(monthly) else 0
        else:
            x = np.arange(len(monthly)).reshape(-1, 1)
            y = monthly.values
            model = LinearRegression().fit(x, y)
            next_quarter_x = np.array([[len(monthly) + 2]])  # 3 months ahead
            predicted = max(model.predict(next_quarter_x)[0], 0)
        predictions.append({"cohort": cohort, "predicted_next_quarter_inr": predicted})

    return pd.DataFrame(predictions).sort_values("cohort")

if profile_file and order_file and traffic_file:

    logs = ["[INGEST] All 3 required files received."]

    try:
        profiles_raw = load_profiles(profile_file.getvalue())
        orders_raw = load_csv(order_file.getvalue())
        traffic_raw = load_csv(traffic_file.getvalue())
        logs.append(f"[LOAD] profiles={len(profiles_raw)} rows | orders={len(orders_raw)} rows | "
                    f"traffic={len(traffic_raw)} rows")

        required_order_cols = {"customer_id", "Amount_INR", "Purchase_Date"}
        missing = required_order_cols - set(orders_raw.columns)
        if missing:
            st.error(f"order_history.csv is missing required column(s): {', '.join(missing)}")
            st.stop()

        profiles_df, orders_df, traffic_df, merged, pipeline_logs, anomalies_purged = clean_and_merge(
            profiles_raw, orders_raw, traffic_raw
        )
        logs.extend(pipeline_logs)

        # Persist cleaned artefacts in session_state so other widget
        # interactions (like the chart toggle below) reuse them instantly
        # instead of touching the pipeline again.
        st.session_state["merged"] = merged
        st.session_state["anomalies_purged"] = anomalies_purged

        st.success("Pipeline executed successfully — data cleaned, validated, and merged.")

        total_revenue = merged["Amount_INR"].sum() if "Amount_INR" in merged.columns else 0
        active_cohorts = merged["cohort"].nunique() if "cohort" in merged.columns else 0

        col1, col2, col3 = st.columns(3)
        col1.metric(label="Total Revenue Checked", value=f"₹{total_revenue:,.0f}")
        col2.metric(label="Active Customer Cohorts", value=f"{active_cohorts}")
        col3.metric(label="Anomalous Records Purged", value=f"{anomalies_purged}",
                    delta="Cleaned", delta_color="inverse")

        st.markdown("---")

        st.subheader("📈 Cohort Analytics View")

        if active_cohorts == 0:
            st.warning("No valid cohorts could be derived — check that signup_date "
                       "and Purchase_Date values are present and parseable.")
        else:
            chart_choice = st.selectbox(
                "Select View Metric",
                ["Cohort Retention", "Predicted Next-Quarter Spending"],
            )

            if chart_choice == "Cohort Retention":
                retention_df = compute_cohort_retention(merged)
                fig = px.bar(
                    retention_df, x="cohort", y="retention_rate_pct",
                    text_auto=".1f",
                    labels={"cohort": "Signup Cohort", "retention_rate_pct": "Retention Rate (%)"},
                    title="Repeat-Purchase Retention Rate by Signup Cohort",
                )
                fig.update_layout(yaxis_ticksuffix="%")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Retention = % of customers in the cohort who purchased in "
                           "2+ distinct calendar months.")
            else:
                pred_df = compute_predicted_spend(merged)
                fig = px.bar(
                    pred_df, x="cohort", y="predicted_next_quarter_inr",
                    text_auto=".2s",
                    labels={"cohort": "Signup Cohort",
                            "predicted_next_quarter_inr": "Predicted Spend (₹)"},
                    title="Predicted Next-Quarter Spending by Signup Cohort (Linear Regression)",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("A per-cohort linear trend line is fit over monthly historical spend "
                           "and extrapolated 3 months forward.")

    except Exception as e:
        st.error(f"Ingestion break: {e}")
        logs.append(f"[ERROR] Pipeline halted: {str(e)}")

    with st.expander("🛠️ System Data Cleaning Logs", expanded=True):
        for entry in logs:
            st.code(entry, language="bash")

else:
    st.info("👋 Please upload all three required dataset files in the sidebar panel to launch the analysis: "
            "`customer_profiles.json`, `order_history.csv`, `web_traffic_logs.csv`.")
