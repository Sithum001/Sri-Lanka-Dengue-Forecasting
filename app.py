import json
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "artifacts")

st.set_page_config(
    page_title="Sri Lanka Dengue Forecasting",
    layout="wide"
)

st.title("Sri Lanka Weekly Dengue Forecasting")
st.caption(
    "Colombo | Research demonstration, "
    "not a public-health alert system"
)

# Check trained artifacts
if not os.path.exists(os.path.join(OUT, "metadata.json")):
    st.error(
        "Missing trained artifacts. Copy the artifacts "
        "folder next to app.py."
    )
    st.stop()

# Load results
with open(os.path.join(OUT, "metadata.json")) as f:
    meta = json.load(f)

scores = pd.read_csv(os.path.join(OUT, "test_metrics.csv"))

preds = pd.read_csv(
    os.path.join(OUT, "test_predictions.csv"),
    parse_dates=["week_start_date"]
)

importance = pd.read_csv(os.path.join(OUT, "feature_importance.csv"))


# ---------- Helpers for loading .pkl models ----------
@st.cache_resource
def load_pkl(filename):
    with open(os.path.join(OUT, filename), "rb") as f:
        return pickle.load(f)


MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "Bagging": "bagging.pkl",
    "Extra Trees": "extra_trees.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
    "CatBoost": "catboost.pkl",
    "ANN": "ann.pkl",
}
STACK_BASES = ["Random Forest", "XGBoost", "CatBoost"]


@st.cache_data
def latest_feature_values(features):
    """Rebuild the next-week feature row from dengue_data.csv
    (same logic as Step 36 in the notebook)."""
    csv_path = os.path.join(ROOT, "dengue_data.csv")
    if not os.path.exists(csv_path):
        return {f: 0.0 for f in features}

    df = pd.read_csv(csv_path, parse_dates=["week_start_date"])
    c = df[df["district"] == "Colombo"].sort_values(
        "week_start_date").reset_index(drop=True)

    latest = c.iloc[-1]
    future_date = latest["week_start_date"] + pd.Timedelta(days=7)
    new = {}
    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        new[f"cases_lag_{lag}"] = float(c["cases"].iloc[-lag])
    for col in ["rainfall_mm", "temp_avg_c", "humidity_pct"]:
        for lag in [1, 2, 4]:
            new[f"{col}_lag_{lag}"] = float(c[col].iloc[-lag])
    for window in [4, 8, 12]:
        new[f"cases_roll_{window}"] = float(c["cases"].iloc[-window:].mean())
    w = int(future_date.isocalendar().week)
    new["week_sin"] = float(np.sin(2 * np.pi * w / 52.18))
    new["week_cos"] = float(np.cos(2 * np.pi * w / 52.18))
    return {f: new.get(f, 0.0) for f in features}


# Sidebar
st.sidebar.header("Controls")

model = st.sidebar.selectbox(
    "Select model",
    scores["Model"].tolist()
)

start = preds["week_start_date"].min().date()
end = preds["week_start_date"].max().date()

period = st.sidebar.date_input(
    "Test period",
    value=(start, end),
    min_value=start,
    max_value=end
)

filtered = preds.copy()

if isinstance(period, tuple) and len(period) == 2:
    filtered = preds[
        preds["week_start_date"].dt.date.between(*period)
    ]

# Performance metrics
a, b, c = st.columns(3)

row = scores.set_index("Model").loc[model]

a.metric("Test MAE", f"{row['MAE']:.2f}")
b.metric("Test RMSE", f"{row['RMSE']:.2f}")
c.metric("Test R²", f"{row['R2']:.3f}")

# Actual vs predicted
st.subheader("Actual vs Predicted Dengue Cases")

chart = filtered[
    ["week_start_date", "Actual", model]
].melt(
    id_vars="week_start_date",
    var_name="Series",
    value_name="Cases"
)

fig = px.line(
    chart,
    x="week_start_date",
    y="Cases",
    color="Series",
    title="Weekly Dengue Cases"
)

st.plotly_chart(fig, use_container_width=True)

# Model comparison
st.subheader("Ensemble Model Comparison")

st.dataframe(
    scores.style.format({
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R2": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

fig = px.bar(
    scores.sort_values("RMSE"),
    x="Model",
    y="RMSE",
    title="Model RMSE Comparison (Lower is Better)"
)

st.plotly_chart(fig, use_container_width=True)

# Feature importance
st.subheader("Random Forest Feature Importance")

fig = px.bar(
    importance.head(12).sort_values("importance"),
    x="importance",
    y="feature",
    orientation="h",
    title="Most Important Features"
)

st.plotly_chart(fig, use_container_width=True)

# ---------- Live prediction using the saved .pkl models ----------
st.subheader("Live Prediction (uses the saved .pkl models)")
st.caption(
    "Defaults are the latest observed week in the CSV. Edit any value "
    "(for example last week's cases or rainfall) and click Predict."
)

features = meta["features"]
defaults = latest_feature_values(tuple(features))

input_df = pd.DataFrame({
    "feature": features,
    "value": [defaults[f] for f in features],
})
edited = st.data_editor(
    input_df,
    disabled=["feature"],
    hide_index=True,
    use_container_width=True,
    key="feature_editor",
)

if st.button("Predict next-week cases"):
    X = pd.DataFrame([edited["value"].astype(float).tolist()],
                     columns=features)
    results = {}
    try:
        for name, fname in MODEL_FILES.items():
            results[name] = float(max(0, load_pkl(fname).predict(X)[0]))

        stack_meta = load_pkl("stacking_meta.pkl")
        results["Stacking"] = float(max(0, stack_meta.predict(
            np.array([[results[n] for n in STACK_BASES]]))[0]))

        wt = meta["rf_blend_weight"]
        results["RF + ANN"] = float(max(
            0, wt * results["Random Forest"] + (1 - wt) * results["ANN"]))

        live = pd.DataFrame({
            "Model": list(results.keys()),
            "Forecast cases": [round(v) for v in results.values()],
        })
        st.dataframe(live, hide_index=True, use_container_width=True)
        st.plotly_chart(
            px.bar(live, x="Model", y="Forecast cases",
                   title="Live Forecast by Model"),
            use_container_width=True,
        )
        st.caption("ARIMA, SARIMA and ARIMA + RF Hybrid are evaluated in "
                   "the notebook only, because they need the full time "
                   "series rather than a single feature row.")
    except FileNotFoundError as ex:
        st.error(f"Missing model file: {ex}. Check the artifacts folder.")
    except Exception as ex:
        st.error(f"Prediction failed: {ex}")

# Next-week forecast
st.subheader("Next Week: Saved Model Forecasts")

st.caption(
    "These forecasts use the latest observed CSV row, "
    "not live surveillance data."
)

future = pd.DataFrame([
    {
        "Model": name,
        "Forecast cases": round(value)
    }
    for name, value in
    meta["next_week_tabular_predictions"].items()
])

st.write(
    f"Latest observed week: {meta['last_observed_date']} "
    f"| Forecast week: {meta['next_week_date']}"
)

st.dataframe(
    future,
    hide_index=True,
    use_container_width=True
)

