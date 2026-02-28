"""
Phase 4: Monitoring Dashboard
Run with:   streamlit run phase4_dashboard.py
Requires:   pip install streamlit plotly scipy --break-system-packages

Features:
  • Live model performance cards
  • Experiment comparison chart
  • Data drift detection (PSI + KS test)
  • Recent predictions vs actuals
"""

import streamlit as st
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TRACKING_URI  = "file:///C:/Users/Akhiliny Vijeyagumar/OneDrive/Desktop/mlops-pipeline/mlruns"
REGISTRY_NAME = "house_price_best_model"
EXPERIMENT    = "california_house_price_prediction"
# ───────────────────────────────────────────────────────────────────────────────

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

st.set_page_config(page_title="MLOps Monitor", page_icon="🏠", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card { background:#1e1e2e; border-radius:10px; padding:16px 20px;
                 border-left:4px solid #4ade80; margin-bottom:8px; }
  .metric-value { font-size:2rem; font-weight:700; color:#4ade80; }
  .metric-label { font-size:.85rem; color:#888; }
  .drift-ok   { color:#4ade80; font-weight:600; }
  .drift-warn { color:#facc15; font-weight:600; }
  .drift-high { color:#f87171; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("🏠 House Price Prediction – MLOps Dashboard")
st.caption(f"Tracking URI: `{TRACKING_URI}`")


# ── Data helpers ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["Price"] = housing.target
    df["rooms_per_person"] = df["AveRooms"] / df["AveOccup"]
    df["bedroom_ratio"]    = df["AveBedrms"] / df["AveRooms"]
    df["income_per_room"]  = df["MedInc"] / df["AveRooms"]
    df["dist_min_city"]    = np.sqrt(
        np.minimum(
            (df["Latitude"] - 37.77)**2 + (df["Longitude"] + 122.42)**2,
            (df["Latitude"] - 34.05)**2 + (df["Longitude"] + 118.24)**2
        )
    )
    for col in ["AveRooms", "AveOccup", "Population"]:
        df = df[df[col] <= df[col].quantile(0.99)]
    return df


@st.cache_data
def get_all_runs():
    try:
        exp = client.get_experiment_by_name(EXPERIMENT)
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.r2_score DESC"]
        )
        records = []
        for r in runs:
            records.append({
                "Run Name": r.info.run_name,
                "R²":       round(r.data.metrics.get("r2_score", 0), 4),
                "RMSE":     round(r.data.metrics.get("rmse", 0), 4),
                "MAE":      round(r.data.metrics.get("mae", 0), 4),
                "CV R²":    round(r.data.metrics.get("cv_r2_mean", 0), 4),
                "Status":   r.info.status,
            })
        return pd.DataFrame(records)
    except Exception as e:
        st.warning(f"Could not fetch runs: {e}")
        return pd.DataFrame()


@st.cache_data
def get_model_versions():
    try:
        versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
        records = []
        for v in versions:
            run = client.get_run(v.run_id)
            records.append({
                "Version": v.version,
                "R²":      round(run.data.metrics.get("r2_score", 0), 4),
                "RMSE":    round(run.data.metrics.get("rmse", 0), 4),
                "Created": pd.Timestamp(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M"),
                "Aliases": ", ".join(v.aliases) if v.aliases else "—",
                "Description": v.description or "—",
            })
        return pd.DataFrame(records).sort_values("Version")
    except Exception as e:
        return pd.DataFrame()


def compute_psi(expected, actual, bins=10):
    """Population Stability Index (PSI). >0.2 = significant drift."""
    breakpoints = np.linspace(np.min(expected), np.max(expected), bins + 1)
    exp_pct  = np.histogram(expected, bins=breakpoints)[0] / len(expected) + 1e-6
    act_pct  = np.histogram(actual,   bins=breakpoints)[0] / len(actual)   + 1e-6
    return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Controls")
drift_sample_pct = st.sidebar.slider("Drift sample size (%)", 10, 100, 30, 5)
show_raw_table   = st.sidebar.checkbox("Show raw runs table", False)
st.sidebar.markdown("---")
st.sidebar.info("Refresh the page to pull latest MLflow data.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Experiment Overview", "📦 Model Registry", "🔍 Data Drift", "🎯 Live Predictions"]
)


# ── TAB 1: Experiment Overview ─────────────────────────────────────────────────
with tab1:
    runs_df = get_all_runs()

    if runs_df.empty:
        st.warning("No runs found. Check your TRACKING_URI or run Phase 1 first.")
    else:
        best = runs_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1, "Best R²",    f"{best['R²']:.4f}",  "#4ade80"),
            (c2, "Best RMSE",  f"{best['RMSE']:.4f}", "#60a5fa"),
            (c3, "Best MAE",   f"{best['MAE']:.4f}",  "#f9a8d4"),
            (c4, "Total Runs", str(len(runs_df)),      "#fbbf24"),
        ]:
            col.markdown(
                f'<div class="metric-card" style="border-color:{color}">'
                f'<div class="metric-value" style="color:{color}">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Model Comparison")
        fig = go.Figure()
        for metric, color in [("R²", "#4ade80"), ("RMSE", "#f87171"), ("MAE", "#60a5fa")]:
            fig.add_trace(go.Bar(
                x=runs_df["Run Name"], y=runs_df[metric],
                name=metric, marker_color=color,
                visible=True if metric == "R²" else "legendonly"
            ))
        fig.update_layout(
            barmode="group", template="plotly_dark",
            title="Metric Comparison Across Runs",
            xaxis_title="Model", yaxis_title="Value",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_raw_table:
            st.dataframe(runs_df, use_container_width=True)


# ── TAB 2: Model Registry ──────────────────────────────────────────────────────
with tab2:
    st.markdown("### Registered Versions")
    versions_df = get_model_versions()

    if versions_df.empty:
        st.warning("No registered models found.")
    else:
        def colour_alias(val):
            if "production" in str(val).lower():
                return "background-color: #14532d; color: #4ade80"
            elif "staging" in str(val).lower():
                return "background-color: #713f12; color: #fbbf24"
            elif "archived" in str(val).lower():
                return "background-color: #1c1917; color: #a8a29e"
            return ""

        styled = versions_df.style.applymap(colour_alias, subset=["Aliases"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # R² trend across versions
        fig2 = px.line(
            versions_df, x="Version", y="R²",
            markers=True, title="R² Across Model Versions",
            template="plotly_dark", color_discrete_sequence=["#4ade80"]
        )
        fig2.add_hline(y=0.82, line_dash="dot", line_color="#fbbf24",
                       annotation_text="Excellence threshold (0.82)")
        st.plotly_chart(fig2, use_container_width=True)


# ── TAB 3: Data Drift ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Data Drift Detection")
    st.caption("Compares training distribution vs a simulated 'new production' sample.")

    df = load_data()
    X  = df.drop("Price", axis=1)
    y  = df["Price"]
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simulate drift: production sample = a random subsample ± noise
    n_sample = max(100, int(len(X_test) * drift_sample_pct / 100))
    prod_sample = X_test.sample(n=n_sample, random_state=99).copy()

    # Add slight synthetic drift to MedInc and AveRooms to make the demo interesting
    prod_sample["MedInc"]   += np.random.normal(0.3, 0.1, n_sample)
    prod_sample["AveRooms"] += np.random.normal(0.2, 0.05, n_sample)

    features = X_train.columns.tolist()
    drift_results = []
    for feat in features:
        psi    = compute_psi(X_train[feat].values, prod_sample[feat].values)
        ks_stat, ks_p = ks_2samp(X_train[feat].values, prod_sample[feat].values)
        level = "🟢 OK" if psi < 0.1 else ("🟡 Moderate" if psi < 0.2 else "🔴 High")
        drift_results.append({"Feature": feat, "PSI": round(psi, 4),
                               "KS Stat": round(ks_stat, 4), "KS p-value": round(ks_p, 4),
                               "Drift": level})

    drift_df = pd.DataFrame(drift_results).sort_values("PSI", ascending=False)
    st.dataframe(drift_df, use_container_width=True, hide_index=True)

    # PSI bar chart
    colours = ["#f87171" if p >= 0.2 else "#facc15" if p >= 0.1 else "#4ade80"
               for p in drift_df["PSI"]]
    fig3 = go.Figure(go.Bar(x=drift_df["Feature"], y=drift_df["PSI"],
                             marker_color=colours, name="PSI"))
    fig3.add_hline(y=0.1, line_dash="dash", line_color="#facc15", annotation_text="Moderate (0.1)")
    fig3.add_hline(y=0.2, line_dash="dash", line_color="#f87171", annotation_text="High (0.2)")
    fig3.update_layout(template="plotly_dark", title="Population Stability Index per Feature",
                       xaxis_title="Feature", yaxis_title="PSI")
    st.plotly_chart(fig3, use_container_width=True)

    # Feature distribution comparison (selected feature)
    selected_feat = st.selectbox("Inspect feature distribution", features, index=0)
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=X_train[selected_feat], name="Train",
                                opacity=0.65, marker_color="#60a5fa", nbinsx=40))
    fig4.add_trace(go.Histogram(x=prod_sample[selected_feat], name="Production",
                                opacity=0.65, marker_color="#f87171", nbinsx=40))
    fig4.update_layout(barmode="overlay", template="plotly_dark",
                       title=f"Distribution: {selected_feat}")
    st.plotly_chart(fig4, use_container_width=True)


# ── TAB 4: Live Predictions ────────────────────────────────────────────────────
with tab4:
    st.markdown("### Live Predictions vs Actuals")

    df = load_data()
    X  = df.drop("Price", axis=1)
    y  = df["Price"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X)
    X_test_s = scaler.transform(X_test)

    try:
        model    = mlflow.pyfunc.load_model(f"models:/{REGISTRY_NAME}@production")
        y_pred   = model.predict(X_test_s[:200])
        y_actual = y_test.values[:200]

        r2_live  = r2_score(y_actual, y_pred)
        rmse_live = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae_live  = mean_absolute_error(y_actual, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("R² (live sample)", f"{r2_live:.4f}")
        c2.metric("RMSE (live)",       f"{rmse_live:.4f}")
        c3.metric("MAE (live)",        f"{mae_live:.4f}")

        fig5 = go.Figure()
        idx  = np.arange(len(y_pred))
        fig5.add_trace(go.Scatter(x=idx, y=y_actual, name="Actual",
                                  line=dict(color="#60a5fa")))
        fig5.add_trace(go.Scatter(x=idx, y=y_pred, name="Predicted",
                                  line=dict(color="#f87171", dash="dot")))
        fig5.update_layout(template="plotly_dark",
                           title="Predicted vs Actual (first 200 test samples)",
                           xaxis_title="Sample", yaxis_title="Price ($100k)")
        st.plotly_chart(fig5, use_container_width=True)

        # Residuals
        residuals = y_actual - y_pred
        fig6 = px.histogram(residuals, nbins=50, template="plotly_dark",
                            title="Residual Distribution",
                            color_discrete_sequence=["#a78bfa"])
        fig6.add_vline(x=0, line_dash="dash", line_color="#ffffff")
        st.plotly_chart(fig6, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load production model: {e}")
        st.info("Run Phase 2 first to register and tag a model as 'production'.")