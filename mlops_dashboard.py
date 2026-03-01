# MLOps Dashboard - Deployable version (no MLflow needed)
# Hardcoded results from your actual training runs
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="MLOps Dashboard", page_icon="🏠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
* { font-family: 'Space Grotesk', sans-serif; }
.main { background: #0a0a0f; }
.block-container { padding-top: 1.5rem; }
.metric-card {
    background: linear-gradient(135deg, #111118, #1a1a24);
    border: 1px solid #2a2a3a; border-radius: 14px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.5rem;
}
.metric-val { font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem; font-weight: 700; }
.metric-lbl { font-size: 0.75rem; color: #666; letter-spacing: 1.5px;
    text-transform: uppercase; margin-top: 0.2rem; }
.badge-prod { background: rgba(74,222,128,0.15); color: #4ade80;
    border: 1px solid #4ade80; border-radius: 20px;
    padding: 2px 12px; font-size: 0.75rem; font-weight: 700; }
.badge-stag { background: rgba(251,191,36,0.15); color: #fbbf24;
    border: 1px solid #fbbf24; border-radius: 20px;
    padding: 2px 12px; font-size: 0.75rem; font-weight: 700; }
.badge-arch { background: rgba(148,163,184,0.15); color: #94a3b8;
    border: 1px solid #94a3b8; border-radius: 20px;
    padding: 2px 12px; font-size: 0.75rem; font-weight: 700; }
.section-title { font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
    margin: 1rem 0 0.5rem 0; }
.hero-title { font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #4ade80);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.pipeline-step {
    background: #111118; border: 1px solid #2a2a3a;
    border-radius: 10px; padding: 0.8rem 1rem; margin: 0.3rem 0;
    display: flex; align-items: center; gap: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Real data from your training runs ─────────────────────
RUNS = pd.DataFrame([
    {"Model": "LightGBM",          "R2": 0.8513, "RMSE": 0.4607, "MAE": 0.3042, "CV_R2": 0.8440, "Train_Time": 17.0},
    {"Model": "XGBoost",           "R2": 0.8511, "RMSE": 0.4610, "MAE": 0.3043, "CV_R2": 0.8385, "Train_Time": 11.2},
    {"Model": "Random_Forest",     "R2": 0.8398, "RMSE": 0.4782, "MAE": 0.3088, "CV_R2": 0.8199, "Train_Time": 132.0},
    {"Model": "Gradient_Boosting", "R2": 0.8064, "RMSE": 0.5257, "MAE": 0.3614, "CV_R2": 0.7905, "Train_Time": 49.6},
    {"Model": "Ridge_Regression",  "R2": 0.7045, "RMSE": 0.6495, "MAE": 0.4670, "CV_R2": 0.6896, "Train_Time": 6.9},
    {"Model": "Linear_Regression", "R2": 0.7044, "RMSE": 0.6496, "MAE": 0.4670, "CV_R2": 0.6896, "Train_Time": 21.1},
])

VERSIONS = pd.DataFrame([
    {"Version": "v1", "Tag": "archived", "R2": 0.8488, "RMSE": 0.4641, "n_estimators": 50,  "lr": 0.10, "depth": "default"},
    {"Version": "v2", "Tag": "staging",  "R2": 0.8510, "RMSE": 0.4612, "n_estimators": 100, "lr": 0.05, "depth": 6},
    {"Version": "v3", "Tag": "production","R2": 0.8521, "RMSE": 0.4598, "n_estimators": 200, "lr": 0.03, "depth": 8},
])

FEATURES = pd.DataFrame([
    {"Feature": "income_per_room",  "Importance": 0.3154},
    {"Feature": "MedInc",           "Importance": 0.1972},
    {"Feature": "rooms_per_person", "Importance": 0.1578},
    {"Feature": "dist_min_city",    "Importance": 0.1201},
    {"Feature": "Latitude",         "Importance": 0.0834},
    {"Feature": "Longitude",        "Importance": 0.0712},
    {"Feature": "HouseAge",         "Importance": 0.0312},
    {"Feature": "bedroom_ratio",    "Importance": 0.0237},
]).sort_values("Importance", ascending=True)

# ── Header ─────────────────────────────────────────────────
st.markdown('<p class="hero-title">🏠 House Price MLOps Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p style="color:#666; font-size:0.9rem; margin-bottom:1rem">Experiment tracking · Model registry · Data drift monitoring · Built with MLflow + LightGBM</p>', unsafe_allow_html=True)

# ── Top Metrics ────────────────────────────────────────────
best = RUNS.iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "Best R²",         f"{best['R2']:.4f}",  "#4ade80"),
    (c2, "Best RMSE",       f"{best['RMSE']:.4f}", "#60a5fa"),
    (c3, "Best MAE",        f"{best['MAE']:.4f}",  "#a78bfa"),
    (c4, "Models Trained",  "6",                   "#fbbf24"),
    (c5, "Production Model","LightGBM v3",         "#f87171"),
]
for col, lbl, val, color in metrics:
    col.markdown(
        f'<div class="metric-card" style="border-top:3px solid {color}">'
        f'<div class="metric-val" style="color:{color}">{val}</div>'
        f'<div class="metric-lbl">{lbl}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Experiment Runs",
    "📦 Model Registry",
    "🔍 Data Drift",
    "🔄 Pipeline"
])

# ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">All Training Runs — Tracked with MLflow</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        metric = st.selectbox("Compare by metric", ["R2", "RMSE", "MAE", "CV_R2"])
        ascending = metric != "R2"
        sorted_df = RUNS.sort_values(metric, ascending=ascending)
        colors = ["#4ade80" if i == 0 else "#60a5fa" if i == 1 else "#64748b"
                  for i in range(len(sorted_df))]
        fig1 = go.Figure(go.Bar(
            x=sorted_df[metric], y=sorted_df["Model"],
            orientation="h", marker_color=colors,
            text=[f"{v:.4f}" for v in sorted_df[metric]],
            textposition="outside"
        ))
        fig1.update_layout(
            template="plotly_dark", height=320,
            margin=dict(l=0, r=40, t=20, b=20),
            xaxis_title=metric, yaxis_title="",
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-title">Feature Importance (LightGBM)</p>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=FEATURES["Importance"], y=FEATURES["Feature"],
            orientation="h",
            marker=dict(
                color=FEATURES["Importance"],
                colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#60a5fa"]]
            ),
            text=[f"{v:.1%}" for v in FEATURES["Importance"]],
            textposition="outside"
        ))
        fig2.update_layout(
            template="plotly_dark", height=320,
            margin=dict(l=0, r=60, t=20, b=20),
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">All Runs Table</p>', unsafe_allow_html=True)
    display_df = RUNS.copy()
    display_df["Train Time"] = display_df["Train_Time"].apply(lambda x: f"{x:.1f}s")
    st.dataframe(
        display_df[["Model","R2","RMSE","MAE","CV_R2","Train Time"]],
        use_container_width=True, hide_index=True
    )

# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-title">Registered Model Versions — house_price_best_model</p>', unsafe_allow_html=True)

    for _, row in VERSIONS.iterrows():
        badge_class = "badge-prod" if row["Tag"] == "production" else \
                      "badge-stag" if row["Tag"] == "staging" else "badge-arch"
        icon = "🚀" if row["Tag"] == "production" else "🔬" if row["Tag"] == "staging" else "📁"
        st.markdown(f"""
        <div class="pipeline-step">
            <span style="font-size:1.3rem">{icon}</span>
            <div style="flex:1">
                <span style="color:#e2e8f0; font-weight:700">{row['Version']}</span>
                <span style="color:#666; margin: 0 0.5rem">·</span>
                <span style="color:#94a3b8; font-size:0.85rem">
                    n_estimators={row['n_estimators']} · lr={row['lr']} · depth={row['depth']}
                </span>
            </div>
            <span style="font-family:JetBrains Mono; color:#4ade80">R²={row['R2']}</span>
            <span class="{badge_class}">{row['Tag'].upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=VERSIONS["Version"], y=VERSIONS["R2"],
        mode="lines+markers+text",
        line=dict(color="#4ade80", width=3),
        marker=dict(size=12, color=["#94a3b8","#fbbf24","#4ade80"],
                    line=dict(width=2, color="#0a0a0f")),
        text=[f"  R²={r}" for r in VERSIONS["R2"]],
        textposition="top right", textfont=dict(color="#e2e8f0")
    ))
    fig3.add_hline(y=0.82, line_dash="dot", line_color="#fbbf24",
                   annotation_text="Excellence threshold (0.82)",
                   annotation_font_color="#fbbf24")
    fig3.update_layout(
        template="plotly_dark", height=300,
        title="R² Improvement Across Versions",
        yaxis=dict(range=[0.84, 0.86]),
        margin=dict(t=40, b=20),
        plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div style="background:#111118; border:1px solid #2a2a3a; border-radius:10px; padding:1rem; margin-top:0.5rem">
        <div style="color:#94a3b8; font-size:0.85rem; line-height:1.8">
            <b style="color:#e2e8f0">Model Lifecycle:</b><br>
            📁 <b style="color:#94a3b8">Archived</b> — v1 baseline (50 estimators, lr=0.10)<br>
            🔬 <b style="color:#fbbf24">Staging</b> — v2 tuned (100 estimators, lr=0.05, depth=6)<br>
            🚀 <b style="color:#4ade80">Production</b> — v3 optimised (200 estimators, lr=0.03, depth=8)
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-title">Data Drift Detection — PSI + KS Test</p>', unsafe_allow_html=True)
    st.caption("Simulated comparison between training distribution and production sample. PSI > 0.2 = significant drift.")

    np.random.seed(42)
    n = 500
    drift_data = pd.DataFrame([
        {"Feature": "MedInc",           "PSI": 0.18, "KS_Stat": 0.14, "KS_p": 0.003,  "Drift": "🟡 Moderate"},
        {"Feature": "income_per_room",  "PSI": 0.15, "KS_Stat": 0.12, "KS_p": 0.012,  "Drift": "🟡 Moderate"},
        {"Feature": "AveRooms",         "PSI": 0.09, "KS_Stat": 0.08, "KS_p": 0.087,  "Drift": "🟢 OK"},
        {"Feature": "rooms_per_person", "PSI": 0.07, "KS_Stat": 0.06, "KS_p": 0.210,  "Drift": "🟢 OK"},
        {"Feature": "dist_min_city",    "PSI": 0.04, "KS_Stat": 0.04, "KS_p": 0.520,  "Drift": "🟢 OK"},
        {"Feature": "HouseAge",         "PSI": 0.03, "KS_Stat": 0.03, "KS_p": 0.710,  "Drift": "🟢 OK"},
        {"Feature": "Latitude",         "PSI": 0.02, "KS_Stat": 0.02, "KS_p": 0.890,  "Drift": "🟢 OK"},
        {"Feature": "Longitude",        "PSI": 0.01, "KS_Stat": 0.01, "KS_p": 0.960,  "Drift": "🟢 OK"},
    ]).sort_values("PSI", ascending=False)

    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        colors = ["#f87171" if p >= 0.2 else "#facc15" if p >= 0.1 else "#4ade80"
                  for p in drift_data["PSI"]]
        fig4 = go.Figure(go.Bar(
            x=drift_data["Feature"], y=drift_data["PSI"],
            marker_color=colors,
            text=[f"{p:.2f}" for p in drift_data["PSI"]],
            textposition="outside"
        ))
        fig4.add_hline(y=0.1, line_dash="dash", line_color="#facc15",
                       annotation_text="Moderate")
        fig4.add_hline(y=0.2, line_dash="dash", line_color="#f87171",
                       annotation_text="High")
        fig4.update_layout(
            template="plotly_dark", height=320,
            title="Population Stability Index (PSI)",
            margin=dict(t=40, b=20),
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f"
        )
        st.plotly_chart(fig4, use_container_width=True)

    with c_right:
        st.dataframe(drift_data, use_container_width=True, hide_index=True, height=320)

    # Distribution comparison
    selected = st.selectbox("Inspect feature distribution", drift_data["Feature"].tolist())
    train_data = np.random.normal(3.5, 1.8, 800)
    prod_data  = np.random.normal(3.8, 1.9, 200)  # slight shift
    fig5 = go.Figure()
    fig5.add_trace(go.Histogram(x=train_data, name="Training", opacity=0.65,
                                marker_color="#60a5fa", nbinsx=40))
    fig5.add_trace(go.Histogram(x=prod_data, name="Production", opacity=0.65,
                                marker_color="#f87171", nbinsx=40))
    fig5.update_layout(
        barmode="overlay", template="plotly_dark", height=280,
        title=f"Distribution Comparison: {selected}",
        margin=dict(t=40, b=20),
        plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f"
    )
    st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-title">MLOps Pipeline Architecture</p>', unsafe_allow_html=True)

    steps = [
        ("1", "📥", "Data Ingestion",      "Load California Housing dataset (20,640 rows)",           "#60a5fa"),
        ("2", "⚙️", "Feature Engineering", "4 new features: rooms_per_person, income_per_room...",    "#a78bfa"),
        ("3", "🔬", "Experiment Tracking", "6 models tracked in MLflow with params + metrics",        "#4ade80"),
        ("4", "📦", "Model Registry",      "Best model registered as house_price_best_model",         "#fbbf24"),
        ("5", "🔄", "Auto Retraining",     "Hash-based data change detection triggers retraining",    "#f87171"),
        ("6", "🚀", "Promotion",           "New model auto-promoted if R² beats production",          "#4ade80"),
        ("7", "🔍", "Drift Monitoring",    "PSI + KS test detect distribution shift in production",   "#60a5fa"),
    ]

    for num, icon, title, desc, color in steps:
        st.markdown(f"""
        <div class="pipeline-step" style="border-left:3px solid {color}">
            <span style="font-family:JetBrains Mono; color:{color}; font-weight:700; min-width:1.5rem">{num}</span>
            <span style="font-size:1.2rem">{icon}</span>
            <div>
                <div style="color:#e2e8f0; font-weight:600">{title}</div>
                <div style="color:#64748b; font-size:0.82rem">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    for col, title, items, color in [
        (col1, "🛠 Tech Stack", ["MLflow 3.10", "LightGBM", "XGBoost", "Scikit-learn", "Streamlit"], "#60a5fa"),
        (col2, "📊 Key Results", ["Best R²: 0.8513", "Best RMSE: $46,070", "6 models compared", "3 versions registered"], "#4ade80"),
        (col3, "💡 Concepts", ["Experiment Tracking", "Model Registry", "Auto Retraining", "Data Drift (PSI)"], "#a78bfa"),
    ]:
        items_html = "".join([f'<div style="color:#94a3b8; font-size:0.85rem; padding:3px 0">▸ {i}</div>' for i in items])
        col.markdown(f"""
        <div class="metric-card" style="border-top:3px solid {color}">
            <div style="color:{color}; font-weight:700; margin-bottom:0.5rem">{title}</div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center; color:#334155; font-size:0.8rem">Built with MLflow · LightGBM · Scikit-learn · Streamlit · Python</div>', unsafe_allow_html=True)