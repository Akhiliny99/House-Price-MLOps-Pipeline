House-Price-MLOps-Pipeline
 
A production-grade MLOps pipeline and monitoring dashboard for California house price prediction.

Live Dashboard Demo : https://house-price-mlops-pipeline-nhukqoihxanizkvya2aflq.streamlit.app/

Features :

6 ML models trained and compared (Linear, Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM)

Model Registry with versioning and production/staging/archived stages

Data Drift Detection using PSI (Population Stability Index) and KS Test

Live Predictions with custom input sliders

Interactive charts powered by Plotly

MLOps Pipeline :

Phase 1MLflow experiment tracking — 6 models logged

Phase 2Model Registry with versioning, lifecycle stages

Phase 3Automated retraining on data change detectionPhase 4This Streamlit dashboard

Tech Stack

ML: scikit-learn, XGBoost, LightGBM

Tracking: MLflow

Dashboard: Streamlit, Plotly

Drift: SciPy (KS Test), PSI
