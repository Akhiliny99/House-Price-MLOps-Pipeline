

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb

print("Libraries imported!")

import os
mlflow.set_tracking_uri("file:///C:/Users/Akhiliny Vijeyagumar/OneDrive/Desktop/mlops-pipeline/mlruns")
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

# Feature Engineering
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
    upper = df[col].quantile(0.99)
    df = df[df[col] <= upper]

X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"Data ready: {X_train.shape[0]} train, {X_test.shape[0]} test")


mlflow.set_experiment("california_house_price_prediction")
print("MLflow experiment created!")

def evaluate_and_log(model, model_name, X_train, X_test, y_train, y_test, params=None):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        cv_mean = cv_scores.mean()
        cv_std  = cv_scores.std()

        mlflow.log_param("model_type", model_name)
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)
        mlflow.log_metric("r2_score",   r2)
        mlflow.log_metric("rmse",       rmse)
        mlflow.log_metric("mae",        mae)
        mlflow.log_metric("cv_r2_mean", cv_mean)
        mlflow.log_metric("cv_r2_std",  cv_std)

        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        if hasattr(model, "feature_importances_"):
            fig, ax = plt.subplots(figsize=(10, 6))
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
            importances.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title(f"{model_name} Feature Importance")
            plt.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close()

        status = "EXCELLENT!" if r2 > 0.82 else "Good" if r2 > 0.7 else "OK"
        print(f"{model_name}: R2={r2:.4f} {status} | RMSE={rmse:.4f} | CV={cv_mean:.4f}")
        return {"model_name": model_name, "model": model, "r2": r2, "rmse": rmse, "mae": mae, "cv_r2": cv_mean}

print("Training 6 models with MLflow tracking...")
results = []
results.append(evaluate_and_log(LinearRegression(), "Linear_Regression", X_train_s, X_test_s, y_train, y_test))
results.append(evaluate_and_log(Ridge(alpha=1.0), "Ridge_Regression", X_train_s, X_test_s, y_train, y_test, params={"alpha": 1.0}))
results.append(evaluate_and_log(RandomForestRegressor(n_estimators=100, random_state=42), "Random_Forest", X_train_s, X_test_s, y_train, y_test, params={"n_estimators": 100}))
results.append(evaluate_and_log(GradientBoostingRegressor(n_estimators=100, random_state=42), "Gradient_Boosting", X_train_s, X_test_s, y_train, y_test, params={"n_estimators": 100}))
results.append(evaluate_and_log(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0), "XGBoost", X_train_s, X_test_s, y_train, y_test, params={"n_estimators": 100, "learning_rate": 0.1}))
results.append(evaluate_and_log(lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1), "LightGBM", X_train_s, X_test_s, y_train, y_test, params={"n_estimators": 100}))

results_df = pd.DataFrame([{"Model": r["model_name"], "R2": round(r["r2"],4), "RMSE": round(r["rmse"],4), "MAE": round(r["mae"],4), "CV_R2": round(r["cv_r2"],4)} for r in results]).sort_values("R2", ascending=False)
print(results_df.to_string(index=False))
best = max(results, key=lambda x: x["r2"])
best_name = best["model_name"]
best_r2 = best["r2"]
print(f"Best Model: {best_name} R2={best_r2:.4f}")


client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("california_house_price_prediction")
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.r2_score DESC"], max_results=1)
if runs:
    best_run_id = runs[0].info.run_id
    model_uri = f"runs:/{best_run_id}/model"
    try:
        registered = mlflow.register_model(model_uri=model_uri, name="house_price_best_model")
        print(f"Registered: house_price_best_model v{registered.version}")
    except Exception as e:
        print(f"Registration: {e}")


fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric, color in zip(axes, ["R2", "RMSE", "MAE"], ["#4ade80", "#f87171", "#60a5fa"]):
    sorted_df = results_df.sort_values(metric, ascending=(metric != "R2"))
    ax.barh(sorted_df["Model"], sorted_df[metric], color=color)
    ax.set_title(metric, fontweight="bold")
plt.suptitle("MLflow Experiment - Model Comparison", fontweight="bold")
plt.tight_layout()
plt.savefig("mlflow_comparison.png", dpi=150)
plt.close()
print("Saved: mlflow_comparison.png")
print("PHASE 1 COMPLETE! Now run: mlflow ui")
print("Then open: http://localhost:5000")