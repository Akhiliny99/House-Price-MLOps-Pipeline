
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import lightgbm as lgb


TRACKING_URI  = "file:///C:/Users/Akhiliny Vijeyagumar/OneDrive/Desktop/mlops-pipeline/mlruns"
EXPERIMENT    = "california_house_price_prediction"
REGISTRY_NAME = "house_price_best_model"


mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


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

X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"Data ready: {X_train_s.shape[0]} train | {X_test_s.shape[0]} test")



def train_and_register(params: dict, version_tag: str, description: str):
    """Train a model, log to MLflow, register in Model Registry."""
    mlflow.set_experiment(EXPERIMENT)
    run_name = f"LightGBM_{version_tag}"

    with mlflow.start_run(run_name=run_name) as run:
        model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        mlflow.log_params({"model_type": "LightGBM", "version_tag": version_tag, **params})
        mlflow.log_metrics({"r2_score": r2, "rmse": rmse, "mae": mae})
        mlflow.set_tag("version_description", description)

        mlflow.sklearn.log_model(model, "model")

        model_uri = f"runs:/{run.info.run_id}/model"
        reg = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
        print(f"  ✅ Registered {REGISTRY_NAME} v{reg.version} | R²={r2:.4f} | RMSE={rmse:.4f}")
        return reg.version, r2, rmse

print("\n📦  Registering 3 versions…")

v1, r2_v1, _ = train_and_register(
    params={"n_estimators": 50, "learning_rate": 0.1},
    version_tag="v1_baseline",
    description="Baseline: 50 estimators, lr=0.1"
)

v2, r2_v2, _ = train_and_register(
    params={"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6},
    version_tag="v2_tuned",
    description="Tuned: 100 estimators, lr=0.05, depth=6"
)

v3, r2_v3, _ = train_and_register(
    params={"n_estimators": 200, "learning_rate": 0.03, "max_depth": 8, "num_leaves": 63},
    version_tag="v3_optimised",
    description="Optimised: 200 estimators, lr=0.03, depth=8, leaves=63"
)



print("\n🚦  Updating lifecycle stages…")


client.set_registered_model_alias(REGISTRY_NAME, "archived_v1", str(v1))
print(f"  📁 v{v1} tagged as 'archived_v1'")


client.set_registered_model_alias(REGISTRY_NAME, "staging", str(v2))
print(f"  🔬 v{v2} promoted to 'staging'")


client.set_registered_model_alias(REGISTRY_NAME, "production", str(v3))
print(f"  🚀 v{v3} promoted to 'production'")



client.update_registered_model(
    name=REGISTRY_NAME,
    description="LightGBM regressor for California house price prediction. "
                "Trained on engineered features including distance-to-city and income ratios."
)

for version, tag_val in [(v1, "archived"), (v2, "staging"), (v3, "production")]:
    client.set_model_version_tag(REGISTRY_NAME, str(version), "stage", tag_val)
    client.update_model_version(
        name=REGISTRY_NAME,
        version=str(version),
        description=f"Model version {version} – manually tagged as '{tag_val}'"
    )

print("\n📊  Version Summary:")
print(f"  {'Version':<10} {'R²':<10} {'Stage':<12}")
print(f"  {'-'*32}")
for ver, r2_val, stage in [(v1, r2_v1, 'archived'), (v2, r2_v2, 'staging'), (v3, r2_v3, 'production')]:
    print(f"  v{ver:<9} {r2_val:<10.4f} {stage:<12}")



print("\n🔍  Loading production model from registry…")
prod_model = mlflow.pyfunc.load_model(f"models:/{REGISTRY_NAME}@production")
sample = X_test_s[:5]
preds  = prod_model.predict(sample)
actual = y_test.values[:5]

print(f"\n  {'Sample':<8} {'Predicted':>12} {'Actual':>10} {'Error':>10}")
print(f"  {'-'*42}")
for i, (p, a) in enumerate(zip(preds, actual)):
    print(f"  {i+1:<8} ${p:>10.2f}   ${a:>8.2f}   ${abs(p-a):>8.2f}")

print("\n✅  Phase 2 Complete!")
print("   Open MLflow UI → http://localhost:5000 → Models tab to explore the registry.")