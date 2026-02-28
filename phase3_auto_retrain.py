
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import os
import json
import hashlib
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb


TRACKING_URI  = "file:///C:/Users/Akhiliny Vijeyagumar/OneDrive/Desktop/mlops-pipeline/mlruns"
EXPERIMENT    = "california_house_price_prediction_retrain"
REGISTRY_NAME = "house_price_best_model"
DATA_DIR      = r"C:\Users\Akhiliny Vijeyagumar\OneDrive\Desktop\mlops-pipeline\data"
STATE_FILE    = os.path.join(DATA_DIR, "data_state.json")


mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
os.makedirs(DATA_DIR, exist_ok=True)



def get_file_hash(filepath: str) -> str:
    """Return MD5 hash of a file – used to detect data changes."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_hash": "", "last_retrain": None, "retrain_count": 0}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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



def simulate_new_data(version: int = 1):
    """
    In a real pipeline, your data would arrive as a new CSV file.
    Here we simulate two 'versions' of the dataset so the script
    detects a change and triggers retraining.
    """
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["Price"] = housing.target

    if version == 1:
       
        df = df.sample(frac=0.80, random_state=1)
    else:
        
        df = df.sample(frac=1.0, random_state=2)

    data_path = os.path.join(DATA_DIR, "housing.csv")
    df.to_csv(data_path, index=False)
    print(f"  📂 Saved {len(df):,} rows → {data_path}")
    return data_path



def retrain(data_path: str, retrain_number: int) -> dict:
    print(f"\n🔄  Retraining (run #{retrain_number})…")
    df = pd.read_csv(data_path)
    df = build_features(df)

    X = df.drop("Price", axis=1)
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    params = {
    "n_estimators": 300,      
    "learning_rate": 0.02,     
    "max_depth": 10,           
    "num_leaves": 127,         
}

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"AutoRetrain_{retrain_number}_{datetime.now():%Y%m%d_%H%M}") as run:
        model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        mlflow.log_params({"model_type": "LightGBM", "retrain_run": retrain_number,
                           "data_rows": len(df), **params})
        mlflow.log_metrics({"r2_score": r2, "rmse": rmse, "mae": mae})
        mlflow.set_tag("trigger", "data_change_detected")
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id

    print(f"  📈 New model  → R²={r2:.4f}  RMSE={rmse:.4f}")
    return {"run_id": run_id, "r2": r2, "rmse": rmse, "mae": mae}


def get_production_r2() -> float:
    """Retrieve the R² of the current production model from the registry."""
    try:
        alias_info = client.get_model_version_by_alias(REGISTRY_NAME, "production")
        run_id = alias_info.run_id
        run    = client.get_run(run_id)
        return run.data.metrics.get("r2_score", 0.0)
    except Exception:
        return 0.0 


def promote_if_better(new_result: dict):
    prod_r2 = get_production_r2()
    print(f"\n  🔍 Production R²={prod_r2:.4f}  |  New model R²={new_result['r2']:.4f}")

    if new_result["r2"] > prod_r2:
        model_uri = f"runs:/{new_result['run_id']}/model"
        reg       = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
        client.set_registered_model_alias(REGISTRY_NAME, "production", str(reg.version))
        client.update_model_version(
            name=REGISTRY_NAME, version=str(reg.version),
            description=f"Auto-promoted on {datetime.now():%Y-%m-%d %H:%M} "
                        f"(R²={new_result['r2']:.4f} > prev {prod_r2:.4f})"
        )
        print(f"  🚀 Promoted to Production  (v{reg.version})")
    else:
        print(f"  ⏭️  Skipped – new model did NOT beat production.")



def run_pipeline():
    state = load_state()
    print("=" * 55)
    print("  🤖  Automated Retraining Pipeline")
    print("=" * 55)

    
    data_path = simulate_new_data(version=1)

    current_hash = get_file_hash(data_path)
    print(f"\n  📊 Data hash  (now):  {current_hash[:12]}…")
    print(f"  📊 Data hash (last):  {state['last_hash'][:12] or 'none'}…")

    if current_hash == state["last_hash"]:
        print("\n  ✅ Data unchanged – no retraining needed.")
        return

    print("\n  ⚡ Data change detected – triggering retraining…")
    state["retrain_count"] += 1
    result = retrain(data_path, state["retrain_count"])
    promote_if_better(result)

    state["last_hash"]    = current_hash
    state["last_retrain"] = datetime.now().isoformat()
    save_state(state)
    print(f"\n  💾 State saved → {STATE_FILE}")
    print(f"\n✅  Phase 3 Complete!  Total retrains: {state['retrain_count']}")


if __name__ == "__main__":
    run_pipeline()