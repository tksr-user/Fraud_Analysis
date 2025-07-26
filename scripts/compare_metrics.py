import mlflow
import pandas as pd
from arize.pandas.logger import Client
from arize.utils.types import ModelTypes, Environments, Schema

# Step 1: Fetch Metrics from MLflow
def get_mlflow_metrics(run_id: str):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id)
    return data.data.metrics

# Step 2: Simulated Arize metrics (normally you’d fetch via API or dashboard export)
def get_arize_metrics():
    # Replace these with what you get from Arize dashboard export (if needed)
    return {
        "accuracy": 0.93,
        "precision": 0.91,
        "recall": 0.89,
        "num_rows": 10000  # ✅ add this to match MLflow
    }
    

# Step 3: Compare both
def compare_metrics(mlflow_metrics, arize_metrics):
    print("\n🔍 Comparing Metrics:")
    for key in mlflow_metrics:
        mlflow_val = mlflow_metrics[key]
        arize_val = arize_metrics.get(key)
        if arize_val is not None:
            print(f"{key}: MLflow = {mlflow_val:.4f}, Arize = {arize_val:.4f}")
        else:
            print(f"{key}: MLflow = {mlflow_val:.4f}, Arize = ❌ Not Found")

# Run the comparison
run_id = "530ee95e267143a5afd361c95d93d367"  # your real run_id
mlflow_metrics = get_mlflow_metrics(run_id)
arize_metrics = get_arize_metrics()
compare_metrics(mlflow_metrics, arize_metrics)
