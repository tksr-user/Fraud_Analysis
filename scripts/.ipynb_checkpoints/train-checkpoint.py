
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import uuid
from datetime import datetime
import os
import io
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from arize.pandas.logger import Client
from arize.utils.types import ModelTypes, Environments, Schema
import optuna
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
run_id = "b70816424b1d414d9cd71633db67bb26"  # your real run_id
mlflow_metrics = get_mlflow_metrics(run_id)
arize_metrics = get_arize_metrics()
compare_metrics(mlflow_metrics, arize_metrics)

from sklearn.model_selection import train_test_split

# Your dataset
df = df.select_dtypes(include=['number'])
target = "Class"
X_full = df.drop(columns=[target])
y_full = df[target]

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "XGBoost": XGBClassifier
}


def objective(trial):
    model_name = trial.suggest_categorical("model", list(models.keys()))
    
    if model_name == "LogisticRegression":
        C = trial.suggest_loguniform("C", 0.01, 10)
        model = LogisticRegression(C=C, max_iter=1000)
        
    elif model_name == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        
    else:  # XGBoost
        eta = trial.suggest_float("eta", 0.01, 0.3)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        model = XGBClassifier(eta=eta, max_depth=max_depth, use_label_encoder=False, eval_metric='logloss')
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = accuracy_score(y_test, preds)

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_params(trial.params)
        mlflow.log_metric("accuracy", score)
        mlflow.sklearn.log_model(model, "model")
    
    return score
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print("Best model:", study.best_params["model"])
print("Best accuracy:", study.best_value)

best_params = study.best_params

model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    random_state=42
)

model.fit(X_train, y_train)


preds = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, preds))
print("📋 Classification Report:\n", classification_report(y_test, preds))


with mlflow.start_run(run_name="best_randomforest_final"):
    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": best_params["n_estimators"],
        "max_depth": best_params["max_depth"]
    })
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.sklearn.log_model(model, "model")
