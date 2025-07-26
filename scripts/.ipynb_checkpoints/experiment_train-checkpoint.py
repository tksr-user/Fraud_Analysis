import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from arize.pandas.logger import Client
from arize.utils.types import ModelTypes, Environments, Schema


# Load your dataset
df = pd.read_parquet("data/fraud_data.parquet")
# ✅ Features and label
target = "Class"
X_full = df.drop(columns=[target])
y = df[target]

# ✅ Define feature subsets
# Drop columns with too many unique values (e.g., more than 100)
for col in X_full.select_dtypes(include="object").columns:
    if X_full[col].nunique() > 100:
        print(f"Dropping column: {col} (unique: {X_full[col].nunique()})")
        X_full = X_full.drop(columns=col)

X_full = pd.get_dummies(X_full)  # One-hot encode all object-type columns
all_features = X_full.columns.tolist()
half_features = all_features[:len(all_features)//2]
top_features = all_features[10:20]  # Example: features ranked important by domain or feature importance

feature_sets = {
    "all_features": all_features,
    "half_features": half_features,
    "top_10_features": top_features
}

# ✅ Define models and hyperparameters
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=500),
        "params": {"C": [0.1, 1, 10]}
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [50, 100], "max_depth": [3, 5]}
    },
    "SVC": {
        "model": SVC(),
        "params": {"C": [0.1, 1], "kernel": ["linear", "rbf"]}
    }
}


for exp_id in range(1, 11):
    experiment_name = f"Fraud_Detection_Exp_{exp_id}"
    mlflow.set_experiment(experiment_name)

    for feature_set_name, selected_features in feature_sets.items():
        X = X_full[selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=exp_id)

        for model_name, config in models.items():
            model = config["model"]
            param_grid = config["params"]

            grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)

            with mlflow.start_run(run_name=f"{model_name}_{feature_set_name}"):
                mlflow.log_param("model", model_name)
                mlflow.log_param("feature_set", feature_set_name)
                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.sklearn.log_model(best_model, artifact_path="model")

                print(f"✅ Run logged for {model_name} with {feature_set_name} in {experiment_name}")
                
# Start a run
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("accuracy", 0.95)

#  Start MLflow run and log
with mlflow.start_run(run_name="data_exploration_run") as run:
    # Log a sample parameter
    mlflow.log_param("data_source", "fraud_data.parquet")

    # Log a sample metric
    mlflow.log_metric("num_rows", len(df))

    # Save and log artifact
    sample_path = "sample_data.csv"
    df.head(100).to_csv(sample_path, index=False)
    mlflow.log_artifact(sample_path)

    print("✅ MLflow run completed. Run ID:", run.info.run_id)

#  Fetch Metrics from MLflow
def get_mlflow_metrics(run_id: str):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id)
    return data.data.metrics

# Simulated Arize metrics (normally you’d fetch via API or dashboard export)
def get_arize_metrics():
    # Replace these with what you get from Arize dashboard export (if needed)
    return {
        "accuracy": 0.93,
        "precision": 0.91,
        "recall": 0.89,
        "num_rows": 10000  # ✅ add this to match MLflow
    }
    

#  Compare both
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

