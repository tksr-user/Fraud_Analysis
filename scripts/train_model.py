import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from arize.pandas.logger import Client
from arize.utils.types import Schema, Environments, ModelTypes
import uuid
import os

# Load data
df = pd.read_parquet("data/fraud_data.parquet")
# Drop unwanted columns
df = df.drop(columns=["transaction_id", "event_id", "prediction_id","uuid","id"], errors="ignore")


# Features and label
X = df.drop("Class", axis=1)
y = df["Class"]
# Keep only numeric columns (safety check)
X = X.select_dtypes(include=["number"])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Setup Arize
ARIZE_SPACE_KEY = "U3BhY2U6MjM3MTI6RThBTQ=="
ARIZE_API_KEY = "ak-8c93aa68-e105-4c23-b977-4ffb437fe7a5-rZPuli0UaGIrRAJ3x-OkK1sg_l5e5mF"

arize_client = Client(space_key=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

# Schema for Arize
schema = Schema(
    prediction_id_column_name="prediction_id",
    prediction_label_column_name="prediction",
    actual_label_column_name="actual",
    feature_column_names=X.columns.tolist()
)

# Models to train
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# MLflow logging
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Or local path
mlflow.set_experiment("Fraud_Detection_Comparison")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"📦 Training: {name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, name)

        # Save predictions to send to Arize
        pred_df = X_test.copy()
        pred_df.reset_index(drop=True, inplace=True)
        pred_df["prediction"] = y_pred
        pred_df["actual"] = y_test.values
        pred_df["prediction_id"] = [str(uuid.uuid4()) for _ in range(len(pred_df))]

        response = arize_client.log(
            dataframe=pred_df,
            schema=schema,
            model_id=f"fraud_model_{name}",
            model_version="1.0.0",
            model_type=ModelTypes.BINARY_CLASSIFICATION,
            environment=Environments.PRODUCTION,
        )

        if response.status_code == 200:
            print(f"✅ Arize logged for {name}")
        else:
            print(f"❌ Arize failed for {name}: {response.status_code}, {response.text}")

print("🏁 All models trained and logged.")