import mlflow
import pandas as pd
from arize.pandas.logger import Client
from arize.utils.types import ModelTypes, Environments, Schema

# Load best model from MLflow
ru "0ffb239e7af248f8a162dfbd05fed5d0"_here"
logged_model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(logged_model_uri)

# Load test data (make sure this matches what you trained on)
df = pd.read_parquet("data/fraud_data.parquet")
target = "Class"
X = df.drop(columns=[target, "TransactionID", "EventTime"], errors="ignore")
X = pd.get_dummies(X)
X = X.select_dtypes(include="number")
X = X.reset_index(drop=True)
y = df[target].reset_index(drop=True)

# Predict
predictions = model.predict(X)

# Build Arize log DataFrame
prediction_df = pd.DataFrame({
    "prediction_id": range(len(predictions)),
    "prediction": predictions,
    "actual": y,
    # You can log features too (up to 1000 rows in free tier)
    "scaled_amou# Your Arize credentials
space_key = "U3BhY2U6MjM3MTI6RThBTQ=="
api_key = "ak-8c93aa68-e105-4c23-b977-4ffb437fe7a5-rZPuli0UaGIrRAJ3x-OkK1sg_l5e5mFT"nt": df["scaled_amount"]
})

# Arize schema and logging
schema = Schema(
    prediction_id_column_name="prediction_id",
    prediction_label_column_name="prediction",
    actual_label_column_name="actual",
    feature_column_names=["scaled_amspace_keyent = Clieapi_keyyour_space_key", api_key="your_api_key")

response = client.log(
    model_id="fraud_model_logistic_regression",
    model_version="1.0.0",
    model_type=ModelTypes.BINARY_CLASSIFICATION,
    environment=Environments.PRODUCTION,
    dataframe=prediction_df,
    schema=schema
)

if response.status_code == 200:
    print("✅ Logged to Arize")
else:
    print(f"❌ Arize failed: {response.status_code} - {response.text}")
