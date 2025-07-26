
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

with mlflow.start_run(run_name="best_randomforest_final"):
    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": best_params["n_estimators"],
        "max_depth": best_params["max_depth"]
    })
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.sklearn.log_model(model, "model")
