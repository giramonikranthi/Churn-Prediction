from pathlib import Path
import os

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data" / "Input_data"
DATA_FILE_PATTERN = ["*.csv", "*.xlsx", "*.json"]
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_RUN_ID = "7e793e3e2f1d4b798a9b90151cc9c8fb"
MODEL_NAME = "XGBClassifier With SMOTE"
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "runs:/7e793e3e2f1d4b798a9b90151cc9c8fb/model")
OUTPUT_DIR = APP_DIR / "data" / "Output_data"
LOG_DIR = APP_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
