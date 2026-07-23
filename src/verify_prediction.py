import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, DATA_FILE_PATTERN
from src.data_loader import load_local_customer_data, get_customer_record
from src.model_loader import load_mlflow_model
from src.predictor import predict_churn, save_prediction_json


def main():
    customer_id = os.getenv("CUSTOMER_ID", "TC-000001")
    df = load_local_customer_data(DATA_DIR, DATA_FILE_PATTERN)
    record = get_customer_record(df, customer_id)
    if record is None or record.empty:
        raise RuntimeError(f"Could not load the selected customer record from {DATA_DIR} for customer {customer_id}.")

    model = load_mlflow_model("7e793e3e2f1d4b798a9b90151cc9c8fb", "XGBClassifier With SMOTE")
    prediction = predict_churn(record.iloc[0].to_dict(), model=model)
    output_path = save_prediction_json(prediction, customer_id)
    print(prediction)
    print(output_path)


if __name__ == "__main__":
    main()
