import pandas as pd
import joblib
from pathlib import Path

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = BASE_DIR / "models" / "bike_demand_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)


def predict_bike_demand(input_data):
    input_data = input_data.copy()

    input_data["Date"] = pd.to_datetime(
        input_data["Date"],
        dayfirst=True
    )

    input_data["Year"] = input_data["Date"].dt.year
    input_data["Month"] = input_data["Date"].dt.month
    input_data["Day"] = input_data["Date"].dt.day
    input_data["DayOfWeek"] = input_data["Date"].dt.dayofweek
    input_data["IsWeekend"] = (
        input_data["DayOfWeek"] >= 5
    ).astype(int)

    input_data = input_data.drop(columns=["Date"])

    prediction = model.predict(input_data)

    return prediction[0]