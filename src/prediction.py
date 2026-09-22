import pandas as pd
import joblib
import requests
import io


MODEL_URL = "https://huggingface.co/PranavDeployer221/urban-bike-demand-model/resolve/main/bike_demand_model.pkl"


# Download and load model from Hugging Face
response = requests.get(MODEL_URL)
response.raise_for_status()

model = joblib.load(io.BytesIO(response.content))


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