from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import joblib
import requests
import numpy as np
from datetime import datetime
import calendar

app = FastAPI()

# Load models and scaler
arima_model = joblib.load('rainfall_modelseries.pkl')  # Pre-trained ARIMA model
storm_occurrence_tree = joblib.load('storm_occurrence_model.pkl')
storm_severity_tree = joblib.load('storm_severity_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define input model
class WeatherInput(BaseModel):
    location: str

# Helper function to fetch weather data from WeatherAPI
def get_weather_data(location):
    api_key = "3a0d4c4e3e304fdd92e190019243007"
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
    response = requests.get(url)
    data = response.json()

    # Extract the necessary features from the WeatherAPI response
    temp_high = data['current']['temp_c']
    temp_low = temp_high - np.random.randint(5, 10)  # Dummy value for low temp
    humidity = data['current']['humidity']
    pressure = data['current']['pressure_mb']
    wind_speed = data['current']['wind_kph']
    cloud_cover = data['current']['cloud']
    visibility = data['current']['vis_km']

    # Prepare the feature array for prediction
    features = np.array([[temp_high, temp_low, humidity, pressure, wind_speed, cloud_cover, visibility]])
    return features

# Helper function to fetch historical rainfall data from GitHub repo
def get_rainfall_data(location):
    # Placeholder for fetching the rainfall data from GitHub
    github_rainfall_url = "https://github.com/gumbogail/FarmersGuide-Datasets/blob/e00bae31ac7c48f63541a3c9c038c917e1f37f9e/rainfalldataset.csv"
    df = pd.read_csv(github_rainfall_url, on_bad_lines='skip')

    # Ensure location contains latitude and longitude
    if ',' not in location:
        return {"error": "Invalid location format. Expected 'latitude,longitude'."}

    # Split the location into latitude and longitude
    lat, lon = location.split(',')
    df_location = df[(df['Latitude'] == float(lat)) & (df['Longitude'] == float(lon))]

    # Extract historical rainfall data for the location
    rainfall_data = df_location['Rainfall'].values
    return rainfall_data

# Helper function to get the current and next 3 months
def get_next_months():
    current_month = datetime.now().month
    months = []

    # Add the current month
    month_name = calendar.month_name[current_month]
    months.append(month_name)

    # Get the next 3 months
    for i in range(1, 4):  # For the next 3 months
        next_month = (current_month + i - 1) % 12 + 1  # Month index (wrap around after December)
        month_name = calendar.month_name[next_month]
        months.append(month_name)

    return months

# API to get current and 3-month rainfall predictions using ARIMA
@app.post("/predict_rain/")
def predict_rainfall(input: WeatherInput):
    # Fetch historical rainfall data from GitHub
    rainfall_data = get_rainfall_data(input.location)

    # Check if there's enough data for ARIMA model input
    if len(rainfall_data) == 0:
        return {"error": "No historical data available for the given location."}

    # Make predictions using the pre-trained ARIMA model
    try:
        # Predict the next 4 months based on the last available data points (includes the current month)
        forecast = arima_model.forecast(steps=4)  # 4 steps, current month + next 3 months
    except Exception as e:
        return {"error": f"ARIMA model prediction failed: {str(e)}"}

    # Fetch weather data for the location (to maintain consistency with storm predictions)
    features = get_weather_data(input.location)
    features_scaled = scaler.transform(features)

    # Predict storm occurrence and severity
    storm_occurrence_pred = storm_occurrence_tree.predict(features_scaled)
    storm_severity_pred = storm_severity_tree.predict(features_scaled)

    # Get the current and next 3 months
    months = get_next_months()

    # Return predictions with months
    return {
        "rainfall_predictions_next_4_months": dict(zip(months, forecast.tolist())),  # Pair months with predictions
        "storm_occurrence": int(storm_occurrence_pred[0]),
        "storm_severity": int(storm_severity_pred[0])
    }
