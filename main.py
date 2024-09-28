from fastapi import FastAPI
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
class CoordinatesInput(BaseModel):
    latitude: float
    longitude: float

# Helper function to fetch weather data from WeatherAPI
def get_weather_data(latitude, longitude):
    api_key = "3a0d4c4e3e304fdd92e190019243007"
    location = f"{latitude},{longitude}"
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

# Helper function to fetch historical rainfall data
def get_rainfall_data(latitude, longitude):
    github_rainfall_url = "https://github.com/gumbogail/FarmersGuide-Datasets/blob/e00bae31ac7c48f63541a3c9c038c917e1f37f9e/rainfalldataset.csv"
    df = pd.read_csv(github_rainfall_url, on_bad_lines='skip')

    df_location = df[(df['Latitude'] == latitude) & (df['Longitude'] == longitude)]
    rainfall_data = df_location['Rainfall'].values
    return rainfall_data

# API to predict current and next 3-month rainfall using ARIMA
@app.post("/predict_rain/")
def predict_rainfall(input: CoordinatesInput):
    latitude = input.latitude
    longitude = input.longitude

    # Fetch historical rainfall data
    rainfall_data = get_rainfall_data(latitude, longitude)
    if len(rainfall_data) == 0:
        return {"error": "No historical data available for the given location."}

    # Predict the next 4 months using the ARIMA model
    forecast = arima_model.forecast(steps=4)

    # Fetch weather data and predict storm occurrence and severity
    features = get_weather_data(latitude, longitude)
    features_scaled = scaler.transform(features)
    storm_occurrence_pred = storm_occurrence_tree.predict(features_scaled)
    storm_severity_pred = storm_severity_tree.predict(features_scaled)

    # Get the current and next 3 months
    months = get_next_months()

    return {
        "rainfall_predictions_next_4_months": dict(zip(months, forecast.tolist())),
        "storm_occurrence": int(storm_occurrence_pred[0]),
        "storm_severity": int(storm_severity_pred[0])
    }

# API to predict today's storm and rainfall
@app.post("/predict_today/")
def predict_today(input: CoordinatesInput):
    latitude = input.latitude
    longitude = input.longitude

    # Fetch weather data and predict storm occurrence and severity
    features = get_weather_data(latitude, longitude)
    features_scaled = scaler.transform(features)
    storm_occurrence_pred = storm_occurrence_tree.predict(features_scaled)
    storm_severity_pred = storm_severity_tree.predict(features_scaled)

    # Predict rainfall for today
    forecast = arima_model.forecast(steps=1)

    return {
        "rainfall_today": forecast[0],
        "storm_occurrence_today": int(storm_occurrence_pred[0]),
        "storm_severity_today": int(storm_severity_pred[0])
    }

# Helper function to get the current and next 3 months
def get_next_months():
    current_month = datetime.now().month
    months = [calendar.month_name[(current_month + i - 1) % 12 + 1] for i in range(4)]
    return months
