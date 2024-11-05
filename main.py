from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.arima.model import ARIMA
import logging
from fastapi.middleware.gzip import GZipMiddleware



app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Configure logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained models
storm_occurrence_model = joblib.load("storm_occurrence_model.pkl")
storm_severity_model = joblib.load("storm_severity_model.pkl")
rainfall_model = joblib.load("rainfall_modelseries.pkl")

# API Key and endpoint for WeatherAPI
WEATHER_API_KEY = "3a0d4c4e3e304fdd92e190019243007"
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

# Define the input data structure
class Location(BaseModel):
    latitude: float
    longitude: float

# Helper function to get weather data
def get_weather_data(lat, lon):
    try:
        logging.info(f"Fetching weather data for lat: {lat}, lon: {lon}")
        url = f"{WEATHER_API_URL}?key={WEATHER_API_KEY}&q={lat},{lon}"
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()["current"]
        
        # Map the fetched weather data to the features used during training
        return {
            'Temp_High (°C)': weather_data["temp_c"],
            'Temp_Low (°C)': weather_data["temp_c"],  # Assuming same for simplicity
            'Humidity (%)': weather_data["humidity"],
            'Atmospheric_Pressure (hPa)': weather_data["pressure_mb"],
            'Wind_Speed (km/h)': weather_data["wind_kph"],
            'Cloud_Cover (%)': weather_data["cloud"],
            'Visibility (km)': weather_data["vis_km"]
        }
    except Exception as e:
        logging.error(f"Error fetching weather data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error fetching weather data: {str(e)}")

# Helper function to get rainfall data from GitHub
def get_rainfall_data():
    try:
        # Assume the file is a CSV on GitHub
        rainfall_data = pd.read_csv("https://raw.githubusercontent.com/gumbogail/FarmersGuide-Datasets/e00bae31ac7c48f63541a3c9c038c917e1f37f9e/rainfalldataset.csv")
        # Keep only the relevant columns, assuming 'rainfall' is the correct column name
        return rainfall_data[['Rainfall']]
    except Exception as e:
        logging.error(f"Error fetching rainfall data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error fetching rainfall data: {str(e)}")

# Route for predicting rainfall and storm occurrence/severity
@app.get("/predict/")
def predict_storm_and_rainfall(location: Location):
    # Get weather features for prediction
    weather_features = get_weather_data(location.latitude, location.longitude)

    # Prepare features for storm occurrence model
    features = np.array([[
        weather_features["Temp_High (°C)"],
        weather_features["Temp_Low (°C)"],
        weather_features["Humidity (%)"],
        weather_features["Atmospheric_Pressure (hPa)"],
        weather_features["Wind_Speed (km/h)"],
        weather_features["Cloud_Cover (%)"],
        weather_features["Visibility (km)"]
    ]])

    # Predict storm occurrence and severity
    storm_occurrence = storm_occurrence_model.predict(features)[0]
    storm_severity = storm_severity_model.predict(features)[0] if storm_occurrence == 1 else None

    # Load rainfall data for ARIMA model
    rainfall_data = get_rainfall_data()
    rainfall_values = rainfall_data['Rainfall'].values

    # Fit ARIMA model to predict the next 3 months
    arima_model = ARIMA(rainfall_values, order=(5, 1, 0))  # ARIMA parameters are placeholders
    arima_fit = arima_model.fit()
    rainfall_forecast = arima_fit.forecast(steps=3)

    return {
        "storm_occurrence": int(storm_occurrence),
        "storm_severity": float(storm_severity) if storm_occurrence == 1 else "No Storm",
        "rainfall_forecast": list(rainfall_forecast)
    }

@app.get("/predict/daily/")
def predict_daily_storm(location: Location):
    # Get weather features for prediction
    weather_features = get_weather_data(location.latitude, location.longitude)

    # Prepare features for storm occurrence model
    features = np.array([[
        weather_features["Temp_High (°C)"],
        weather_features["Temp_Low (°C)"],
        weather_features["Humidity (%)"],
        weather_features["Atmospheric_Pressure (hPa)"],
        weather_features["Wind_Speed (km/h)"],
        weather_features["Cloud_Cover (%)"],
        weather_features["Visibility (km)"]
    ]])

    # Predict daily storm occurrence and severity
    storm_occurrence = storm_occurrence_model.predict(features)[0]
    storm_severity = storm_severity_model.predict(features)[0] if storm_occurrence == 1 else None

     # Load rainfall data for ARIMA model
    rainfall_data = get_rainfall_data()
    rainfall_values = rainfall_data['Rainfall'].values

    # Fit ARIMA model to predict the next day's rainfall
    arima_model = ARIMA(rainfall_values, order=(5, 1, 0))  # ARIMA parameters are placeholders
    arima_fit = arima_model.fit()
    rainfall_forecast = arima_fit.forecast(steps=1)
    

    return {
        "daily_storm_occurrence": int(storm_occurrence),
        "daily_storm_severity": float(storm_severity) if storm_occurrence == 1 else "No Storm",
        "rainfall_forecast": list(rainfall_forecast)
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)