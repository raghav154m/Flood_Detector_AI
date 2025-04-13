import streamlit as st
import requests
import joblib
import numpy as np
import random

# Load trained model and scaler
model = joblib.load("flood_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Flood Detection AI", page_icon="🌊")
st.title("🌊 Flood Detection AI")

# State to coordinate mapping (India)
state_coords = {
    "Andhra Pradesh": (16.5062, 80.6480),
    "Arunachal Pradesh": (27.0844, 93.6053),
    "Assam": (26.1433, 91.7898),
    "Bihar": (25.5941, 85.1376),
    "Chhattisgarh": (21.2514, 81.6296),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (23.0225, 72.5714),
    "Haryana": (28.4595, 77.0266),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (12.9716, 77.5946),
    "Kerala": (8.5241, 76.9366),
    "Madhya Pradesh": (23.2599, 77.4126),
    "Maharashtra": (19.0760, 72.8777),
    "Manipur": (24.8170, 93.9368),
    "Meghalaya": (25.5788, 91.8933),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (25.6751, 94.1086),
    "Odisha": (20.2961, 85.8245),
    "Punjab": (30.7333, 76.7794),
    "Rajasthan": (26.9124, 75.7873),
    "Sikkim": (27.3314, 88.6138),
    "Tamil Nadu": (13.0827, 80.2707),
    "Telangana": (17.3850, 78.4867),
    "Tripura": (23.8315, 91.2868),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.3165, 78.0322),
    "West Bengal": (22.5726, 88.3639),
    "Delhi": (28.6139, 77.2090),
    "Chandigarh (UT)": (30.7333, 76.7794),
    "Jammu & Kashmir": (34.0837, 74.7973),
    "Ladakh": (34.1526, 77.5770),
    "Puducherry": (11.9416, 79.8083),
    "Andaman & Nicobar": (11.7401, 92.6586),
    "Dadra & Nagar Haveli": (20.1809, 73.0169),
    "Daman & Diu": (20.4283, 72.8397)
}

# Dropdown for state selection
state = st.selectbox("Select your state", list(state_coords.keys()))

# Fetch real-time forecast data from Open-Meteo API
def get_weather_data(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=temperature_2m,"
        f"relative_humidity_2m,precipitation,windspeed_10m"
        f"&forecast_days=1&timezone=auto"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        idx = 0  # First forecasted hour
        return {
            "temperature": data["hourly"]["temperature_2m"][idx],
            "humidity": data["hourly"]["relative_humidity_2m"][idx],
            "wind_speed": data["hourly"]["windspeed_10m"][idx],
            "rainfall": data["hourly"]["precipitation"][idx]
        }
    else:
        return None

# Button to run prediction
if st.button("Predict Flood Risk"):
    lat, lon = state_coords[state]
    weather = get_weather_data(lat, lon)

    if weather:
        # Simulated values for missing inputs
        soil_moisture = random.uniform(35, 70)
        river_level = random.uniform(1.0, 8.0)

        # Input features for prediction
        features = np.array([[
            weather["rainfall"],
            river_level,
            soil_moisture,
            weather["humidity"],
            weather["temperature"],
            weather["wind_speed"]
        ]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        # Output result
        if prediction == 1:
            st.error(f"🚨 High Flood Risk! (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Low Flood Risk. (Confidence: {1 - prob:.2f})")

        # Display data used
        st.subheader("📊 Data Used:")
        st.json({
            "State": state,
            "Rainfall (mm/hr)": weather["rainfall"],
            "River Level (simulated, m)": round(river_level, 2),
            "Soil Moisture (simulated, %)": round(soil_moisture, 2),
            "Humidity (%)": weather["humidity"],
            "Temperature (°C)": weather["temperature"],
            "Wind Speed (km/h)": weather["wind_speed"]
        })
    else:
        st.warning("⚠️ Failed to fetch weather data. Try again later.")