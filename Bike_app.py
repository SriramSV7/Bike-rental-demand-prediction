import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and tools
model = joblib.load('bike_rental_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

st.title("🚲 Bike Rental Demand Predictor")
st.write("Enter the weather and time details to predict hourly demand.")

# User Inputs
temp = st.slider("Normalized Temperature", 0.0, 1.0, 0.5)
hum = st.slider("Humidity", 0.0, 1.0, 0.5)
wind = st.slider("Windspeed", 0.0, 1.0, 0.1)
hr = st.selectbox("Hour of Day", list(range(24)))

# Note: In a real app, you'd add all inputs (season, holiday, etc.)
# For this example, we create a template row of zeros
input_data = pd.DataFrame(0, index=[0], columns=features)

# Fill in the user-provided values
input_data['temp'] = temp
input_data['hum'] = hum
input_data['windspeed'] = wind
# Handle the dummy variable for hour (e.g., hr_1, hr_2...)
if f'hr_{hr}' in features:
    input_data[f'hr_{hr}'] = 1

# Predict
if st.button("Predict Demand"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Bike Demand: {int(prediction[0])} bikes")