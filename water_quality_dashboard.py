import streamlit as st
import joblib
import numpy as np

# Load pre-trained models and scaler
rf_clf = joblib.load('random_forest_model.pkl')  # Load RandomForest model
scaler = joblib.load('scaler.pkl')               # Load the scaler for preprocessing

# Function for checking water quality
def check_water_quality(input_data):
    input_data_scaled = scaler.transform([input_data])  # Standardize the input data
    prediction = rf_clf.predict(input_data_scaled)[0]  # Predict using the model
    
    if prediction == 0:
        return "🚨 ALERT: Water is unsafe! Immediate action needed!"
    else:
        return "✅ Water is safe for consumption."

# Streamlit UI setup
st.title("Water Quality Prediction & Alert System")
st.write("Enter the following parameters to predict water quality:")

# Input fields for each feature
pH = st.number_input("pH Level", min_value=5.0, max_value=9.0, value=7.0)
Hardness = st.number_input("Hardness (mg/L)", min_value=100, max_value=500, value=200)
Solids = st.number_input("Total Dissolved Solids (mg/L)", min_value=500, max_value=50000, value=10000)
Chloramines = st.number_input("Chloramines (mg/L)", min_value=0.5, max_value=10.0, value=5.0)
Sulfate = st.number_input("Sulfate (mg/L)", min_value=100, max_value=500, value=200)
Conductivity = st.number_input("Conductivity (µS/cm)", min_value=100, max_value=800, value=400)
Organic_carbon = st.number_input("Organic Carbon (mg/L)", min_value=2, max_value=30, value=10)
Trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=10, max_value=120, value=60)
Turbidity = st.number_input("Turbidity (NTU)", min_value=1, max_value=7, value=3)

# Button for prediction
if st.button("Check Water Quality"):
    input_data = [pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
    alert_message = check_water_quality(input_data)
    st.write(alert_message)
