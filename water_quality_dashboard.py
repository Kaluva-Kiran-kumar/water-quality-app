import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
model = joblib.load('random_forest_model.pkl')  # Replace with your model path
scaler = joblib.load('scaler.pkl')  # Replace with your scaler path

# Function to predict water quality based on user input
def check_water_quality(input_data):
    # Scaling the input data
    input_data_scaled = scaler.transform([input_data])
    
    # Predicting the water quality
    prediction = model.predict(input_data_scaled)[0]
    
    # Return the result message
    if prediction == 0:
        return "🚨 ALERT: Water is unsafe! Immediate action required!"
    else:
        return "✅ Water is safe for consumption."

# Streamlit User Interface
st.title("Water Quality Prediction & Alert System")
st.write("Enter the following parameters to predict water quality:")

# User input fields for water quality parameters
pH = st.number_input("pH Level", min_value=5.0, max_value=9.0)
Hardness = st.number_input("Hardness (mg/L)", min_value=100, max_value=500, value=200)
Solids = st.number_input("Total Dissolved Solids (mg/L)", min_value=500, max_value=50000, value=10000)
Chloramines = st.number_input("Chloramines (mg/L)", min_value=0.5, max_value=10.0, value=5.0)
Sulfate = st.number_input("Sulfate (mg/L)", min_value=100, max_value=500, value=200)
Conductivity = st.number_input("Conductivity (µS/cm)", min_value=100, max_value=800, value=400)
Organic_carbon = st.number_input("Organic Carbon (mg/L)", min_value=2, max_value=30, value=10)
Trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=10, max_value=120, value=60)
Turbidity = st.number_input("Turbidity (NTU)", min_value=1, max_value=7, value=3)

# Button to check the water quality based on user input
if st.button("Check Water Quality"):
    # Collect all inputs into one list
    input_data = [pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
    
    # Get the result from the model
    alert_message = check_water_quality(input_data)
    
    # Display the result in the output area
    st.write(alert_message)
