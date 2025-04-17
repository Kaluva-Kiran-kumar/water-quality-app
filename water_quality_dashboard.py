import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Dummy model and scaler loading - replace these with your actual model and scaler
# model = joblib.load('random_forest_model.pkl')  # Load your trained model
# scaler = joblib.load('scaler.pkl')  # Load your scaler

# Example pre-trained model and scaler (for illustration purposes)
# Replace this with actual loading of your model
model = xgb.XGBClassifier()  # This is just a placeholder
scaler = StandardScaler()  # This is just a placeholder

def check_water_quality(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)[0]
    if prediction == 0:
        return "🚨 ALERT: Water is unsafe! Immediate action needed!"
    else:
        return "✅ Water is safe for consumption."

# Streamlit UI
st.title("Water Quality Prediction & Alert System")
st.write("Enter the following parameters to predict water quality:")

# Create two columns: one for inputs and one for output
col1, col2 = st.columns([1, 2])  # First column for inputs, second for outputs

# Input section
with col1:
    pH = st.number_input("pH Level", min_value=5.0, max_value=9.0, value=7.0)
    Hardness = st.number_input("Hardness (mg/L)", min_value=100, max_value=500, value=200)
    Solids = st.number_input("Total Dissolved Solids (mg/L)", min_value=500, max_value=50000, value=10000)
    Chloramines = st.number_input("Chloramines (mg/L)", min_value=0.5, max_value=10.0, value=5.0)
    Sulfate = st.number_input("Sulfate (mg/L)", min_value=100, max_value=500, value=200)
    Conductivity = st.number_input("Conductivity (µS/cm)", min_value=100, max_value=800, value=400)
    Organic_carbon = st.number_input("Organic Carbon (mg/L)", min_value=2, max_value=30, value=10)
    Trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=10, max_value=120, value=60)
    Turbidity = st.number_input("Turbidity (NTU)", min_value=1, max_value=7, value=3)

# Output section
with col2:
    # Prediction button
    if st.button("Check Water Quality"):
        input_data = [pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
        alert_message = check_water_quality(input_data)
        st.write(alert_message)

    # Additional information for better understanding of the parameters
    st.write("""
    **pH Level**: pH is a measure of the water's acidity or alkalinity. Safe drinking water typically has a pH between 6.5 and 8.5.
    **Hardness**: The amount of dissolved calcium and magnesium in water. Water hardness can cause scale buildup and affect appliances.
    **Total Dissolved Solids (TDS)**: TDS is a measure of all dissolved substances. A high level of TDS can indicate contamination.
    **Chloramines**: Used as a disinfectant, chloramines can affect water quality and cause irritation to skin or eyes.
    **Sulfate**: High levels of sulfate can cause a bitter taste and, in large amounts, digestive issues.
    **Conductivity**: Measures the ability of water to conduct electricity, which correlates with ion concentration in water.
    **Organic Carbon**: High levels may indicate pollution and affect the taste or quality of the water.
    **Trihalomethanes**: These byproducts of chlorination can be harmful and potentially carcinogenic.
    **Turbidity**: Measures the cloudiness of the water, indicating possible contamination with particles or microorganisms.
    """)

