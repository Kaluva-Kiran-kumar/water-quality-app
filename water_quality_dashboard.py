import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Assume that 'rf_clf' is your trained RandomForestClassifier and 'scaler' is your StandardScaler
# You would load these from your previously trained models

# Function to provide advice based on the input
def water_quality_advice(parameter, value):
    if parameter == 'pH':
        if value < 6.5 or value > 8.5:
            return f"<span style='color:red;'>⚠️ Unsafe pH: {value}. pH should be between 6.5 and 8.5 for safe drinking water.</span>"
        else:
            return f"<span style='color:green;'>✅ Safe pH: {value}. Water pH is within the recommended range.</span>"
    elif parameter == 'Hardness':
        if value < 150 or value > 300:
            return f"<span style='color:red;'>⚠️ Unsafe Hardness: {value}. Ideal range is 150-300 mg/L for safe drinking water.</span>"
        else:
            return f"<span style='color:green;'>✅ Safe Hardness: {value}. Water hardness is within the safe range.</span>"
    elif parameter == 'Sulfate':
        if value > 250:
            return f"<span style='color:red;'>⚠️ Unsafe Sulfate: {value}. Sulfate levels should not exceed 250 mg/L.</span>"
        else:
            return f"<span style='color:green;'>✅ Safe Sulfate: {value}. Sulfate level is within the safe limit.</span>"
    elif parameter == 'Turbidity':
        if value > 5:
            return f"<span style='color:red;'>⚠️ High Turbidity: {value}. Turbidity should be under 5 NTU for safe water.</span>"
        else:
            return f"<span style='color:green;'>✅ Safe Turbidity: {value}. Turbidity is within the safe range.</span>"
    else:
        return f"<span style='color:green;'>✅ {parameter} is within a safe range.</span>"

# Function for checking water quality
def check_water_quality(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = rf_clf.predict(input_data_scaled)[0]
    
    if prediction == 0:
        return "<span style='color:red;'>🚨 ALERT: Water is unsafe! Immediate action needed due to high contaminants!</span>"
    else:
        return "<span style='color:green;'>✅ Water is safe for consumption, but it might still need some treatment based on the parameters.</span>"

# Streamlit UI
st.title("Water Quality Prediction & Alert System")
st.write("Enter the following parameters to predict water quality:")

# Input fields for each feature (example: pH, Hardness, Sulfate, etc.)
pH = st.number_input("pH Level", min_value=5.0, max_value=9.0, value=7.0)
Hardness = st.number_input("Hardness (mg/L)", min_value=100, max_value=500, value=200)
Solids = st.number_input("Total Dissolved Solids (mg/L)", min_value=500, max_value=50000, value=10000)
Chloramines = st.number_input("Chloramines (mg/L)", min_value=0.5, max_value=10.0, value=5.0)
Sulfate = st.number_input("Sulfate (mg/L)", min_value=100, max_value=500, value=200)
Conductivity = st.number_input("Conductivity (µS/cm)", min_value=100, max_value=800, value=400)
Organic_carbon = st.number_input("Organic Carbon (mg/L)", min_value=2, max_value=30, value=10)
Trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=10, max_value=120, value=60)
Turbidity = st.number_input("Turbidity (NTU)", min_value=1, max_value=7, value=3)

# Prediction button
if st.button("Check Water Quality"):
    input_data = [pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
    
    # Check overall water quality
    alert_message = check_water_quality(input_data)
    st.markdown(alert_message, unsafe_allow_html=True)
    
    # Display individual feedback for each input
    st.write("### Parameter Advice:")
    st.markdown(water_quality_advice("pH", pH), unsafe_allow_html=True)
    st.markdown(water_quality_advice("Hardness", Hardness), unsafe_allow_html=True)
    st.markdown(water_quality_advice("Sulfate", Sulfate), unsafe_allow_html=True)
    st.markdown(water_quality_advice("Turbidity", Turbidity), unsafe_allow_html=True)
    
    # Visual representation: Display a bar chart showing input values
    feature_names = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity']
    feature_values = [pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
    
    st.write("### Water Quality Parameters Overview:")
    st.bar_chart(dict(zip(feature_names, feature_values)))
    
    # Provide a message about possible water treatment if necessary
    if Sulfate > 250 or Turbidity > 5:
        st.write("<span style='color:red;'>🚨 Water quality requires immediate treatment (high contaminants detected).</span>", unsafe_allow_html=True)
    elif pH < 6.5 or pH > 8.5:
        st.write("<span style='color:yellow;'>⚠️ pH levels are not optimal for safe consumption. Consider adjusting the pH.</span>", unsafe_allow_html=True)
