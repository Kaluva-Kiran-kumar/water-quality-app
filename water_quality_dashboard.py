import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Water Quality Predictor", layout="centered")

st.title("💧 Water Quality Prediction & Alert System")
st.markdown("Enter water test values to check if it's safe to drink. We'll help you understand each parameter too!")

# Inputs with explanation tooltips
pH = st.number_input("pH Level", 5.0, 9.0, 7.0, help="Ideal pH for drinking water is between 6.5 and 8.5.")
Hardness = st.number_input("Hardness (mg/L)", 100, 500, 200, help="Hardness is caused by calcium/magnesium. High values can affect taste.")
Solids = st.number_input("Total Dissolved Solids (mg/L)", 500, 50000, 10000, help="High TDS may include harmful metals or salts.")
Chloramines = st.number_input("Chloramines (mg/L)", 0.5, 10.0, 5.0, help="Used to disinfect water. Should be below 4 mg/L ideally.")
Sulfate = st.number_input("Sulfate (mg/L)", 100, 500, 200, help="Sulfates above 250 mg/L may affect taste and cause issues.")
Conductivity = st.number_input("Conductivity (µS/cm)", 100, 800, 400, help="High conductivity indicates high ion concentration.")
Organic_carbon = st.number_input("Organic Carbon (mg/L)", 2, 30, 10, help="Measures organic contaminants. Higher values can reduce quality.")
Trihalomethanes = st.number_input("Trihalomethanes (µg/L)", 10, 120, 60, help="By-products of disinfection. WHO limit: ~100 µg/L.")
Turbidity = st.number_input("Turbidity (NTU)", 1, 7, 3, help="Clarity of water. WHO suggests <5 NTU.")

# Button to predict
if st.button("🔍 Predict Water Quality"):
    input_data = [pH, Hardness, Solids, Chloramines, Sulfate,
                  Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Water is **safe** for consumption.")
        st.markdown("**Great!** Your water meets the general quality standards.")
    else:
        st.error("🚨 ALERT: Water is **unsafe** for consumption!")
        st.markdown("Please consider treating or testing the water further before use.")
