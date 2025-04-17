import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("water_potability.csv")

# Preprocess data
df = df.fillna(df.median())

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models (you can choose any, here RandomForest is used for prediction)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Define prediction function
def check_water_quality(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = rf_clf.predict(input_data_scaled)[0]
    if prediction == 0:
        return "🚨 ALERT: Water is unsafe! Immediate action needed!"
    else:
        return "✅ Water is safe for consumption."

# ---------------- STREAMLIT UI ----------------

st.title("💧 Smart Water Quality Prediction & Alert System")
st.write("Enter the following parameters to predict whether water is safe to drink:")

# Info box for users
st.info("💡 Tip: Use a water testing kit, digital meter, or lab report to measure values like pH, Turbidity, etc., before entering them.")

# Input fields for all features
pH = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
Hardness = st.number_input("Hardness", min_value=50.0, max_value=500.0, value=150.0)
Solids = st.number_input("Solids", min_value=100.0, max_value=50000.0, value=10000.0)
Chloramines = st.number_input("Chloramines", min_value=0.0, max_value=10.0, value=5.0)
Sulfate = st.number_input("Sulfate", min_value=50.0, max_value=500.0, value=200.0)
Conductivity = st.number_input("Conductivity", min_value=100.0, max_value=800.0, value=400.0)
Organic_carbon = st.number_input("Organic Carbon", min_value=2.0, max_value=30.0, value=10.0)
Trihalomethanes = st.number_input("Trihalomethanes", min_value=10.0, max_value=120.0, value=60.0)
Turbidity = st.number_input("Turbidity", min_value=1.0, max_value=10.0, value=3.0)

# Button to check water quality
if st.button("Check Water Quality"):
    input_data = [
        pH, Hardness, Solids, Chloramines, Sulfate,
        Conductivity, Organic_carbon, Trihalomethanes, Turbidity
    ]
    result = check_water_quality(input_data)
    st.success(result)
