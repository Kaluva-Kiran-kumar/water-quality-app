import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("water_potability.csv")
df = df.fillna(df.median())

# Split and scale data
X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Streamlit UI
st.title("💧 Smart Water Quality Prediction & Alert System")
st.write("Enter the water quality parameters to check potability:")

# Inputs
pH = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
Hardness = st.number_input("Hardness", min_value=0.0, max_value=500.0, value=200.0)
Solids = st.number_input("Solids", min_value=0.0, max_value=50000.0, value=10000.0)
Chloramines = st.number_input("Chloramines", min_value=0.0, max_value=10.0, value=5.0)
Sulfate = st.number_input("Sulfate", min_value=0.0, max_value=500.0, value=250.0)
Conductivity = st.number_input("Conductivity", min_value=0.0, max_value=1000.0, value=400.0)
Organic_carbon = st.number_input("Organic Carbon", min_value=0.0, max_value=30.0, value=10.0)
Trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, max_value=120.0, value=60.0)
Turbidity = st.number_input("Turbidity", min_value=0.0, max_value=10.0, value=3.0)

# Prediction logic
if st.button("🚰 Check Water Quality"):
    input_data = np.array([[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity,
                            Organic_carbon, Trihalomethanes, Turbidity]])
    input_scaled = scaler.transform(input_data)
    prediction = rf_clf.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Water is **safe** for drinking.")
    else:
        st.error("🚨 Water is **unsafe** for drinking!")
