import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    df = df.fillna(df.median(numeric_only=True))
    return df

df = load_data()

# Preprocessing
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# App layout
st.title("💧 Water Quality Prediction & Alert System")
st.markdown("Enter water test values to check if it's safe to drink.")

# User inputs
ph = st.number_input("pH Level", 0.0, 14.0, 7.0)
hardness = st.number_input("Hardness (mg/L)", 50.0, 500.0, 200.0)
solids = st.number_input("Total Dissolved Solids (mg/L)", 100.0, 50000.0, 10000.0)
chloramines = st.number_input("Chloramines (mg/L)", 0.0, 10.0, 5.0)
sulfate = st.number_input("Sulfate (mg/L)", 50.0, 500.0, 200.0)
conductivity = st.number_input("Conductivity (µS/cm)", 100.0, 1000.0, 400.0)
organic_carbon = st.number_input("Organic Carbon (mg/L)", 2.0, 30.0, 10.0)
trihalomethanes = st.number_input("Trihalomethanes (µg/L)", 0.0, 120.0, 60.0)
turbidity = st.number_input("Turbidity (NTU)", 1.0, 10.0, 3.0)

input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                        conductivity, organic_carbon, trihalomethanes, turbidity]])
input_data_scaled = scaler.transform(input_data)

if st.button("🔍 Check Water Quality"):
    prediction = model.predict(input_data_scaled)[0]
    if prediction == 1:
        st.success("✅ Water is safe for consumption.")
    else:
        st.error("🚨 ALERT: Water is unsafe! Immediate action needed.")
