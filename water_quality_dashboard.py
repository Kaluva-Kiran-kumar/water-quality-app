import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle  # For saving and loading models

# Load the dataset (Make sure the file is in the right directory or uploaded)
df = pd.read_csv('water_potability.csv')

# Data Preprocessing
df = df.fillna(df.median())  # Fill missing values

# Split the dataset into features and target
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Save the trained model and scaler using pickle (you'll need to load these later in the Streamlit app)
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define function for checking water quality
def check_water_quality(input_data):
    """
    Function to predict water quality and send an alert if unsafe.
    
    Args:
    input_data: list of feature values (e.g., pH, Hardness, Sulfate, etc.)
    
    Returns:
    str: Alert message (safe or unsafe)
    """
    # Load the trained model and scaler
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Transform the input data to match the format of the training data
    input_data_scaled = scaler.transform([input_data])
    
    # Predict using the trained model
    prediction = model.predict(input_data_scaled)[0]
    
    if prediction == 0:
        return "🚨 ALERT: Water is unsafe! Immediate action needed!"
    else:
        return "✅ Water is safe for consumption."

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
    alert_message = check_water_quality(input_data)
    st.write(alert_message)
