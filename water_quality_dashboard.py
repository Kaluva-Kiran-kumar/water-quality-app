import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample data to simulate the trained model
# Note: Replace these with actual training data and models

# Sample pre-trained model (RandomForestClassifier) and scaler
scaler = StandardScaler()
rf_clf = RandomForestClassifier()

# Train a model (just as an example, you will use your actual model)
# Generating dummy training data for demonstration
data = {
    'pH': [7.0, 6.5, 8.0, 6.0, 7.5],
    'Hardness': [200, 150, 250, 300, 220],
    'Solids': [10000, 9000, 12000, 11000, 10500],
    'Chloramines': [5.0, 4.0, 6.0, 5.5, 4.5],
    'Sulfate': [200, 250, 180, 220, 210],
    'Conductivity': [400, 350, 450, 420, 410],
    'Organic_carbon': [10, 15, 12, 18, 14],
    'Trihalomethanes': [60, 55, 70, 65, 60],
    'Turbidity': [3, 4, 2, 5, 3],
    'Potability': [1, 0, 1, 0, 1]  # Target column: 1 = Safe, 0 = Unsafe
}

df = pd.DataFrame(data)

# Splitting data into X and y
X = df.drop('Potability', axis=1)
y = df['Potability']

# Standardizing the data
X_scaled = scaler.fit_transform(X)

# Train the RandomForest model (just for demonstration)
rf_clf.fit(X_scaled, y)

# Save the trained model and scaler (you can also load your pre-trained models here)
with open('rf_clf_model.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Load pre-trained model and scaler for prediction
with open('rf_clf_model.pkl', 'rb') as f:
    rf_clf = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to check water quality based on input data
def check_water_quality(input_data):
    """
    Function to predict water quality and send an alert if unsafe.
    Args:
    input_data: list of feature values (e.g., pH, Hardness, Sulfate, etc.)
    Returns:
    str: Alert message (safe or unsafe)
    """
    input_data_scaled = scaler.transform([input_data])  # Standardize the input data
    prediction = rf_clf.predict(input_data_scaled)[0]  # Predict using the RandomForest model
    if prediction == 0:
        return "🚨 ALERT: Water is unsafe! Immediate action needed!"
    else:
        return "✅ Water is safe for consumption."

# Streamlit UI
st.title("Water Quality Prediction & Alert System")
st.write("Enter the following parameters to predict water quality:")

# Input fields for each feature
pH = st.number_input("pH Level", min_value=5.0, max_value=9.0, value=7.0)
Hardness = st.number_input("Hardness (mg/L)", min_value=100, max_value=500,_
