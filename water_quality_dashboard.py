import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Title and Introduction
st.set_page_config(page_title="💧 Water Quality Prediction", page_icon="💧", layout="wide")
st.title("💧 Water Quality Prediction & Alert System")
st.markdown("""
Welcome to the **Water Quality Prediction & Alert System**! 🧪
Enter the water test values below to check if the water is safe to drink.
""")
st.markdown("----")

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

# Section with input fields
st.sidebar.header("🌊 Enter Water Quality Parameters")
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
hardness = st.sidebar.slider("Hardness (mg/L)", 50.0, 500.0, 200.0)
solids = st.sidebar.slider("Total Dissolved Solids (mg/L)", 100.0, 50000.0, 10000.0)
chloramines = st.sidebar.slider("Chloramines (mg/L)", 0.0, 10.0, 5.0)
sulfate = st.sidebar.slider("Sulfate (mg/L)", 50.0, 500.0, 200.0)
conductivity = st.sidebar.slider("Conductivity (µS/cm)", 100.0, 1000.0, 400.0)
organic_carbon = st.sidebar.slider("Organic Carbon (mg/L)", 2.0, 30.0, 10.0)
trihalomethanes = st.sidebar.slider("Trihalomethanes (µg/L)", 0.0, 120.0, 60.0)
turbidity = st.sidebar.slider("Turbidity (NTU)", 1.0, 10.0, 3.0)

input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                        conductivity, organic_carbon, trihalomethanes, turbidity]])
input_data_scaled = scaler.transform(input_data)

# Prediction button
if st.button("🔍 Check Water Quality"):
    prediction = model.predict(input_data_scaled)[0]
    
    if prediction == 1:
        st.success("✅ **Water is safe for consumption.**")
    else:
        st.error("🚨 **ALERT: Water is unsafe! Immediate action needed.**")

# Additional UI elements for visuals
st.markdown("----")
st.markdown("### About the Model")
st.markdown("""
The **Random Forest Classifier** model used here predicts the potability of water based on various chemical and physical parameters like pH, hardness, turbidity, and others. The model was trained on a dataset of water potability and uses these inputs to determine whether the water is safe for consumption.
""")

# Footer
st.markdown("----")
st.markdown("""
#### 🚰 Developed by [Your Name]
#### Contact: your-email@example.com
""")
