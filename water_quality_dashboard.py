import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set up Streamlit page config
st.set_page_config(page_title="💧 Water Quality Prediction", page_icon="💧", layout="wide")

# Title and Introduction
st.title("💧 **Water Quality Prediction & Alert System**")
st.markdown("""
This system helps you predict whether the water is safe to drink based on various parameters.
Simply enter the water test values below, and get immediate feedback on whether the water is safe or unsafe.
""")

# Sidebar for user inputs
st.sidebar.header("🌊 **Enter Water Quality Parameters**")
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

# Data preprocessing and model training
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    df = df.fillna(df.median(numeric_only=True))  # Fill missing values with the median
    return df

df = load_data()

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔍 Check Water Quality"):
    prediction = model.predict(input_data_scaled)[0]
    
    if prediction == 1:
        st.success("✅ **Water is safe for consumption.**")
        st.markdown("**Good to go!** You can drink this water without any worries.")
    else:
        st.error("🚨 **ALERT: Water is unsafe! Immediate action needed.**")
        st.markdown("**Warning!** The water is unsafe for consumption. Please take necessary steps to purify the water.")

# Section for visuals or model explanation
st.markdown("----")
st.markdown("### **How the System Works**")
st.markdown("""
This system uses machine learning models to assess the potability of water based on various input features such as pH, hardness, turbidity, and more. The model predicts whether the water is safe to drink (Potability = 1) or not (Potability = 0).
""")
