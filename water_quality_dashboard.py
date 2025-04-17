import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set up Streamlit page config
st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="centered")

# Title and Introduction
st.title("💧 **Water Quality Prediction**")
st.markdown("""
**Enter water parameters below to check if the water is safe to drink.**
The system will give you immediate feedback on the quality of the water.
""")

# User Input for Water Quality Parameters
st.sidebar.header("Input Water Parameters")
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

# Load and Train Model (Placeholder for actual model training code)
@st.cache_data
def load_data():
    # Placeholder code: Replace with actual data loading and model training
    df = pd.read_csv("water_potability.csv")
    df = df.fillna(df.median(numeric_only=True))  # Fill missing values with the median
    return df

df = load_data_
