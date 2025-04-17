import streamlit as st
import numpy as np
import pandas as pd
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

# Sample Data to train the model (This will be replaced with actual data in your use case)
@st.cache_data
def load_data():
    # Sample mock data for testing
    data = {
        'ph': [7.0, 6.5, 7.2, 8.0, 6.8],
        'Hardness': [200, 180, 250, 190, 210],
        'Solids': [10000, 20000, 15000, 12000, 13000],
        'Chloramines': [5.0, 4.5, 5.5, 6.0, 4.8],
        'Sulfate': [200, 180, 210, 190, 200],
        'Conductivity': [400, 450, 500, 470, 480],
        'Organic_carbon': [10, 12, 8, 9, 11],
        'Trihalomethanes': [60, 50, 70, 65, 60],
        'Turbidity': [3, 4, 2, 3, 3],
        'Potability': [1, 0, 1, 0, 1]  # 1 = Safe, 0 = Unsafe
    }
    df = pd.DataFrame(data)
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
    
    # Show result (highlight message in green for safe and red for unsafe)
    if prediction == 1:
        st.success("✅ **Water is Safe for Consumption**")
    else:
        st.error("🚨 **Water is Unsafe! Immediate Action Needed**")
