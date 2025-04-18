import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title
st.title("💧 Smart Water Quality Prediction and Contamination Alert System")
st.markdown("This app predicts **whether water is potable** (safe for drinking) based on various quality parameters.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    return df.dropna()

df = load_data()

# Show Data
if st.checkbox("🔍 Show Raw Data"):
    st.write(df.head())

# EDA Visualization
if st.checkbox("📊 Show Data Distribution"):
    st.subheader("Potability Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Potability', data=df, ax=ax)
    st.pyplot(fig)

# Preprocessing
X = df.drop("Potability", axis=1)
y = df["Potability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Performance")
st.write(f"**Accuracy**: {accuracy:.2f}")

# User Input
st.subheader("🔬 Check Your Water Sample")

ph = st.slider("pH Level", 0.0, 14.0, 7.0)
hardness = st.slider("Hardness", 0.0, 300.0, 150.0)
solids = st.slider("Solids (ppm)", 0.0, 50000.0, 20000.0)
chloramines = st.slider("Chloramines", 0.0, 15.0, 5.0)
sulfate = st.slider("Sulfate", 0.0, 500.0, 250.0)
conductivity = st.slider("Conductivity", 0.0, 1000.0, 400.0)
organic_carbon = st.slider("Organic Carbon", 0.0, 30.0, 10.0)
trihalomethanes = st.slider("Trihalomethanes", 0.0, 120.0, 60.0)
turbidity = st.slider("Turbidity", 0.0, 10.0, 3.0)

input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                        conductivity, organic_carbon, trihalomethanes, turbidity]])

if st.button("🚰 Predict Potability"):
    result = model.predict(input_data)
    st.success("✅ Potable Water" if result[0] == 1 else "❌ Not Potable Water")
