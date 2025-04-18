code = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Title
st.title("Smart Water Quality Prediction and Contamination Alert System")

# Upload file
uploaded_file = st.file_uploader("Upload Water Quality CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Show basic stats
    st.write("Basic Description:")
    st.write(df.describe())

    # Visualizations
    st.write("Water Quality Feature Correlation Heatmap:")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Model
    if "Potability" in df.columns:
        X = df.drop("Potability", axis=1)
        y = df["Potability"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.write("Confusion Matrix:")
        st.text(confusion_matrix(y_test, y_pred))
    else:
        st.warning("Dataset must have 'Potability' as target column.")
"""

# Save it to a .py file
with open("water_quality_dashboard.py", "w") as f:
    f.write(code)
