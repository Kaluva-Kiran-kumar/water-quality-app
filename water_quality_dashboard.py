# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb

# Load the dataset
df = pd.read_csv('water_potability.csv')  # Make sure this file is in your working directory
print(df.head())

# Check for missing values and handle them
print("\n❓ Missing Values:")
print(df.isnull().sum())

# Fill missing values with the median of each column
df = df.fillna(df.median())

# Exploratory Data Analysis (EDA)
df.hist(bins=20, figsize=(10, 10))
plt.suptitle('Feature Distributions')
plt.show()

sns.boxplot(x='Potability', y='ph', data=df)
plt.title('pH vs Potability')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Split the data into features and target
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# Train XGBoost Classifier
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

# Display classification reports for each model
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Step 5: Alert System (Basic example using Random Forest)
def check_water_quality(input_data):
    """
    Function to predict water quality and send an alert if unsafe.
    
    Args:
    input_data: list of feature values (e.g., pH, Hardness, Sulfate, etc.)
    
    Returns:
    str: Alert message (safe or unsafe)
    """
    # Standardize the input data to match the format of the training data
    input_data_scaled = scaler.transform([input_data])  
    
    # Predict using the Random Forest model
    prediction = rf_clf.predict(input_data_scaled)[0]
    
    if prediction == 0:
        return "🚨 ALERT: Water is unsafe! Immediate action needed!"
    else:
        return "✅ Water is safe for consumption."

# Example Usage: (Replace with actual input data)
input_data_example = [7.5, 200, 5000, 5.2, 300, 450, 12, 50, 2]  # Example feature values
alert = check_water_quality(input_data_example)
print(alert)
