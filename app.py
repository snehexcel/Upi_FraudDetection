import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="UPI Transaction Category Predictor",
    page_icon="💳",
    layout="wide"
)

st.title("💳 UPI Transaction Category Predictor")
st.write("Predict the transaction category using withdrawal, deposit and balance.")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("MyTransaction.csv")

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.dropna(subset=["Category"])
df["Withdrawal"] = df["Withdrawal"].fillna(0)
df["Deposit"] = df["Deposit"].fillna(0)
df["Balance"] = df["Balance"].fillna(0)

# Features & target
X = df[["Withdrawal", "Deposit", "Balance"]]
y = df["Category"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y_encoded)

# -----------------------------
# User Inputs
# -----------------------------
withdrawal = st.number_input("Enter Withdrawal Amount", min_value=0.0, value=100.0)
deposit = st.number_input("Enter Deposit Amount", min_value=0.0, value=0.0)
balance = st.number_input("Enter Current Balance", min_value=0.0, value=1000.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Category"):
    input_data = np.array([[withdrawal, deposit, balance]])
    prediction = model.predict(input_data)[0]
    category = le.inverse_transform([prediction])[0]

    st.success(f"✅ Predicted Category: {category}")
