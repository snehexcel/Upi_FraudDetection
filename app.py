import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Sneha's UPI Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 UPI Fraud Detection System")
st.write("Check whether a UPI transaction is **fraudulent or legitimate**.")

# =========================
# Safe File Paths
# =========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "LinearRegression.pkl")
DATA_PATH = os.path.join(BASE_DIR, "UPI_Fraud.csv")

# =========================
# Load Model & Data
# =========================
try:
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# =========================
# Label Encoding
# =========================
le_city = LabelEncoder()
le_city.fit(data["Transaction_City"].astype(str))

le_channel = LabelEncoder()
le_channel.fit(data["Transaction_Channel"].astype(str))

# =========================
# Input Widgets
# =========================
merchant_ID = st.slider(
    "Select Merchant ID",
    min_value=int(data["Merchant_ID"].min()),
    max_value=int(data["Merchant_ID"].max()),
    value=int(data["Merchant_ID"].max())
)

device_ID = st.slider(
    "Select Device ID",
    min_value=int(data["Device_ID"].min()),
    max_value=int(data["Device_ID"].max()),
    value=int(data["Device_ID"].max())
)

cities = sorted(data["Transaction_City"].astype(str).unique())
location = st.selectbox(
    "Select Transaction City",
    options=cities
)

channels = sorted(data["Transaction_Channel"].astype(str).unique())
transaction_channel = st.selectbox(
    "Select Transaction Channel",
    options=channels
)

amount = st.slider(
    "Select Amount",
    min_value=float(data["amount"].min()),
    max_value=float(data["amount"].max()),
    value=float(data["amount"].max() * 0.8)
)

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Fraud"):
    try:
        encoded_city = le_city.transform([location])[0]
        encoded_channel = le_channel.transform([transaction_channel])[0]

        input_data = np.array(
            [[merchant_ID, device_ID, encoded_city, encoded_channel, amount]]
        )

        prediction = model.predict(input_data)[0]

        # Demo fraud logic for high amount
        if amount > data["amount"].max() * 0.9:
            prediction = 1

        st.subheader("📊 Prediction Result")

        if prediction > 0.3:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction Seems Legitimate")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
