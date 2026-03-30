import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the model
model = joblib.load("LinearRegression.pkl")

# Load dataset
data = pd.read_csv("UPI_Fraud.csv")

# Label encoders
le_city = LabelEncoder()
le_city.fit(data['Transaction_City'])

le_channel = LabelEncoder()
le_channel.fit(data['Transaction_Channel'])

# UI
st.set_page_config(
    page_title="Sneha's UPI Fraud Detection",
    page_icon="👀",
    layout="wide"
)

st.title("💳 UPI Fraud Detection")

merchant_ID = st.slider(
    "Select Merchant ID",
    min_value=0,
    max_value=647,
    value=640
)

device_ID = st.slider(
    "Select Device ID",
    min_value=0,
    max_value=647,
    value=630
)

cities = list(data['Transaction_City'].unique())
location = st.selectbox(
    "Select Transaction City",
    options=cities,
    index=len(cities)-1
)

channels = list(data['Transaction_Channel'].unique())
default_channel_index = 0

for i, c in enumerate(channels):
    if "online" in str(c).lower() or "upi" in str(c).lower():
        default_channel_index = i
        break

transaction_channel = st.selectbox(
    "Select Transaction Channel",
    options=channels,
    index=default_channel_index
)

amount = st.slider(
    "Select Amount",
    min_value=float(data["amount"].min()),
    max_value=float(data["amount"].max()),
    value=float(data["amount"].max())
)

if st.button("Predict Fraud?"):
    encoded_city = le_city.transform([location])[0]
    encoded_channel = le_channel.transform([transaction_channel])[0]

    input_data = np.array([[merchant_ID, device_ID, encoded_city, encoded_channel, amount]])

    prediction = model.predict(input_data)[0]

    if amount > data["amount"].max() * 0.9:
        prediction = 1

    if prediction > 0.3:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction Seems Legitimate")
