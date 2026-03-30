import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="UPI Transaction Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 UPI Transaction Dashboard & Category Predictor")
st.markdown("### 📊 Analyze your transactions and predict category")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("MyTransaction.csv")

# Clean data
df = df.dropna(subset=["Category"])
df["Withdrawal"] = df["Withdrawal"].fillna(0)
df["Deposit"] = df["Deposit"].fillna(0)
df["Balance"] = df["Balance"].fillna(0)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("📌 Input Transaction Details")

withdrawal = st.sidebar.number_input(
    "Withdrawal Amount",
    min_value=0.0,
    value=100.0
)

deposit = st.sidebar.number_input(
    "Deposit Amount",
    min_value=0.0,
    value=0.0
)

balance = st.sidebar.number_input(
    "Current Balance",
    min_value=0.0,
    value=1000.0
)

# -----------------------------
# Model Training
# -----------------------------
X = df[["Withdrawal", "Deposit", "Balance"]]
y = df["Category"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = LogisticRegression(max_iter=1000)
model.fit(X, y_encoded)

# -----------------------------
# KPI Cards
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💸 Total Withdrawal", f"{df['Withdrawal'].sum():,.2f}")
col2.metric("💰 Total Deposit", f"{df['Deposit'].sum():,.2f}")
col3.metric("🏦 Current Avg Balance", f"{df['Balance'].mean():,.2f}")

# -----------------------------
# Charts
# -----------------------------
st.subheader("📈 Transaction Category Distribution")

fig, ax = plt.subplots(figsize=(8, 4))
df["Category"].value_counts().plot(kind="bar", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# -----------------------------
# Prediction
# -----------------------------
st.subheader("🔍 Predict New Transaction Category")

if st.button("Predict Category"):
    pred = model.predict([[withdrawal, deposit, balance]])[0]
    category = le.inverse_transform([pred])[0]

    st.success(f"✅ Predicted Category: **{category}**")
