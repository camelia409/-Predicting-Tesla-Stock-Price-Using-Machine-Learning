import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# ✅ Step 1: Ensure 'models/' directory exists and download missing models
MODEL_URLS = {
    "logistic_regression_model.pkl": "https://raw.githubusercontent.com/camelia409/-Predicting-Tesla-Stock-Price-Using-Machine-Learning/main/models/logistic_regression_model.pkl",
    "svc_model.pkl": "https://raw.githubusercontent.com/camelia409/-Predicting-Tesla-Stock-Price-Using-Machine-Learning/main/models/svc_model.pkl",
    "xgboost_model.pkl": "https://raw.githubusercontent.com/camelia409/-Predicting-Tesla-Stock-Price-Using-Machine-Learning/main/models/xgboost_model.pkl"
}

os.makedirs("models", exist_ok=True)

for model_name, url in MODEL_URLS.items():
    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        st.write(f"Downloading {model_name}...")
        response = requests.get(url)
        with open(model_path, "wb") as file:
            file.write(response.content)
        st.write(f"{model_name} downloaded successfully.")

# ✅ Step 2: Verify models directory in Streamlit
st.write("📂 Current working directory:", os.getcwd())
st.write("📂 Models directory exists:", os.path.exists("models"))
st.write("📂 Contents of 'models/' directory:", os.listdir("models") if os.path.exists("models") else "Directory not found")

# ✅ Step 3: Load models dynamically
models = {}
EXPECTED_FEATURES = ["Open", "High", "Low", "Close", "Volume", "Open_Close_Spread", "High_Low_Spread", "Quarter_End_Flag"]

for file_name in os.listdir("models"):
    if file_name.endswith(".pkl"):
        model_name = file_name.split(".pkl")[0]
        try:
            file_path = os.path.join("models", file_name)
            try:
                models[model_name] = joblib.load(file_path)
            except Exception:
                with open(file_path, "rb") as f:
                    models[model_name] = pickle.load(f)
        except Exception as e:
            st.error(f"Failed to load {model_name}: {e}")

# ✅ Step 4: Fetch stock data
ticker = "TSLA"

@st.cache_data
def fetch_stock_data(start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        st.error("⚠️ No data fetched from Yahoo Finance.")
        return None
    stock_data.reset_index(inplace=True)
    return stock_data

# ✅ Step 5: Preprocess data
def preprocess_data(stock_data, input_date):
    stock_data["Open_Close_Spread"] = stock_data["Close"] - stock_data["Open"]
    stock_data["High_Low_Spread"] = stock_data["High"] - stock_data["Low"]
    stock_data["Quarter_End_Flag"] = stock_data["Date"].dt.month % 3 == 0
    stock_data.dropna(inplace=True)

    input_date = pd.to_datetime(input_date).replace(tzinfo=None)
    stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.tz_localize(None)

    if input_date > stock_data["Date"].max():
        st.error("⚠️ Selected date exceeds available data.")
        return None

    filtered_data = stock_data[stock_data["Date"] <= input_date]
    if filtered_data.empty:
        st.error(f"⚠️ No data available for {input_date}")
        return None

    features = filtered_data.iloc[-1][EXPECTED_FEATURES].values.reshape(1, -1)
    return features, filtered_data

# ✅ Step 6: Streamlit App UI
st.title("📈 Tesla Stock Price Movement Prediction")

st.sidebar.header("🔍 Model Selection")
selected_model_name = st.sidebar.selectbox("Select a model", list(models.keys()))

st.sidebar.header("📅 Stock Data Input")
input_date = st.sidebar.date_input("Prediction Date (prior to today):", max_value=datetime.today().date())

if st.sidebar.button("🚀 Fetch and Predict"):
    stock_data = fetch_stock_data("2010-01-01", datetime.today().date().strftime("%Y-%m-%d"))
    
    if stock_data is not None:
        features, filtered_data = preprocess_data(stock_data, input_date)
        
        if features is not None:
            st.write(f"✅ Using model: **{selected_model_name}**")
            model = models[selected_model_name]
            prediction = model.predict(features)
            prediction_label = "📈 Up" if prediction[0] == 1 else "📉 Down"
            st.write(f"### Prediction for {input_date}: **{prediction_label}**")

            last_open = round(float(filtered_data.iloc[-1]["Open"]), 2)
            last_close = round(float(filtered_data.iloc[-1]["Close"]), 2)
            st.write(f"### Last available prices on {input_date}: **TSLA Open: {last_open}, Close: {last_close}**")

            st.write("### 📊 Stock Price Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(stock_data["Date"], stock_data["Close"], label="Close Price", color="blue")
            ax.axvline(pd.to_datetime(input_date), color="red", linestyle="--", label="Prediction Date")
            ax.set_title(f"Tesla Stock Price with {selected_model_name} Predictions")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
