import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf  # Yahoo Finance API

# Load all three models with error handling
model_files = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Support Vector Regression (SVR)": "svr_model.pkl",
    "XGBoost Classifier": "xgboost_model.pkl",
}

models = {}
for name, file in model_files.items():
    try:
        with open(file, "rb") as f:
            models[name] = pickle.load(f)
        st.success(f"{name} model loaded successfully.")
    except (pickle.UnpicklingError, FileNotFoundError, EOFError) as e:
        st.error(f"Failed to load {name}: {e}")

# Define expected features for each model
EXPECTED_FEATURES = [
    "Open", "High", "Low", "Close", "Volume", 
    "Open_Close_Spread", "High_Low_Spread", "Quarter_End_Flag"
]

# Fixed stock ticker for Tesla
ticker = "TSLA"

# Fetch stock data from Yahoo Finance
@st.cache_data
def fetch_stock_data(start_date, end_date):
    st.write("Fetching stock data from Yahoo Finance...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        st.error("No data fetched from Yahoo Finance. Please check the ticker or date range.")
        return None
    stock_data.reset_index(inplace=True)
    return stock_data

def preprocess_data(stock_data, input_date, selected_model_name):
    # Derived features
    stock_data["Open_Close_Spread"] = stock_data["Close"] - stock_data["Open"]
    stock_data["High_Low_Spread"] = stock_data["High"] - stock_data["Low"]
    
    # Add features
    stock_data["Quarter_End_Flag"] = stock_data["Date"].dt.month % 3 == 0
    stock_data["MA_5"] = stock_data["Close"].rolling(window=5).mean()
    stock_data["MA_10"] = stock_data["Close"].rolling(window=10).mean()
    stock_data["Volatility"] = stock_data["Close"].rolling(window=10).std()

    # Drop NaN values introduced by rolling features
    stock_data.dropna(inplace=True)
    
    # Ensure the final feature set matches the expected columns
    if not all(col in stock_data.columns for col in EXPECTED_FEATURES):
        st.error(f"Missing required columns: {set(EXPECTED_FEATURES) - set(stock_data.columns)}")
        return None
    
    # Filter data based on the input date
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    
    # Convert input_date to datetime
    input_date = pd.to_datetime(input_date)
    
    # Make sure input_date is less than or equal to the last available data point
    if input_date > stock_data["Date"].max():
        st.error("Selected prediction date exceeds available data. Please select a date within the available range.")
        return None
    
    # Filter data based on the input date
    filtered_data = stock_data[stock_data["Date"] <= input_date]
    
    if filtered_data.empty:
        st.error(f"No data available for the selected date: {input_date}")
        return None
    
    # Extract features based on the selected model
    if selected_model_name in ["Logistic Regression", "Support Vector Regression (SVR)", "XGBoost Classifier"]:
        # Use only 'Open' and 'Close' features for all models
        features = filtered_data[["Open", "Close"]].iloc[-1, :].values.reshape(1, -1)
    
    return features


# Main Streamlit App
st.title("Tesla Stock Price Movement Prediction")

# Sidebar for user inputs
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Select a model", list(models.keys()))

st.sidebar.header("Stock Data Input")
input_date = st.sidebar.date_input("Prediction Date (prior to today):", max_value=datetime.today().date())

if st.sidebar.button("Fetch and Predict"):
    # Fetch stock data (using a fixed large date range here as it's only needed for fetching past data)
    stock_data = fetch_stock_data("2010-01-01", datetime.today().date().strftime("%Y-%m-%d"))
    
    if stock_data is not None:
        # Preprocess data based on the input date and selected model
        features = preprocess_data(stock_data, input_date, selected_model_name)
        
        if features is not None:
            # Make prediction using the selected model
            st.write(f"Using model: {selected_model_name}")
            model = models[selected_model_name]
            prediction = model.predict(features)
            
            # Show the prediction result
            prediction_label = "Up" if prediction[0] == 1 else "Down"
            st.write(f"Prediction for {input_date}: **{prediction_label}**")

            # Display the stock data (showing recent close price and prediction)
            st.write("## Stock Data with Prediction")
            st.write(f"Last available close price on {input_date}: {stock_data.iloc[-1]['Close']}")
            
            # Plot the historical stock data with the prediction
            st.write("## Visualizing Stock Price Movement")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(stock_data["Date"], stock_data["Close"], label="Close Price", color="blue")
            ax.axvline(pd.to_datetime(input_date), color="red", linestyle="--", label="Prediction Date")
            ax.set_title(f"Tesla Stock Price with {selected_model_name} Predictions")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
