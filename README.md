# Tesla Stock Price Prediction

## Overview
This project aims to predict whether the closing price of Tesla stock will be higher the next day using historical stock data. It's a binary classification problem where machine learning models help predict a buy signal based on past stock price data.

## Website link
https://tesla-stock-price-prediction-project.streamlit.app/

## Data
The dataset includes Tesla's daily stock prices from January 2010 to December 2017. Key features include:
- **Open**: Opening price
- **High**: Highest price
- **Low**: Lowest price
- **Close**: Closing price
- **Volume**: Total shares traded

## Features Engineered
- **Open-Close Spread**: Price difference between open and close
- **High-Low Spread**: Volatility indicator
- **Quarter-End Flag**: Indicator for quarter-end days

## Models Used
- Logistic Regression
- Support Vector Machines (SVM)
- XGBoost

## Evaluation
Model performance is evaluated using ROC-AUC, trained on 90% of the data with 10% reserved for validation.
