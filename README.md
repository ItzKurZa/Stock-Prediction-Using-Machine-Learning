# 📈 Stock Price Prediction with Machine Learning

This project leverages machine learning models to predict stock prices based on historical data. It provides a RESTful API built with Flask, allowing users to input a stock ticker and receive future price predictions.

## 🧠 Project Overview

- **Data Source**: Historical stock data is fetched from Yahoo Finance using the `yfinance` library.
- **Preprocessing**: Data is cleaned and prepared using `pandas` and `numpy`.
- **Machine Learning Models**: Implements models like LightGBM, XGBoost, and Random Forest via `scikit-learn`.
- **Model Evaluation**: Utilizes metrics such as MAE, RMSE, and R² to assess model performance.
- **API Interface**: A Flask-based RESTful API enables users to request predictions.

## 📁 Project Structure

```bash
├── app.py                  # Flask API to serve predictions
├── stock_prediction.py     # Data processing and model training
├── requirements.txt        # List of required libraries
├── README.md               # Project documentation
