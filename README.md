# 📈 Stock Price Prediction with Machine Learning

This project predicts future stock prices using machine learning techniques. It combines powerful Python libraries with a web-based Flask interface, supporting both real-time and synthetic data. The system supports multiple models, including XGBoost, LightGBM, SVR, Random Forest, and more.

## 🔧 Features

- 🏦 Fetch historical stock data via [yfinance](https://pypi.org/project/yfinance/)
- 🧠 Technical indicator calculation: Moving Averages, RSI, MACD, Bollinger Bands, Volatility
- 📊 PCA for dimensionality reduction
- 🤖 Multiple regression models (XGBoost, LightGBM, SVR, KNN, Random Forest, Ridge)
- 📈 Evaluation metrics: MSE, R²
- 🖼️ Prediction visualization with Matplotlib
- 🌐 Web interface using Flask
- 💾 Model persistence and data quality validation
- 📁 Exportable data and results

---

## 🧰 Tech Stack

- Python 3.x
- Flask
- scikit-learn
- yfinance
- XGBoost, LightGBM
- Matplotlib, Seaborn
- Pandas, NumPy

---

## 📁 Project Structure

```bash
├── stock_prediction.py      # Core ML logic & model training
├── app.py                   # Flask web app
├── templates/
│   └── index.html           # Web interface template
├── models/                  # Saved models per ticker
├── data/                    # CSV exports of raw and feature data
├── data_issues/             # Automatically generated data quality reports
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is missing, install manually:

```bash
pip install flask yfinance pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib
```

### 3. Run Main Script (Batch Mode)

```bash
python stock_prediction.py
```

This will:
- Fetch stock data for selected tickers
- Compute features & indicators
- Train & evaluate models (original and PCA-reduced)
- Save models and results to disk
- Generate performance comparison charts

### 4. Run Flask Web App (Interactive Mode)

```bash
python app.py
```

Then open your browser at: http://127.0.0.1:5000
You can input a stock ticker and date range to:
- See model predictions
- View comparative model performance
- Get price forecast for the next day

---

## 📉 Example Outputs

- 📊 Model Comparison Table (MSE, R²)
- 🖼️ Forecast Chart: Actual vs Predicted
- 💾 Saved Models: XGBoost, LightGBM, etc.
- 🗃️ Cleaned Datasets in CSV

---

## 🔍 Supported Tickers (by default)

- AAPL
- MSFT
- GOOGL
- AMZN
- META

If stock data is unavailable, the script generates synthetic time series data.

---

## 📌 Notes

- The models predict next-day closing prices
- PCA is optional and automatically compared with the original feature set
- Data validation includes checks for missing, invalid, or duplicate values
