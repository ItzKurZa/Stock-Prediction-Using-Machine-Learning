📈 Stock Price Prediction with Machine Learning
This project leverages machine learning models to predict stock prices based on historical data. It includes a Flask-based API that allows users to input a stock ticker and receive future price predictions.

🧠 Project Overview
Data Acquisition: Historical stock data is fetched from Yahoo Finance using the yfinance library.

Data Preprocessing: Utilizes pandas and numpy for data cleaning and normalization.

Machine Learning Models: Implements models such as LightGBM, XGBoost, and Random Forest from scikit-learn for training and prediction.

Model Evaluation: Employs metrics like MAE, RMSE, and R² to assess model performance.

API Interface: Provides a RESTful API using Flask for users to request predictions.

📁 Project Structure
bash
Copy
Edit
├── app.py                  # Flask API for serving predictions
├── stock_prediction.py     # Data processing and model training
├── requirements.txt        # List of required libraries
├── README.md               # Project documentation
🚀 Installation Guide
1. Create a Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
2. Install Required Libraries
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Application
bash
Copy
Edit
python app.py
The application will be accessible at http://localhost:5000.

🛠️ API Endpoint
POST /predict

Request Parameters (JSON):

json
Copy
Edit
{
  "ticker": "AAPL",
  "start_date": "2020-01-01",
  "end_date": "2024-12-31"
}
Response (JSON):

json
Copy
Edit
{
  "predicted_price": 175.23,
  "model": "LightGBM",
  "metrics": {
    "MAE": 1.12,
    "RMSE": 1.45,
    "R2": 0.92
  }
}
📦 Dependencies
The project relies on the following Python libraries:

Flask==3.1.1

joblib==1.5.1

lightgbm==4.6.0

matplotlib==3.10.3

numpy==2.2.6

pandas==2.2.3

seaborn==0.13.2

scikit-learn==1.6.1

xgboost==3.0.0

yfinance==0.2.59

📊 Model Performance
The LightGBM model demonstrates high prediction accuracy, as indicated by evaluation metrics such as MAE, RMSE, and R². Visual comparisons between actual and predicted prices are generated using matplotlib and seaborn.

📌 Notes
Ensure an active internet connection to fetch data from Yahoo Finance.

The code is compatible with Python 3.10 and above.

Future enhancements may include support for additional models and features.

If you need further assistance or have any questions, feel free to reach out to the development team.
