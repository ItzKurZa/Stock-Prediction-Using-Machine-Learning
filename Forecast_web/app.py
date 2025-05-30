import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import traceback
import logging
from sklearn.metrics import r2_score

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Model configurations
models = {
    "xgboost": {
        "model": "../XGBoost Regressor/xgboost_model.pkl",
        "scaler": "../XGBoost Regressor/scaler.pkl",
        "feature_columns": "../XGBoost Regressor/xgboost_feature_columns.pkl",
        "target_cols": ["employment_rate_overall", "gross_monthly_mean"]
    },
    "random_forest": {
        "model": "../Random Forest/employment_model.pkl",
        "scaler": None,
        "feature_columns": "../Random Forest/feature_columns.pkl",
        "target_cols": ["employment_rate_overall", "gross_monthly_mean"]
    }
}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.errorhandler(500)
def internal_error(error):
    tb_str = ''.join(traceback.format_exception(None, error, error.__traceback__))
    logging.error("Internal Server Error:\n%s", tb_str)
    return jsonify({"error": "Internal Server Error", "details": tb_str}), 500

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model', 'xgboost')
    if model_choice not in models:
        return jsonify({"error": "Mô hình không hợp lệ"}), 400

    config = models[model_choice]
    model_path = config["model"]
    scaler_path = config["scaler"]
    feature_path = config["feature_columns"]
    target_cols = config["target_cols"]

    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        return jsonify({"error": f"Mô hình hoặc feature_columns không tồn tại: {model_choice}"}), 500

    try:
        model = joblib.load(model_path)
        feature_columns = joblib.load(feature_path)
        scaler = joblib.load(scaler_path) if scaler_path else None
    except Exception as e:
        return jsonify({"error": f"Lỗi khi tải mô hình hoặc scaler: {str(e)}"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        data = pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        return jsonify({"error": f"Không thể đọc file Excel: {str(e)}"}), 400

    data.columns = data.columns.str.strip().str.lower()
    data.replace('na', np.nan, inplace=True)
    data.dropna(inplace=True)

    required_cols = ["year", "university", "degree"] + target_cols
    for col in required_cols:
        if col not in data.columns:
            return jsonify({"error": f"Thiếu cột '{col}' trong dữ liệu"}), 400

    orig_year = data["year"]
    orig_university = data["university"]
    orig_degree = data["degree"]

    X = data.drop(columns=target_cols, errors='ignore')
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)

    try:
        X_proc = scaler.transform(X) if scaler else X
        predictions = model.predict(X_proc)

        if predictions.ndim == 1 and len(target_cols) == 2:
            predictions = np.column_stack((predictions, np.zeros_like(predictions)))
    except Exception as e:
        return jsonify({"error": f"Lỗi khi dự đoán: {str(e)}"}), 500

    results = []
    for i in range(len(predictions)):
        results.append({
            "year": int(orig_year.iloc[i]) if i < len(orig_year) else "",
            "university": orig_university.iloc[i] if i < len(orig_university) else "",
            "degree": orig_degree.iloc[i] if i < len(orig_degree) else "",
            "predicted_employment_rate_overall": round(float(predictions[i][0]), 2),
            "predicted_gross_monthly_mean": round(float(predictions[i][1]), 2)
        })

    if not results:
        return jsonify({"error": "Không có dữ liệu dự đoán."}), 400

    return jsonify({"predictions": results}), 200

@app.route('/accuracy_graph', methods=['GET'])
def accuracy_graph():
    model_choice = request.args.get('model', 'xgboost')
    if model_choice not in models:
        return jsonify({"error": "Mô hình không hợp lệ"}), 400

    graph_filename = (
        '../accuracy_img/XGB_accuracy_graph.png' if model_choice == 'xgboost' else '../accuracy_img/accuracy_graph.png'
    )
    graph_path = os.path.join('static', graph_filename)

    if not os.path.exists(graph_path):
        return jsonify({"error": f"Ảnh đồ thị cho mô hình {model_choice} chưa được tạo!"}), 404

    try:
        with open(graph_path, "rb") as f:
            return Response(f.read(), mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Lỗi khi đọc ảnh: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
