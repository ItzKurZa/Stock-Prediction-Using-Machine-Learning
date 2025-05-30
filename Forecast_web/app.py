


import io
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import traceback
from sklearn.metrics import r2_score

app = Flask(__name__)
CORS(app)

# Cấu hình cho hai mô hình
models = {
    "xgboost": {
        "model": "XGBoost Regressor/xgboost_model.pkl",
        "scaler": "XGBoost Regressor/scaler.pkl",
        "feature_columns": "XGBoost Regressor/xgboost_feature_columns.pkl",
        "target_cols": ["employment_rate_overall", "gross_monthly_mean"]
    },
    "random_forest": {
        "model": "employment_model.pkl",
        "scaler": None,
        "feature_columns": "feature_columns.pkl",
        "target_cols": ["employment_rate_overall", "gross_monthly_mean"]
    }
}

@app.errorhandler(500)
def internal_error(error):
    tb_str = ''.join(traceback.format_exception(None, error, error.__traceback__))
    print("Internal Server Error:\n", tb_str)
    return jsonify({"error": "Internal Server Error", "details": tb_str}), 500

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model', 'xgboost')
    if model_choice not in models:
        return jsonify({"error": "Mô hình không hợp lệ"}), 400

    model_config = models[model_choice]
    model_path = model_config["model"]
    scaler_path = model_config["scaler"]
    feat_path = model_config["feature_columns"]
    target_cols = model_config["target_cols"]

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        return jsonify({"error": f"Mô hình {model_choice} hoặc file feature_columns chưa được tạo!"}), 500

    try:
        model = joblib.load(model_path)
        feature_columns = joblib.load(feat_path)
    except Exception as e:
        return jsonify({"error": "Lỗi khi tải mô hình/feature_columns: " + str(e)}), 500

    scaler = None
    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            return jsonify({"error": "Lỗi khi tải scaler: " + str(e)}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        data = pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        return jsonify({"error": "Không thể đọc file Excel: " + str(e)}), 400

    data.columns = data.columns.str.strip().str.lower()
    data.replace('na', np.nan, inplace=True)
    data.dropna(inplace=True)

    required_cols = ["year", "university", "degree"] + target_cols
    for col in required_cols:
        if col not in data.columns:
            return jsonify({"error": f"Thiếu cột '{col}' trong dữ liệu"}), 400

    orig_year = data["year"].copy()
    orig_university = data["university"].copy()
    orig_degree = data["degree"].copy()

    X = data.drop(columns=target_cols, errors='ignore')
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)

    if scaler:
        X_proc = scaler.transform(X)
    else:
        X_proc = X

    try:
        predictions = model.predict(X_proc)
        if predictions.ndim == 1:
            predictions = np.array([[p, 0] for p in predictions])
    except Exception as e:
        return jsonify({"error": "Lỗi khi dự đoán: " + str(e)}), 500

    n_samples = predictions.shape[0]
    results = []
    for i in range(n_samples):
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


# lấy ảnh từ file train
@app.route('/accuracy_graph', methods=['GET'])
def accuracy_graph():
    model_choice = request.args.get('model', 'xgboost')

    # Kiểm tra xem mô hình có hợp lệ không
    if model_choice not in models:
        return jsonify({"error": "Mô hình không hợp lệ"}), 400

    try:
        # Định nghĩa tên ảnh tùy theo mô hình được chọn
        if model_choice == 'xgboost':
            graph_filename = 'XGB_accuracy_graph.png'
        elif model_choice == 'random_forest':
            graph_filename = 'accuracy_graph.png'

        # Kiểm tra sự tồn tại của tệp ảnh trong thư mục static
        graph_path = os.path.join( graph_filename)
        if not os.path.exists(graph_path):
            return jsonify({"error": f"Ảnh đồ thị cho mô hình {model_choice} chưa được tạo!"}), 404

        # Đọc ảnh đã lưu từ thư mục 'static'
        with open(graph_path, "rb") as f:
            return Response(f.read(), mimetype='image/png')

    except Exception as e:
        return jsonify({"error": "Lỗi khi tạo đồ thị: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)












