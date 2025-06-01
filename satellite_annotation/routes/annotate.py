import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Blueprint, request, jsonify, render_template, flash, current_app, send_file, url_for
from flask_login import login_required, current_user
from models import db, StockData, StockPrediction, Folder, ModelPerformance
import numpy as np
import joblib
import io
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from werkzeug.utils import secure_filename, redirect
from flask import current_app as app
import zipfile
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb
import lightgbm as lgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import json

# Đường dẫn đến mô hình và scaler đã lưu
model_path = r"D:\Code\model\stock_models"
os.makedirs(model_path, exist_ok=True)

# Global variables for models
models = {}
scaler = None
pca = None
use_pca = False

# Danh sách các mô hình hỗ trợ
SUPPORTED_MODELS = ['XGBoost', 'LightGBM', 'KNN', 'RandomForest', 'SVR', 'Ridge']

# Load mô hình và scaler từ file
try:
    # Load PCA và scaler
    pca = joblib.load(os.path.join(model_path, "pca.pkl"))
    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
    
    # Xác định xem có sử dụng PCA hay không
    use_pca = hasattr(pca, 'n_components_') and pca.n_components_ > 1
    
    # Load các mô hình
    for model_name in SUPPORTED_MODELS:
        model_file = os.path.join(model_path, f"{model_name}_model.pkl")
        if os.path.exists(model_file):
            models[model_name] = joblib.load(model_file)
    
    if models:
        print(f"Đã tải {len(models)} mô hình thành công: {', '.join(models.keys())}")
        print(f"Sử dụng PCA: {use_pca}")
        models_loaded = True
    else:
        print("Không tìm thấy mô hình nào")
        models_loaded = False
except Exception as e:
    print(f"Lỗi khi tải mô hình: {str(e)}")
    print("Chưa có mô hình, sẽ tạo mới khi có dữ liệu")
    models_loaded = False

# Tạo Blueprint
annotate_bp = Blueprint('annotate', __name__)

# Helper functions
def get_stock_data(ticker, start_date, end_date):
    """Lấy dữ liệu chứng khoán từ yfinance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df, None
    except Exception as e:
        return None, str(e)

def calculate_technical_indicators(df):
    """Tính toán các chỉ báo kỹ thuật"""
    # Returns
    df['Returns'] = df['Close'].pct_change()
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # Volume indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
    
    return df

def prepare_data_for_prediction(df):
    """Chuẩn bị dữ liệu cho dự đoán"""
    # Chọn features
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'MA5', 'MA20', 'MA50', 'RSI',
                'MACD', 'Signal_Line', 'BB_middle', 'BB_upper',
                'BB_lower', 'Volume_MA5', 'Volume_MA20', 'Volatility']
    
    # Xóa các dòng có giá trị NaN
    df = df.dropna()
    
    X = df[features]
    
    # Chuẩn hóa dữ liệu
    if not models_loaded:
        global scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))
    else:
        X_scaled = scaler.transform(X)
    
    # Áp dụng PCA nếu cần
    if use_pca:
        X_scaled = pca.transform(X_scaled)
    
    return X_scaled, df.index

def apply_pca(X_train, X_test, n_components=0.95):
    """Áp dụng PCA để giảm chiều dữ liệu"""
    global pca, use_pca
    
    # Chuẩn hóa dữ liệu trước khi áp dụng PCA
    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_test_scaled = std_scaler.transform(X_test)
    
    # Áp dụng PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    use_pca = True
    joblib.dump(pca, os.path.join(model_path, "pca.pkl"))
    joblib.dump(std_scaler, os.path.join(model_path, "scaler.pkl"))
    
    return X_train_pca, X_test_pca

def train_models(X, y):
    """Huấn luyện các mô hình"""
    global models, models_loaded
    
    trained_models = {}
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X, y)
    trained_models['XGBoost'] = xgb_model
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    lgb_model.fit(X, y)
    trained_models['LightGBM'] = lgb_model
    
    # K-Nearest Neighbors
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X, y)
    trained_models['KNN'] = knn_model
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    trained_models['RandomForest'] = rf_model
    
    # Support Vector Regression
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr_model.fit(X, y)
    trained_models['SVR'] = svr_model
    
    # Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y)
    trained_models['Ridge'] = ridge_model
    
    # Lưu mô hình
    for name, model in trained_models.items():
        joblib.dump(model, os.path.join(model_path, f"{name}_model.pkl"))
    
    models = trained_models
    models_loaded = True
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Đánh giá các mô hình"""
    results = {}
    predictions = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
        predictions[name] = y_pred
    
    return results, predictions

# Routes

@annotate_bp.route('/dashboard')
@login_required
def dashboard():
    result = StockPrediction.query.filter_by(user_id=current_user.id).order_by(StockPrediction.prediction_date.desc()).all()
    return render_template('dashboard.html', results=result)

@annotate_bp.route('/delete/<string:id_data>', methods = ['GET'])
@login_required
def delete(id_data):
    # Tìm bản ghi theo ID
    record = StockPrediction.query.get_or_404(id_data)

    # Xóa bản ghi
    db.session.delete(record)
    db.session.commit()

    flash("Record has been deleted successfully", "success")
    return redirect(url_for('annotate.dashboard'))

@annotate_bp.route('/download-results/<int:folder_id>')
@login_required
def download_results(folder_id):
    records = StockPrediction.query.filter_by(folder_id=folder_id, user_id=current_user.id).all()
    if not records:
        return "Không tìm thấy kết quả", 404

    # Ghi CSV vào RAM
    output = io.StringIO()
    
    # Get all model columns dynamically
    model_columns = []
    for record in records:
        if hasattr(record, 'predictions_json') and record.predictions_json:
            model_columns = list(json.loads(record.predictions_json).keys())
            break
    
    # Write header
    header = ["Ticker", "Date", "Actual_Close"]
    header.extend([f"{model}_Prediction" for model in model_columns])
    output.write(",".join(header) + "\n")
    
    # Write data
    for r in records:
        row = [r.ticker, r.prediction_date.strftime('%Y-%m-%d'), str(r.actual_price)]
        
        # Add predictions for each model
        if hasattr(r, 'predictions_json') and r.predictions_json:
            predictions = json.loads(r.predictions_json)
            for model in model_columns:
                row.append(str(predictions.get(model, "")))
        else:
            # For backwards compatibility
            row.append(str(r.xgboost_prediction))
            row.append(str(r.lightgbm_prediction))
        
        output.write(",".join(row) + "\n")
    
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        as_attachment=True,
        download_name=f"stock_predictions_folder_{folder_id}.csv",
        mimetype='text/csv'
    )

@annotate_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    # Lấy dữ liệu từ form
    ticker = request.form.get('ticker')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    use_pca_param = request.form.get('use_pca', 'true').lower() == 'true'
    
    if not ticker or not start_date or not end_date:
        return jsonify({"error": "Thiếu thông tin cần thiết"}), 400
    
    # Lấy dữ liệu chứng khoán
    df, error = get_stock_data(ticker, start_date, end_date)
    if error:
        return jsonify({"error": f"Không thể lấy dữ liệu chứng khoán: {error}"}), 400
    
    if df.empty:
        return jsonify({"error": "Không có dữ liệu cho khoảng thời gian đã chọn"}), 400
    
    # Tính toán các chỉ báo kỹ thuật
    df = calculate_technical_indicators(df)
    
    # Tạo target (giá đóng cửa ngày tiếp theo)
    df['Target'] = df['Close'].shift(-1)
    
    # Chia data thành train/test
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()
    
    X_train = train_data.drop('Target', axis=1, errors='ignore').dropna()
    y_train = train_data.loc[X_train.index, 'Target']
    
    X_test = test_data.drop('Target', axis=1, errors='ignore').dropna()
    y_test = test_data.loc[X_test.index, 'Target']
    
    if len(X_train) < 30 or len(X_test) < 10:
        return jsonify({"error": "Không đủ dữ liệu để huấn luyện mô hình"}), 400
    
    # Chuẩn bị dữ liệu
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'MA5', 'MA20', 'MA50', 'RSI',
                'MACD', 'Signal_Line', 'BB_middle', 'BB_upper',
                'BB_lower', 'Volume_MA5', 'Volume_MA20', 'Volatility']
    
    X_train = X_train[features].values
    X_test = X_test[features].values
    
    # Áp dụng PCA nếu được yêu cầu
    if use_pca_param:
        X_train_processed, X_test_processed = apply_pca(X_train, X_test, n_components=0.95)
    else:
        # Chỉ chuẩn hóa dữ liệu
        global scaler
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
        joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))
        
        # Tạo PCA rỗng
        global pca, use_pca
        pca = PCA(n_components=1)
        pca.fit(X_train_processed[:1])  # Chỉ để khởi tạo
        use_pca = False
        joblib.dump(pca, os.path.join(model_path, "pca.pkl"))
    
    # Huấn luyện mô hình
    trained_models = train_models(X_train_processed, y_train)
    
    # Đánh giá mô hình
    results, predictions = evaluate_models(trained_models, X_test_processed, y_test)
    
    # Vẽ đồ thị
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', linewidth=2)
    
    # Chỉ hiển thị 3 mô hình tốt nhất trên đồ thị để tránh rối
    best_models = sorted(results.items(), key=lambda x: x[1]['MSE'])[:3]
    for name, _ in best_models:
        plt.plot(y_test.index, predictions[name], label=f'{name} Prediction', linestyle='--')
    
    plt.title(f'Stock Price Prediction for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Chuyển đồ thị thành base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Lưu thông tin dự đoán vào database
    folder = Folder.query.filter_by(user_id=current_user.id).order_by(Folder.id.desc()).first()
    if not folder:
        folder = Folder(name=f"{ticker}_predictions", user_id=current_user.id)
        db.session.add(folder)
        db.session.commit()
    
    # Lưu kết quả đánh giá mô hình
    for model_name, metrics in results.items():
        model_performance = ModelPerformance(
            ticker=ticker,
            model_name=model_name,
            mse=metrics['MSE'],
            r2_score=metrics['R2'],
            user_id=current_user.id,
            params=json.dumps({
                'n_components': pca.n_components_ if use_pca else 0,
                'use_pca': use_pca
            })
        )
        db.session.add(model_performance)
    
    # Lưu dự đoán cho từng mẫu test
    for i, (date, actual) in enumerate(zip(y_test.index, y_test.values)):
        model_predictions = {name: float(predictions[name][i]) for name in predictions}
        
        new_prediction = StockPrediction(
            ticker=ticker,
            prediction_date=date,
            actual_price=float(actual),
            xgboost_prediction=float(predictions["XGBoost"][i]),  # Giữ lại để tương thích ngược
            lightgbm_prediction=float(predictions["LightGBM"][i]),  # Giữ lại để tương thích ngược
            predictions_json=json.dumps(model_predictions),
            user_id=current_user.id,
            folder_id=folder.id
        )
        db.session.add(new_prediction)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'results': results,
        'plot': plot_url,
        'folder_id': folder.id
    })

@annotate_bp.route('/predict-next', methods=['POST'])
@login_required
def predict_next():
    """Dự đoán giá cổ phiếu trong x ngày tiếp theo"""
    ticker = request.form.get('ticker')
    days = int(request.form.get('days', 5))
    model_name = request.form.get('model', 'XGBoost')  # Mô hình mặc định
    
    if not ticker:
        return jsonify({"error": "Thiếu mã chứng khoán"}), 400
    
    if not models_loaded:
        return jsonify({"error": "Chưa có mô hình, vui lòng huấn luyện mô hình trước"}), 400
    
    if model_name not in models:
        available_models = list(models.keys())
        model_name = available_models[0] if available_models else 'XGBoost'
    
    # Lấy dữ liệu lịch sử
    end_date = datetime.now()
    # Lấy dữ liệu thêm 100 ngày trước cho việc tính các chỉ số kỹ thuật
    start_date = end_date - timedelta(days=100)
    
    df, error = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if error:
        return jsonify({"error": f"Không thể lấy dữ liệu chứng khoán: {error}"}), 400
    
    if df.empty:
        return jsonify({"error": "Không có dữ liệu cho khoảng thời gian đã chọn"}), 400
    
    # Tính toán các chỉ báo kỹ thuật
    df = calculate_technical_indicators(df)
    
    # Xóa các dòng có giá trị NaN
    df = df.dropna()
    
    if len(df) < 30:
        return jsonify({"error": "Không đủ dữ liệu để dự đoán"}), 400
    
    # Dự đoán cho ngày tiếp theo
    next_day_predictions = {}
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'MA5', 'MA20', 'MA50', 'RSI',
                'MACD', 'Signal_Line', 'BB_middle', 'BB_upper',
                'BB_lower', 'Volume_MA5', 'Volume_MA20', 'Volatility']
    
    # Lấy dữ liệu mới nhất
    last_data = df.iloc[-1][features].values.reshape(1, -1)
    
    # Chuẩn hóa dữ liệu
    last_data_scaled = scaler.transform(last_data)
    
    # Áp dụng PCA nếu cần
    if use_pca:
        last_data_scaled = pca.transform(last_data_scaled)
    
    # Dự đoán với các mô hình
    for name, model in models.items():
        next_day_predictions[name] = float(model.predict(last_data_scaled)[0])
    
    # Dự đoán nhiều ngày tiếp theo
    future_dates = [end_date + timedelta(days=i+1) for i in range(days)]
    future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
    
    # Khởi tạo DataFrame với dữ liệu cuối cùng
    future_df = df.tail(1).copy()
    
    # Dictionary lưu dự đoán cho mỗi mô hình
    model_predictions = {name: [pred] for name, pred in next_day_predictions.items()}
    
    for i in range(1, days):
        # Giả định giá mở cửa ngày mai = giá đóng cửa hôm nay (dự đoán của mô hình được chọn)
        next_price = model_predictions[model_name][-1]
        
        future_df.loc[future_df.index[0] + pd.Timedelta(days=i), 'Open'] = next_price
        # Giả định giá cao/thấp là +/- 1% giá mở cửa
        future_df.loc[future_df.index[0] + pd.Timedelta(days=i), 'High'] = next_price * 1.01
        future_df.loc[future_df.index[0] + pd.Timedelta(days=i), 'Low'] = next_price * 0.99
        # Giả định giá đóng cửa = giá mở cửa
        future_df.loc[future_df.index[0] + pd.Timedelta(days=i), 'Close'] = next_price
        # Giả định khối lượng giao dịch 
        future_df.loc[future_df.index[0] + pd.Timedelta(days=i), 'Volume'] = future_df['Volume'].iloc[0]
        
        # Tính lại các chỉ báo kỹ thuật
        future_df = calculate_technical_indicators(future_df)
        
        # Lấy dữ liệu mới nhất
        new_data = future_df.iloc[-1][features].values.reshape(1, -1)
        
        # Chuẩn hóa dữ liệu
        new_data_scaled = scaler.transform(new_data)
        
        # Áp dụng PCA nếu cần
        if use_pca:
            new_data_scaled = pca.transform(new_data_scaled)
        
        # Dự đoán với các mô hình
        for name, model in models.items():
            pred = float(model.predict(new_data_scaled)[0])
            model_predictions[name].append(pred)
    
    # Vẽ đồ thị
    plt.figure(figsize=(12, 6))
    
    # Vẽ dữ liệu lịch sử
    historical_dates = df.index[-30:]  # 30 ngày gần nhất
    plt.plot(historical_dates, df['Close'].iloc[-30:].values, label='Historical', color='blue')
    
    # Vẽ dự đoán cho 3 mô hình tốt nhất
    # Sắp xếp mô hình theo MSE từ các kết quả đã lưu
    best_models = ModelPerformance.query.filter_by(ticker=ticker, user_id=current_user.id).order_by(ModelPerformance.mse).limit(3).all()
    best_model_names = [model.model_name for model in best_models]
    
    # Nếu không có dữ liệu về hiệu suất mô hình, sử dụng các mô hình mặc định
    if not best_model_names:
        best_model_names = ['XGBoost', 'LightGBM', 'RandomForest']
    
    colors = ['red', 'green', 'purple']
    markers = ['o', 'x', 's']
    
    for i, name in enumerate(best_model_names):
        if name in model_predictions:
            plt.plot(future_dates, model_predictions[name], 
                    label=f'{name} Prediction', 
                    linestyle='--', 
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)])
    
    plt.title(f'Future Stock Price Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    # Chuyển đồ thị thành base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Chuẩn bị dữ liệu để trả về
    prediction_results = {}
    for name in model_predictions:
        prediction_results[name] = model_predictions[name]
    
    return jsonify({
        'success': True,
        'dates': future_dates_str,
        'predictions': prediction_results,
        'plot': plot_url
    })