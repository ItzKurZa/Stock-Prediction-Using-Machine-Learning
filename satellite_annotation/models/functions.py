import os
import pandas as pd
import numpy as np
import joblib

def prepare_features(df):
    """Chuẩn bị dữ liệu cho dự đoán"""
    # Tính toán các chỉ báo kỹ thuật nếu chưa có
    if 'Returns' not in df.columns:
        df = calculate_technical_indicators(df)
    
    # Xóa các dòng có giá trị NaN
    df = df.dropna()
    
    # Chọn features
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'MA5', 'MA20', 'MA50', 'RSI',
                'MACD', 'Signal_Line', 'BB_middle', 'BB_upper',
                'BB_lower', 'Volume_MA5', 'Volume_MA20', 'Volatility']
    
    return df[features]

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

def load_model(timeframe, model_path):
    """Load mô hình đã lưu cho khung thời gian cụ thể"""
    try:
        # Đường dẫn thư mục mô hình
        model_dir = os.path.join(model_path, f"models_{timeframe}")
        
        if not os.path.exists(model_dir):
            return None, None, None
        
        # Load mô hình tốt nhất (XGBoost)
        model_file = os.path.join(model_dir, "XGBoost_model.pkl")
        if not os.path.exists(model_file):
            return None, None, None
            
        model = joblib.load(model_file)
        
        # Load PCA và scaler
        pca_file = os.path.join(model_dir, "pca.pkl")
        scaler_file = os.path.join(model_dir, "scaler.pkl")
        
        if os.path.exists(pca_file) and os.path.exists(scaler_file):
            pca = joblib.load(pca_file)
            scaler = joblib.load(scaler_file)
            return model, pca, scaler
        else:
            return model, None, None
            
    except Exception as e:
        print(f"Lỗi khi load mô hình cho {timeframe}: {str(e)}")
        return None, None, None

def predict_future_prices(model, X, steps):
    """Dự đoán giá trong tương lai"""
    predictions = []
    
    # Kiểm tra X có phải là DataFrame không
    is_dataframe = isinstance(X, pd.DataFrame)
    
    # Nếu X là DataFrame, chuyển thành numpy array
    if is_dataframe:
        X_array = X.values
    else:
        X_array = X.copy()
    
    # Lấy dữ liệu cuối cùng
    current_data = X_array[-1:].copy()
    
    # Dự đoán nhiều bước
    for _ in range(steps):
        # Dự đoán giá tiếp theo
        next_price = float(model.predict(current_data)[0])
        predictions.append(next_price)
        
        # Cập nhật dữ liệu mới (giả định giá không đổi)
        # Chỉ cập nhật giá đóng cửa (index 3)
        if len(current_data[0]) > 3:
            current_data[0, 3] = next_price
    
    return predictions 