import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def get_stock_data(ticker, start_date, end_date):
    """
    Lấy dữ liệu chứng khoán từ yfinance
    """
    try:
        # Cấu hình yfinance nếu cần
        print(f"Đang tải dữ liệu cho {ticker}...")
        
        # Tạo đối tượng Ticker
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"Warning: Không có dữ liệu cho {ticker} trong khoảng thời gian đã chọn")
            return pd.DataFrame()
            
        # In thông tin cơ bản để kiểm tra
        print(f"Đã lấy {len(df)} dòng dữ liệu cho {ticker}")
        if not df.empty:
            print(f"Phạm vi thời gian: {df.index.min()} đến {df.index.max()}")
        
        return df
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu cho {ticker}: {str(e)}")
        # Trả về DataFrame rỗng trong trường hợp lỗi
        return pd.DataFrame()

def prepare_features(df, window_size=5):
    """
    Chuẩn bị features cho mô hình
    """
    # Tính toán các chỉ báo kỹ thuật
    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
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
    
    # Tạo target (giá đóng cửa ngày tiếp theo)
    df['Target'] = df['Close'].shift(-1)
    
    # Xóa các dòng có giá trị NaN
    df = df.dropna()
    
    # Chọn features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'Returns', 'MA5', 'MA20', 'MA50', 'RSI', 
                'MACD', 'Signal_Line', 'BB_middle', 'BB_upper', 'BB_lower', 
                'Volume_MA5', 'Volume_MA20', 'Volatility']
    
    return df[features], df['Target']

def calculate_rsi(prices, period=14):
    """
    Tính toán chỉ báo RSI
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def apply_pca(X_train, X_test, n_components=0.95):
    """
    Áp dụng PCA để giảm chiều dữ liệu
    n_components: số thành phần chính muốn giữ lại hoặc tỷ lệ phương sai (0-1)
    """
    # Chuẩn hóa dữ liệu trước khi áp dụng PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Áp dụng PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Hiển thị thông tin về số thành phần chính và tỷ lệ phương sai giải thích được
    n_components = pca.n_components_
    explained_variance_ratio = pca.explained_variance_ratio_
    total_variance = sum(explained_variance_ratio)
    
    print(f"Số thành phần chính: {n_components}")
    print(f"Tổng phương sai giải thích được: {total_variance:.4f} ({total_variance*100:.2f}%)")
    
    # Trực quan hóa phương sai giải thích được
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), 'r-')
    plt.xlabel('Thành phần chính')
    plt.ylabel('Tỷ lệ phương sai giải thích được')
    plt.title('PCA - Phương sai giải thích được')
    plt.grid(True)
    plt.show()
    
    return X_train_pca, X_test_pca, pca, scaler

def train_models(X_train, y_train):
    """
    Huấn luyện các mô hình
    """
    models = {}
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    
    # K-Nearest Neighbors
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    models['KNN'] = knn_model
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    # Support Vector Regression
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr_model.fit(X_train, y_train)
    models['SVR'] = svr_model
    
    # Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    models['Ridge'] = ridge_model
    
    return models

def evaluate_models(models, X_test, y_test):
    """
    Đánh giá các mô hình
    """
    results = {}
    predictions = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
        predictions[name] = y_pred
    
    # Tạo bảng so sánh các mô hình
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'MSE': [results[model]['MSE'] for model in results],
        'R2': [results[model]['R2'] for model in results]
    })
    
    comparison = comparison.sort_values('MSE')
    print("\nSo sánh hiệu suất các mô hình:")
    print(comparison)
    
    return results, predictions

def plot_predictions(y_true, predictions, title="Predictions Comparison"):
    """
    Vẽ đồ thị so sánh giá trị thực tế và dự đoán từ nhiều mô hình
    """
    plt.figure(figsize=(14, 7))
    
    # Vẽ giá trị thực tế
    plt.plot(y_true.values, label='Actual', linewidth=2)
    
    # Vẽ giá trị dự đoán từ các mô hình
    for name, pred in predictions.items():
        plt.plot(pred, label=f'{name} Prediction', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_models(models, pca, scaler, save_dir="models"):
    """
    Lưu các mô hình, PCA và scaler
    """
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Lưu các mô hình
    for name, model in models.items():
        joblib.dump(model, os.path.join(save_dir, f"{name}_model.pkl"))
    
    # Lưu PCA và scaler
    joblib.dump(pca, os.path.join(save_dir, "pca.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    
    print(f"Đã lưu tất cả mô hình vào thư mục {save_dir}")

def generate_sample_data(ticker, start_date, end_date):
    """
    Tạo dữ liệu mẫu trong trường hợp không lấy được dữ liệu từ yfinance
    """
    print(f"Tạo dữ liệu mẫu cho {ticker}...")
    
    # Chuyển đổi chuỗi ngày tháng thành datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Tạo dãy ngày cho dữ liệu (chỉ ngày làm việc)
    date_range = pd.date_range(start=start, end=end, freq='B')
    
    # Tạo giá ban đầu
    initial_price = 100
    
    # Tạo dữ liệu ngẫu nhiên
    np.random.seed(42)  # Để kết quả có thể tái tạo
    
    # Tạo độ biến động ngẫu nhiên cho các giá trị
    price_changes = np.random.normal(0, 1, len(date_range)) / 100
    
    # Tích lũy các thay đổi để tạo xu hướng
    cumulative_changes = np.cumprod(1 + price_changes)
    
    # Tạo giá đóng cửa
    close_prices = initial_price * cumulative_changes
    
    # Tạo các giá trị khác dựa trên giá đóng cửa
    data = {
        'Open': close_prices * np.random.uniform(0.99, 1.01, len(date_range)),
        'High': close_prices * np.random.uniform(1.01, 1.03, len(date_range)),
        'Low': close_prices * np.random.uniform(0.97, 0.99, len(date_range)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, len(date_range))
    }
    
    # Tạo DataFrame
    df = pd.DataFrame(data, index=date_range)
    
    # Đảm bảo High luôn >= Open, Close và Low luôn <= Open, Close
    df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    print(f"Đã tạo {len(df)} dòng dữ liệu mẫu cho {ticker}")
    print(f"Phạm vi thời gian: {df.index.min()} đến {df.index.max()}")
    
    return df

def check_data_quality(df, file_prefix):
    """
    Kiểm tra chất lượng dữ liệu và xuất ra các dòng có vấn đề
    """
    issues = []
    
    # Kiểm tra dữ liệu trùng lặp
    duplicates = df[df.index.duplicated(keep=False)]
    if not duplicates.empty:
        issues.append({
            'issue_type': 'Duplicates',
            'data': duplicates,
            'description': 'Các dòng có ngày trùng lặp'
        })
    
    # Kiểm tra dữ liệu thiếu (NaN)
    missing_data = df[df.isnull().any(axis=1)]
    if not missing_data.empty:
        issues.append({
            'issue_type': 'Missing',
            'data': missing_data,
            'description': 'Các dòng có dữ liệu thiếu'
        })
    
    # Kiểm tra giá trị bất thường (ví dụ: giá âm)
    price_columns = ['Open', 'High', 'Low', 'Close']
    invalid_prices = df[df[price_columns].le(0).any(axis=1)]
    if not invalid_prices.empty:
        issues.append({
            'issue_type': 'Invalid_Prices',
            'data': invalid_prices,
            'description': 'Các dòng có giá <= 0'
        })
    
    # Kiểm tra volume bất thường
    invalid_volume = df[df['Volume'] <= 0]
    if not invalid_volume.empty:
        issues.append({
            'issue_type': 'Invalid_Volume',
            'data': invalid_volume,
            'description': 'Các dòng có volume <= 0'
        })
    
    # Kiểm tra High < Low
    invalid_hl = df[df['High'] < df['Low']]
    if not invalid_hl.empty:
        issues.append({
            'issue_type': 'Invalid_HL',
            'data': invalid_hl,
            'description': 'Các dòng có High < Low'
        })
    
    # Nếu có vấn đề, xuất ra file
    if issues:
        # Tạo thư mục data_issues nếu chưa tồn tại
        if not os.path.exists('data_issues'):
            os.makedirs('data_issues')
        
        # Xuất từng loại vấn đề ra file riêng
        for issue in issues:
            issue_file = f'data_issues/{file_prefix}_{issue["issue_type"]}.csv'
            issue['data'].to_csv(issue_file)
            print(f"Đã lưu {issue['description']} vào: {issue_file}")
        
        # Tạo file tổng hợp
        summary_file = f'data_issues/{file_prefix}_quality_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Báo cáo chất lượng dữ liệu cho {file_prefix}\n")
            f.write(f"Thời gian tạo: {pd.Timestamp.now()}\n\n")
            
            for issue in issues:
                f.write(f"{issue['description']}:\n")
                f.write(f"Số lượng dòng: {len(issue['data'])}\n")
                f.write("-------------------\n")
        
        print(f"Đã lưu báo cáo tổng hợp vào: {summary_file}")
        return False
    
    return True

def save_to_csv(df, features, target, file_prefix):
    """
    Lưu dữ liệu gốc và đặc trưng ra file CSV
    """
    # Kiểm tra chất lượng dữ liệu trước khi lưu
    data_quality_ok = check_data_quality(df, file_prefix)
    if not data_quality_ok:
        print("Cảnh báo: Dữ liệu có một số vấn đề, vui lòng kiểm tra thư mục data_issues")
    
    # Tạo thư mục data nếu chưa tồn tại
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Lưu dữ liệu gốc
    raw_data_file = f'data/{file_prefix}_raw_data.csv'
    df.to_csv(raw_data_file)
    print(f"Đã lưu dữ liệu gốc vào: {raw_data_file}")
    
    # Lưu đặc trưng và target
    features_data = pd.DataFrame(features)
    features_data['Target'] = target
    features_file = f'data/{file_prefix}_features.csv'
    features_data.to_csv(features_file)
    print(f"Đã lưu đặc trưng vào: {features_file}")

def main():
    # Danh sách các mã chứng khoán phổ biến để thử
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    start_date = "2022-01-01"
    end_date = "2025-5-22"
    
    # Biến để theo dõi xem đã lấy được dữ liệu thật chưa
    successful_tickers = []
    
    # Thử với từng mã chứng khoán
    for ticker in tickers:
        try:
            print(f"\n{'='*50}")
            print(f"Đang xử lý mã chứng khoán {ticker}")
            print(f"{'='*50}")
            print(f"Đang lấy dữ liệu từ {start_date} đến {end_date}...")
            
            df = get_stock_data(ticker, start_date, end_date)
            
            if df.empty:
                print(f"Không có dữ liệu cho {ticker}. Chuyển sang mã tiếp theo...")
                continue
            
            # Chuẩn bị dữ liệu
            print("Đang chuẩn bị dữ liệu và tính toán các chỉ báo kỹ thuật...")
            X, y = prepare_features(df)
            
            # Lưu dữ liệu và đặc trưng ra file CSV
            save_to_csv(df, X, y, ticker)
            
            if X.empty or len(X) < 30:  # Kiểm tra có đủ dữ liệu không
                print(f"Không đủ dữ liệu cho {ticker} sau khi chuẩn bị. Chuyển sang mã tiếp theo...")
                continue
            
            # Hiển thị thông tin về các features đã tạo
            print(f"\nDanh sách các features ({len(X.columns)}):")
            for i, col in enumerate(X.columns):
                print(f"{i+1}. {col}")
            
            # Chia train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            print(f"\nTập train: {X_train.shape}, Tập test: {X_test.shape}")
            
            # Áp dụng PCA
            print("\nĐang áp dụng PCA để giảm chiều dữ liệu...")
            X_train_pca, X_test_pca, pca, pca_scaler = apply_pca(X_train, X_test, n_components=0.95)
            
            # Huấn luyện mô hình với dữ liệu gốc
            print("\nĐang huấn luyện các mô hình với dữ liệu gốc...")
            original_models = train_models(X_train, y_train)
            
            # Đánh giá mô hình với dữ liệu gốc
            print("\nKết quả đánh giá các mô hình với dữ liệu gốc:")
            original_results, original_predictions = evaluate_models(original_models, X_test, y_test)
            
            # Huấn luyện mô hình với dữ liệu PCA
            print("\nĐang huấn luyện các mô hình với dữ liệu sau khi áp dụng PCA...")
            pca_models = train_models(X_train_pca, y_train)
            
            # Đánh giá mô hình với dữ liệu PCA
            print("\nKết quả đánh giá các mô hình với dữ liệu sau khi áp dụng PCA:")
            pca_results, pca_predictions = evaluate_models(pca_models, X_test_pca, y_test)
            
            # So sánh hiệu suất giữa dữ liệu gốc và dữ liệu PCA
            print("\nSo sánh hiệu suất giữa dữ liệu gốc và dữ liệu PCA:")
            comparison = []
            
            for model_name in original_results.keys():
                original_mse = original_results[model_name]['MSE']
                original_r2 = original_results[model_name]['R2']
                pca_mse = pca_results[model_name]['MSE']
                pca_r2 = pca_results[model_name]['R2']
                
                comparison.append({
                    'Model': model_name,
                    'Original MSE': original_mse,
                    'PCA MSE': pca_mse,
                    'MSE Diff (%)': (pca_mse - original_mse) / original_mse * 100,
                    'Original R2': original_r2,
                    'PCA R2': pca_r2,
                    'R2 Diff': pca_r2 - original_r2
                })
            
            comparison_df = pd.DataFrame(comparison)
            comparison_df = comparison_df.sort_values('Original MSE')
            print(comparison_df)
            
            # Vẽ đồ thị dự đoán với dữ liệu gốc
            print("\nĐồ thị dự đoán với dữ liệu gốc:")
            plot_predictions(y_test, original_predictions, f"{ticker} Stock Price Predictions (Original Data)")
            
            # Vẽ đồ thị dự đoán với dữ liệu PCA
            print("\nĐồ thị dự đoán với dữ liệu PCA:")
            plot_predictions(y_test, pca_predictions, f"{ticker} Stock Price Predictions (PCA Data)")
            
            # Lưu mô hình tốt nhất
            best_model_name = comparison_df.iloc[0]['Model']
            print(f"\nMô hình tốt nhất cho {ticker}: {best_model_name}")
            
            # Chọn bộ mô hình tốt nhất (dữ liệu gốc hoặc PCA)
            use_pca = comparison_df['PCA MSE'].min() < comparison_df['Original MSE'].min()
            
            if use_pca:
                print(f"Sử dụng mô hình với dữ liệu PCA cho {ticker}")
                best_models = pca_models
                save_models(best_models, pca, pca_scaler, save_dir=f"models/{ticker}")
            else:
                print(f"Sử dụng mô hình với dữ liệu gốc cho {ticker}")
                best_models = original_models
                # Tạo PCA và scaler rỗng để giữ cấu trúc nhất quán
                dummy_pca = PCA(n_components=1)
                dummy_scaler = StandardScaler()
                save_models(best_models, dummy_pca, dummy_scaler, save_dir=f"models/{ticker}")
            
            successful_tickers.append(ticker)
            print(f"\nHoàn thành phân tích cho mã {ticker}!")
            
        except Exception as e:
            print(f"Lỗi khi xử lý mã {ticker}: {str(e)}")
            print("Chuyển sang mã tiếp theo...")
    
    # In tổng kết
    print("\n" + "="*50)
    print("KẾT QUẢ TỔNG HỢP:")
    print(f"Tổng số mã đã xử lý thành công: {len(successful_tickers)}")
    if successful_tickers:
        print("Các mã thành công:", ", ".join(successful_tickers))
    else:
        print("Không có mã nào được xử lý thành công.")
        # Thử tạo dữ liệu mẫu
        try:
            print("\nThử tạo dữ liệu mẫu để demo...")
            ticker = "SAMPLE"
            df = generate_sample_data(ticker, start_date, end_date)
            
            # Chuẩn bị dữ liệu
            print("Đang chuẩn bị dữ liệu và tính toán các chỉ báo kỹ thuật...")
            X, y = prepare_features(df)
            
            # Hiển thị thông tin về các features đã tạo
            print(f"\nDanh sách các features ({len(X.columns)}):")
            for i, col in enumerate(X.columns):
                print(f"{i+1}. {col}")
            
            # Chia train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            print(f"\nTập train: {X_train.shape}, Tập test: {X_test.shape}")
            
            # Áp dụng PCA
            print("\nĐang áp dụng PCA để giảm chiều dữ liệu...")
            X_train_pca, X_test_pca, pca, pca_scaler = apply_pca(X_train, X_test, n_components=0.95)
            
            # Huấn luyện mô hình với dữ liệu gốc
            print("\nĐang huấn luyện các mô hình với dữ liệu gốc...")
            original_models = train_models(X_train, y_train)
            
            # Đánh giá mô hình với dữ liệu gốc
            print("\nKết quả đánh giá các mô hình với dữ liệu gốc:")
            original_results, original_predictions = evaluate_models(original_models, X_test, y_test)
            
            # Huấn luyện mô hình với dữ liệu PCA
            print("\nĐang huấn luyện các mô hình với dữ liệu sau khi áp dụng PCA...")
            pca_models = train_models(X_train_pca, y_train)
            
            # Đánh giá mô hình với dữ liệu PCA
            print("\nKết quả đánh giá các mô hình với dữ liệu sau khi áp dụng PCA:")
            pca_results, pca_predictions = evaluate_models(pca_models, X_test_pca, y_test)
            
            # So sánh hiệu suất giữa dữ liệu gốc và dữ liệu PCA
            print("\nSo sánh hiệu suất giữa dữ liệu gốc và dữ liệu PCA:")
            comparison = []
            
            for model_name in original_results.keys():
                original_mse = original_results[model_name]['MSE']
                original_r2 = original_results[model_name]['R2']
                pca_mse = pca_results[model_name]['MSE']
                pca_r2 = pca_results[model_name]['R2']
                
                comparison.append({
                    'Model': model_name,
                    'Original MSE': original_mse,
                    'PCA MSE': pca_mse,
                    'MSE Diff (%)': (pca_mse - original_mse) / original_mse * 100,
                    'Original R2': original_r2,
                    'PCA R2': pca_r2,
                    'R2 Diff': pca_r2 - original_r2
                })
            
            comparison_df = pd.DataFrame(comparison)
            comparison_df = comparison_df.sort_values('Original MSE')
            print(comparison_df)
            
            # Vẽ đồ thị dự đoán với dữ liệu gốc
            print("\nĐồ thị dự đoán với dữ liệu gốc:")
            plot_predictions(y_test, original_predictions, f"{ticker} Stock Price Predictions (Original Data)")
            
            # Vẽ đồ thị dự đoán với dữ liệu PCA
            print("\nĐồ thị dự đoán với dữ liệu PCA:")
            plot_predictions(y_test, pca_predictions, f"{ticker} Stock Price Predictions (PCA Data)")
            
            # Lưu mô hình tốt nhất
            best_model_name = comparison_df.iloc[0]['Model']
            print(f"\nMô hình tốt nhất cho {ticker}: {best_model_name}")
            
            # Chọn bộ mô hình tốt nhất (dữ liệu gốc hoặc PCA)
            use_pca = comparison_df['PCA MSE'].min() < comparison_df['Original MSE'].min()
            
            if use_pca:
                print(f"Sử dụng mô hình với dữ liệu PCA cho {ticker}")
                best_models = pca_models
                save_models(best_models, pca, pca_scaler, save_dir=f"models/{ticker}")
            else:
                print(f"Sử dụng mô hình với dữ liệu gốc cho {ticker}")
                best_models = original_models
                # Tạo PCA và scaler rỗng để giữ cấu trúc nhất quán
                dummy_pca = PCA(n_components=1)
                dummy_scaler = StandardScaler()
                save_models(best_models, dummy_pca, dummy_scaler, save_dir=f"models/{ticker}")
            
            print(f"\nHoàn thành phân tích với dữ liệu mẫu!")
            
        except Exception as e:
            print(f"Lỗi khi xử lý dữ liệu mẫu: {str(e)}")
            print("Không thể chạy chương trình. Vui lòng kiểm tra lại hoặc thử lại sau.")

if __name__ == "__main__":
    main() 