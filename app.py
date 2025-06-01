from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from stock_prediction import prepare_features, train_models, evaluate_models, generate_sample_data, apply_pca
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    try:
        # Lấy dữ liệu
        print(f"Đang lấy dữ liệu cho {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"Không tìm thấy dữ liệu cho {ticker}, sử dụng dữ liệu mẫu...")
            df = generate_sample_data(ticker, start_date, end_date)
            
        # Chuẩn bị dữ liệu
        X, y = prepare_features(df)
        
        if len(X) < 30:
            return jsonify({'error': 'Không đủ dữ liệu để huấn luyện mô hình. Vui lòng chọn khoảng thời gian dài hơn.'})
        
        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Áp dụng PCA
        print("Áp dụng PCA...")
        X_train_pca, X_test_pca, pca, scaler = apply_pca(X_train, X_test, n_components=0.95)
        
        # Huấn luyện mô hình với dữ liệu gốc
        print("Huấn luyện các mô hình...")
        models = train_models(X_train, y_train)
        
        # Đánh giá mô hình
        results, predictions = evaluate_models(models, X_test, y_test)
        
        # Tạo đồ thị
        plt.figure(figsize=(12, 6))
        
        # Lấy ngày tháng từ index của y_test
        dates = y_test.index
        
        # Vẽ đường giá thực tế
        plt.plot(dates, y_test.values, label='Thực tế', linewidth=2)
        
        # Hiển thị 3 mô hình tốt nhất trên đồ thị
        best_models = sorted(results.items(), key=lambda x: x[1]['MSE'])[:3]
        for name, _ in best_models:
            plt.plot(dates, predictions[name], label=f'Dự đoán {name}', linestyle='--')
        
        plt.title(f'Dự đoán giá cổ phiếu cho {ticker}')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá ($)')
        
        # Định dạng trục thời gian
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()  # Xoay nhãn ngày tháng để dễ đọc
        
        plt.legend()
        plt.grid(True)
        
        # Chuyển đồ thị thành base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Chuyển định dạng kết quả để trả về cho client
        formatted_results = {}
        for model_name, metrics in results.items():
            formatted_results[model_name] = {
                'MSE': float(metrics['MSE']),
                'R2': float(metrics['R2'])
            }
        
        # Lấy giá dự đoán cho ngày tiếp theo từ mô hình tốt nhất
        best_model_name = min(results.items(), key=lambda x: x[1]['MSE'])[0]
        next_day_prediction = models[best_model_name].predict(X_test[-1:])
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'plot': plot_url,
            'next_day_prediction': float(next_day_prediction[0]),  # Giá dự đoán cho ngày tiếp theo
            'current_price': float(df['Close'].iloc[-1])  # Giá hiện tại
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 