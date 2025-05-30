import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

# 1. Đọc dữ liệu
file_path = "dulieu_totnghiep.xlsx"
data = pd.read_excel(file_path, engine='openpyxl')
data.columns = data.columns.str.strip().str.lower()

# 2. Xử lý giá trị 'na'
data.replace('na', np.nan, inplace=True)
data.dropna(inplace=True)

# 3. Chuyển đổi kiểu dữ liệu số
for col in data.select_dtypes(include=['number']).columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 4. Xác định cột đặc trưng và cột mục tiêu
target_cols = ["employment_rate_overall", "gross_monthly_mean"]
X = data.drop(columns=target_cols)
y = data[target_cols]

# 5. Mã hóa one-hot cho các cột phân loại
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 6. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Huấn luyện mô hình XGBoost đa đầu ra
base_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train_scaled, y_train)

# 9. Đánh giá mô hình bằng R² tổng thể
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
accuracy_percent = r2 * 100
print(f"Độ chính xác (R²) tổng thể của mô hình: {accuracy_percent:.2f}%")

# 10. Hiển thị đồ thị độ chính xác
fig, ax = plt.subplots()
ax.bar(["R² Tổng thể"], [accuracy_percent], color='skyblue')
ax.set_ylim(0, 100)
ax.set_ylabel('Phần trăm (%)')
ax.set_title(f"Độ chính xác (R²) tổng thể: {accuracy_percent:.2f}%")
plt.show()
# Đảm bảo rằng đồ thị được vẽ đầy đủ trước khi lưu
fig.tight_layout()  # Đảm bảo rằng layout không bị cắt
fig.savefig('XGB_accuracy_graph.png')  # Lưu ảnh trong thư mục 'static'

# Đóng figure sau khi lưu
plt.close(fig)

# 11. Lưu mô hình, scaler và danh sách cột đặc trưng
folder_path = "XGBoost Regressor"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

joblib.dump(model, os.path.join(folder_path, "xgboost_model.pkl"))
joblib.dump(scaler, os.path.join(folder_path, "scaler.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(folder_path, "xgboost_feature_columns.pkl"))

print(" Mô hình XGBoost đã được huấn luyện và lưu thành công!")
