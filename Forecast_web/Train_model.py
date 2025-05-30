import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score  # đo R² score
import joblib
import matplotlib.pyplot as plt  # Import thư viện để vẽ đồ thị

# 1. Đọc dữ liệu
file_path = "dulieu_totnghiep.xlsx"
data = pd.read_excel(file_path, engine='openpyxl')
data.columns = data.columns.str.strip().str.lower()

# 2. Xử lý giá trị 'na'
data = data.applymap(lambda x: np.nan if isinstance(x, str) and x.strip().lower() == 'na' else x)
data.dropna(inplace=True)  # Xóa các dòng có giá trị thiếu

# 3. Ép kiểu các cột số về dạng số
for col in data.select_dtypes(include=['number']).columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 4. Xác định các cột mục tiêu và đặc trưng
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

# 7. Huấn luyện mô hình Random Forest (Hồi Quy)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# 9. Tính R² score và in ra dưới dạng phần trăm
r2 = r2_score(y_test, y_pred)
accuracy_percent = r2 * 100
print(f"Độ chính xác (R²) của mô hình: {accuracy_percent:.2f}%")

# Thêm đồ thị hiển thị % chính xác của mô hình
fig, ax = plt.subplots()
ax.bar(["R² Score"], [accuracy_percent], color='skyblue')
ax.set_ylim(0, 100)
ax.set_ylabel('Phần trăm (%)')
ax.set_title(f"Độ chính xác (R²) của mô hình: {accuracy_percent:.2f}%")
plt.show()

# Đảm bảo rằng đồ thị được vẽ đầy đủ trước khi lưu
fig.tight_layout()  # Đảm bảo rằng layout không bị cắt
fig.savefig('accuracy_graph.png') 

# Đóng figure sau khi lưu
plt.close(fig)

# 10. Lưu mô hình và danh sách các cột đặc trưng (sau khi one-hot encoding)
joblib.dump(model, "employment_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
print("Huấn luyện xong và đã lưu mô hình!")



