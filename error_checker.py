import numpy as np
import pandas as pd

def safe_iloc(obj, index):
    """Hàm an toàn để thay thế .iloc, kiểm tra loại dữ liệu trước khi sử dụng"""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        # Nếu là DataFrame hoặc Series, sử dụng .iloc
        if isinstance(index, slice):
            return obj.iloc[index]
        elif isinstance(index, list):
            return obj.iloc[index]
        else:
            return obj.iloc[[index]] if index >= 0 else obj.iloc[[len(obj) + index]]
    elif isinstance(obj, np.ndarray):
        # Nếu là numpy array, sử dụng index thông thường
        if isinstance(index, slice):
            return obj[index]
        elif isinstance(index, list):
            return np.array([obj[i] for i in index])
        else:
            return obj[index:index+1] if index >= 0 else obj[len(obj) + index:len(obj) + index + 1]
    else:
        # Nếu là kiểu khác, thử sử dụng index thông thường
        try:
            return obj[index]
        except Exception as e:
            raise ValueError(f"Không thể sử dụng .iloc hoặc [] trên kiểu {type(obj)}: {str(e)}")

# Sử dụng trong satellite_annotation/routes/annotate.py

print("=== KIỂM TRA SAFE_ILOC ===")

# Kiểm tra với DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print("DataFrame original:", df)
print("DataFrame safe_iloc[-1]:", safe_iloc(df, -1))
print("DataFrame safe_iloc[[-1]]:", safe_iloc(df, [-1]))
print("DataFrame safe_iloc[0:2]:", safe_iloc(df, slice(0, 2)))

# Kiểm tra với Series
s = pd.Series([10, 20, 30])
print("\nSeries original:", s)
print("Series safe_iloc[-1]:", safe_iloc(s, -1))
print("Series safe_iloc[[-1]]:", safe_iloc(s, [-1]))
print("Series safe_iloc[0:2]:", safe_iloc(s, slice(0, 2)))

# Kiểm tra với numpy array
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("\nNumpy array original:", arr)
print("Numpy array safe_iloc[-1]:", safe_iloc(arr, -1))
print("Numpy array safe_iloc[0:2]:", safe_iloc(arr, slice(0, 2)))

# Code mẫu của hàm predict_next
print("\n=== MẪU CODE CHO PREDICT_NEXT ===")
features = ['A', 'B', 'C']

# Giả lập df có thể là DataFrame
df_sample = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
print("Trường hợp df là DataFrame:")
last_data = safe_iloc(df_sample[features], [-1])
print("last_data =", last_data)
last_data_values = last_data.values if isinstance(last_data, (pd.DataFrame, pd.Series)) else last_data
print("last_data_values =", last_data_values)

# Giả lập df có thể là numpy array
np_sample = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nTrường hợp df là numpy array:")
last_data = safe_iloc(np_sample, [-1])
print("last_data =", last_data)
last_data_values = last_data.values if isinstance(last_data, (pd.DataFrame, pd.Series)) else last_data
print("last_data_values =", last_data_values)

print("\n=== CÁCH SỬA LỖI TRONG PREDICT_NEXT ===")
print("""
# Cách sửa trong annotate.py:

# 1. Thêm hàm safe_iloc vào đầu file

def safe_iloc(obj, index):
    \"\"\"Hàm an toàn để thay thế .iloc, kiểm tra loại dữ liệu trước khi sử dụng\"\"\"
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        # Nếu là DataFrame hoặc Series, sử dụng .iloc
        if isinstance(index, slice):
            return obj.iloc[index]
        elif isinstance(index, list):
            return obj.iloc[index]
        else:
            return obj.iloc[[index]] if index >= 0 else obj.iloc[[len(obj) + index]]
    elif isinstance(obj, np.ndarray):
        # Nếu là numpy array, sử dụng index thông thường
        if isinstance(index, slice):
            return obj[index]
        elif isinstance(index, list):
            return np.array([obj[i] for i in index])
        else:
            return obj[index:index+1] if index >= 0 else obj[len(obj) + index:len(obj) + index + 1]
    else:
        # Nếu là kiểu khác, thử sử dụng index thông thường
        try:
            return obj[index]
        except Exception as e:
            raise ValueError(f"Không thể sử dụng .iloc hoặc [] trên kiểu {type(obj)}: {str(e)}")

# 2. Thay thế tất cả .iloc trong predict_next như sau:

# Thay vì:
# last_data = df[features].iloc[[-1]]
# last_data_scaled = scaler.transform(last_data)

# Sử dụng:
last_data = safe_iloc(df[features], [-1])
last_data_values = last_data.values if isinstance(last_data, (pd.DataFrame, pd.Series)) else last_data
last_data_scaled = scaler.transform(last_data_values)

# Thay vì:
# volume_value = future_df['Volume'].iloc[-1]

# Sử dụng:
volume_value = safe_iloc(future_df['Volume'], -1)

# Tương tự cho tất cả các vị trí khác dùng .iloc
""") 