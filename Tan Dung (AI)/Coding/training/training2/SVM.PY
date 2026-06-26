import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Đọc dữ liệu
data = pd.read_csv(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\pifr_test_dataset.csv")

# Tách đặc trưng (X) và nhãn (y)
X = data.drop("pose_class", axis=1)
y = data["pose_class"]

# 2. Chia tập dữ liệu (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Chuẩn hóa dữ liệu (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lưu scaler lại để dùng trong quá trình test real-time sau này
joblib.dump(scaler, "scaler.pkl")

# 4. Huấn luyện mô hình SVM
print("🚀 Đang bắt đầu huấn luyện mô hình SVM...")
# C và gamma là các tham số quan trọng của SVM (tối ưu hóa theo thực tế)
clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
clf.fit(X_train_scaled, y_train)

# 5. Đánh giá mô hình
y_pred = clf.predict(X_test_scaled)
print("\n✅ Kết quả huấn luyện:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Lưu mô hình
joblib.dump(clf, "fall_detection_svm.pkl")
print("\n💾 Đã lưu mô hình tại 'fall_detection_svm.pkl'")