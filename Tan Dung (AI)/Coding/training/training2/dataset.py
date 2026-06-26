import matplotlib
matplotlib.use('Agg') # Tránh treo luồng nạp thư viện đồ họa

import cv2
import os
import random
import glob
import math
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Khởi tạo mô hình YOLOv11-pose
model = YOLO("yolo11n-pose.pt")

# =========================================================================
# PHẦN 1: CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG HÌNH HỌC (AN TOÀN KHÔNG SẬP)
# =========================================================================
def calculate_angle_3points(A, B, C):
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot_product = BA[0]*BC[0] + BA[1]*BC[1]
    norm_BA = math.sqrt(BA[0]**2 + BA[1]**2)
    norm_BC = math.sqrt(BC[0]**2 + BA[1]**2)
    if norm_BA == 0 or norm_BC == 0: return 0
    cos_angle = dot_product / (norm_BA * norm_BC)
    cos_angle = max(-1.0, min(1.0, cos_angle)) 
    return math.degrees(math.acos(cos_angle))

def calculate_angle_with_vertical(A, B):
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    norm = math.sqrt(dx**2 + dy**2)
    if norm == 0: return 0
    return math.degrees(math.acos(dy / norm))

def calculate_angle_with_horizontal(A, B):
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    norm = math.sqrt(dx**2 + dy**2)
    if norm == 0: return 0
    return math.degrees(math.acos(dx / norm))

def extract_biomechanic_features(image):
    results = model(image, verbose=False)
    if len(results) == 0 or results[0].keypoints is None or results[0].keypoints.xy.shape[0] == 0:
        return None # Trả về None nếu KHÔNG CÓ NGƯỜI (Khung hình trống sẽ bị loại bỏ hoàn toàn tại đây)
        
    kp = results[0].keypoints.xy[0].cpu().numpy() 
    if len(kp) < 17 or np.all(kp == 0): 
        return None 

    nose = kp[0]
    l_shoulder, r_shoulder = kp[5], kp[6]
    l_hip, r_hip = kp[11], kp[12]
    l_knee, r_knee = kp[13], kp[14]
    l_ankle, r_ankle = kp[15], kp[16]

    mid_hip = ((l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2)
    mid_ankle = ((l_ankle[0] + r_ankle[0])/2, (l_ankle[1] + r_ankle[1])/2)

    com_x = np.mean(kp[:, 0])
    com_y = np.mean(kp[:, 1])
    shoulder_nose_angle = calculate_angle_3points(l_shoulder, nose, r_shoulder)
    torso_angle = calculate_angle_with_vertical(nose, mid_hip)
    hip_angle = calculate_angle_with_horizontal(l_hip, r_hip)
    shoulder_angle = calculate_angle_with_horizontal(l_shoulder, r_shoulder)
    left_leg_angle = calculate_angle_3points(l_hip, l_knee, l_ankle)
    right_leg_angle = calculate_angle_3points(r_hip, r_knee, r_ankle)
    nose_to_ankle_angle = calculate_angle_with_vertical(nose, mid_ankle)
    
    return [com_x, com_y, shoulder_nose_angle, torso_angle, hip_angle, 
            shoulder_angle, left_leg_angle, right_leg_angle, nose_to_ankle_angle]

def process_video_to_features_dynamic_label(video_path, is_fall_video):
    """Trích xuất 1 FPS và tự động gán nhãn Đứng/Nằm linh hoạt dựa trên góc nghiêng cơ thể"""
    records = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return records
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % int(fps) == 0:
            features = extract_biomechanic_features(frame)
            if features is not None:
                if is_fall_video:
                    # Lấy đặc trưng torso_angle (chỉ mục số 3 trong danh sách trả về)
                    torso_angle = features[3]
                    # Nếu người nghiêng > 45 độ so với phương thẳng đứng thì gán nhãn Ngã (0), ngược lại vẫn là Đứng (1)
                    actual_label = 0 if torso_angle >= 45.0 else 1
                else:
                    # Video ADL thuần túy thì mặc định 100% các frame là Đứng (1)
                    actual_label = 1
                
                records.append(features + [actual_label])
        count += 1
    cap.release()
    return records

# =========================================================================
# PHẦN 2: TỰ ĐỘNG LỌC VIDEO MỚI THEO FILE (CHỐNG LỆCH NHÃN)
# =========================================================================
def create_test_dataset(root_dir, output_test_csv, sample_ratio=0.3):
    print("\n[STEP 1] Đang quét danh sách file video trên máy...")
    mcfd_dataset_path = os.path.join(root_dir, "MCFD", "dataset")
    urfd_cam_path = os.path.join(root_dir, "URFD", "cam")
    
    all_videos = glob.glob(os.path.join(mcfd_dataset_path, "chute*", "*.mp4")) + \
                 glob.glob(os.path.join(mcfd_dataset_path, "chute*", "*.avi")) + \
                 glob.glob(os.path.join(urfd_cam_path, "FALL", "*.mp4")) + \
                 glob.glob(os.path.join(urfd_cam_path, "FALL", "*.avi")) + \
                 glob.glob(os.path.join(urfd_cam_path, "ADL", "*.mp4")) + \
                 glob.glob(os.path.join(urfd_cam_path, "ADL", "*.avi"))

    lying_pool = [v for v in all_videos if "fall" in v.lower() or "chute" in v.lower()]
    standing_pool = [v for v in all_videos if "adl" in v.lower()]

    # Đảo trộn danh sách video có seed cố định
    random.seed(42)
    random.shuffle(lying_pool)
    random.shuffle(standing_pool)
    
    # Bốc ngẫu nhiên 30% lượng video từ phân đoạn cuối (tránh trùng lặp tối đa với tập Train cũ của bạn)
    num_lying_test = max(1, int(len(lying_pool) * sample_ratio))
    num_standing_test = max(1, int(len(standing_pool) * sample_ratio))
    
    sampled_test_videos = lying_pool[-num_lying_test:] + standing_pool[-num_standing_test:]
    print(f"🎬 Đã lọc theo file! Chọn {num_lying_test} video chuỗi ngã và {num_standing_test} video ADL làm tập TEST...")
    
    all_test_records = []
    for v_path in sampled_test_videos:
        is_fall = True if ("fall" in v_path.lower() or "chute" in v_path.lower()) else False
        features = process_video_to_features_dynamic_label(v_path, is_fall_video=is_fall)
        all_test_records.extend(features)

    if all_test_records:
        columns = ["com_x", "com_y", "shoulder_nose_angle", "torso_angle", "hip_angle",
                   "shoulder_angle", "left_leg_angle", "right_leg_angle", "nose_to_ankle_angle", "pose_class"]
        pd.DataFrame(all_test_records, columns=columns).to_csv(output_test_csv, index=False)
        print(f"✅ Đã tạo xong tập kiểm thử động chuẩn xác tại: {output_test_csv}")
        return True
    else:
        print("❌ Lỗi trích xuất dữ liệu kiểm thử.")
        return False

# =========================================================================
# PHẦN 3: LUỒNG HUẤN LUYỆN SVM VÀ ĐO LƯỜNG ĐỘ CHÍNH XÁC ĐỘC LẬP
# =========================================================================
def train_and_evaluate_svm(train_csv_path, test_csv_path):
    print("\n[STEP 2] Đang khởi chạy quy trình Huấn luyện SVM...")
    train_data = pd.read_csv(train_csv_path)
    X_train = train_data.drop("pose_class", axis=1)
    y_train = train_data["pose_class"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, "scaler.pkl")

    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    clf.fit(X_train_scaled, y_train)
    joblib.dump(clf, "fall_detection_svm.pkl")
    print("💾 Đã huấn luyện xong và lưu mô hình tại 'fall_detection_svm.pkl'")

    print("\n[STEP 3] Đang tiến hành chấm điểm mô hình trên tập dữ liệu TEST mới lạ...")
    test_data = pd.read_csv(test_csv_path)
    X_test = test_data.drop("pose_class", axis=1)
    y_test = test_data["pose_class"]

    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)

    print("\n" + "=" * 60)
    print(f"🎯 ĐỘ CHÍNH XÁC THỰC TẾ (ACCURACY) TRÊN VIDEO MỚI: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("=" * 60)
    print("\n📊 MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print(confusion_matrix(y_test, y_pred))
    print("\n📋 BÁO CÁO CHI TIẾT HIỆU NĂNG PHÂN LOẠI:")
    print(classification_report(y_test, y_pred, target_names=["0_Lying", "1_Standing"]))


# =========================================================================
# THỰC THI HỆ THỐNG
# =========================================================================
root_data_directory = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2"
train_csv_file = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Coding\training\training2\pifr_custom_dataset.csv"
test_csv_file = "./pifr_test_dataset.csv" 

if __name__ == '__main__':
    # Chạy Tác vụ 1: Tạo file test phân tách động
    success = create_test_dataset(root_data_directory, test_csv_file, sample_ratio=0.3)
    
    # Chạy Tác vụ 2: Train SVM và chấm điểm thực tế
    if success:
        train_and_evaluate_svm(train_csv_file, test_csv_file)