import cv2
import os
import random
import glob
import math
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO

# 1. Nạp các mô hình nền tảng
try:
    clf = joblib.load("fall_detection_svm.pkl")
    scaler = joblib.load("scaler.pkl")
    model = YOLO("yolo11n-pose.pt")
    print("✅ Đã nạp thành công các mô hình phục vụ kiểm thử nâng cao!")
except Exception as e:
    print(f"❌ Lỗi nạp mô hình: {e}")
    exit()

FEATURE_NAMES = ["com_x", "com_y", "shoulder_nose_angle", "torso_angle", "hip_angle", 
                 "shoulder_angle", "left_leg_angle", "right_leg_angle", "nose_to_ankle_angle"]

# =========================================================================
# CÁC HÀM TÍNH TOÁN BIOMECHANICS & HÌNH HỌC
# =========================================================================
def calculate_angle_3points(A, B, C):
    BA = (A[0] - B[0], A[1] - B[1]); BC = (C[0] - B[0], C[1] - B[1])
    dot = BA[0]*BC[0] + BA[1]*BC[1]
    norm = math.sqrt(BA[0]**2 + BA[1]**2) * math.sqrt(BC[0]**2 + BC[1]**2)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / norm)))) if norm != 0 else 0

def calculate_angle_with_vertical(A, B):
    dx, dy = B[0] - A[0], B[1] - A[1]
    norm = math.sqrt(dx**2 + dy**2)
    return math.degrees(math.acos(dy / norm)) if norm != 0 else 0

def calculate_angle_with_horizontal(A, B):
    dx, dy = B[0] - A[0], B[1] - A[1]
    norm = math.sqrt(dx**2 + dy**2)
    return math.degrees(math.acos(dx / norm)) if norm != 0 else 0

def extract_features_and_metrics(image):
    results = model(image, verbose=False)
    if not results or results[0].keypoints is None or results[0].keypoints.xy.shape[0] == 0:
        return None
        
    kp = results[0].keypoints.xy[0].cpu().numpy()
    if len(kp) < 17 or np.all(kp == 0): return None
    
    # Tính bounding box chuẩn từ các điểm chốt hợp lệ
    x_coords = kp[:, 0][kp[:, 0] > 0]
    y_coords = kp[:, 1][kp[:, 1] > 0]
    if len(x_coords) == 0 or len(y_coords) == 0: return None
    
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    aspect_ratio = width / height if height > 0 else 0
    
    nose, l_shoulder, r_shoulder = kp[0], kp[5], kp[6]
    l_hip, r_hip = kp[11], kp[12]; l_knee, r_knee = kp[13], kp[14]; l_ankle, r_ankle = kp[15], kp[16]
    mid_hip = ((l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2)
    mid_ankle = ((l_ankle[0] + r_ankle[0])/2, (l_ankle[1] + r_ankle[1])/2)
    
    features = [np.mean(kp[:,0]), np.mean(kp[:,1]), calculate_angle_3points(l_shoulder, nose, r_shoulder),
                calculate_angle_with_vertical(nose, mid_hip), calculate_angle_with_horizontal(l_hip, r_hip),
                calculate_angle_with_horizontal(l_shoulder, r_shoulder), calculate_angle_3points(l_hip, l_knee, l_ankle),
                calculate_angle_3points(r_hip, r_knee, r_ankle), calculate_angle_with_vertical(nose, mid_ankle)]
                
    return {"aspect_ratio": aspect_ratio, "features": features}

# =========================================================================
# HÀM KIỂM TRA CHUỖI VIDEO (CƠ CHẾ SLIDING WINDOW THÍCH ỨNG)
# =========================================================================
def evaluate_single_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Sử dụng cửa sổ trượt dài hơn (1.5 giây) để tăng độ chín chắn khi biểu quyết
    window_size = int(fps * 1.5) 
    buffer = []
    triggered = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        data = extract_features_and_metrics(frame)
        if data is not None:
            aspect_ratio = data["aspect_ratio"]
            features = data["features"]
            
            # Khởi tạo ma trận DataFrame đưa vào SVM
            features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
            features_scaled = scaler.transform(features_df)
            pred = clf.predict(features_scaled)[0]
            
            # 🎯 ĐỘT PHÁ LOGIC: Đánh giá dựa trên phân phối hình học động
            # Một cú ngã thực sự sẽ kéo theo góc nghiêng người (features[3]) lớn hơn 35 độ
            # HOẶC khi ngã hẳn xuống sàn, tỷ lệ hộp bao ngang/dọc phải bẹt ra (aspect_ratio > 0.8)
            torso_angle = features[3]
            
            if pred == 0:
                if aspect_ratio > 0.80 or torso_angle > 35.0:
                    buffer.append(0)  # Ghi nhận frame Nghi vấn Ngã
                else:
                    buffer.append(1)  # Lọc nhiễu: Dù SVM báo 0 nhưng dáng đứng/ngồi thẳng lưng vẫn tính là An toàn
            else:
                buffer.append(1)
        else:
            # Nếu mất dấu keypoint (do khuất hình), giữ nguyên giá trị frame trước để chuỗi không bị đứt gãy
            if len(buffer) > 0:
                buffer.append(buffer[-1])
            else:
                buffer.append(1)
                
        if len(buffer) > window_size: 
            buffer.pop(0)
            
        # Nâng ngưỡng biểu quyết: Chỉ chốt NGÃ khi có trên 50% số frame trong 1.5 giây đồng thuận
        if len(buffer) == window_size and buffer.count(0) / window_size > 0.50:
            triggered = True
            break
            
    cap.release()
    return triggered

# =========================================================================
# LUỒNG THỰC NGHIỆM TỰ ĐỘNG
# =========================================================================
def run_experiment_10_videos(root_dir):
    all_v = glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True) + \
            glob.glob(os.path.join(root_dir, "**", "*.avi"), recursive=True)
            
    fall_v = [v for v in all_v if "fall" in v.lower() or "chute" in v.lower()]
    adl_v = [v for v in all_v if "adl" in v.lower()]
    
    if len(fall_v) < 5 or len(adl_v) < 5:
        print(f"❌ Thư mục không đủ video (Hiện có: {len(fall_v)} Fall, {len(adl_v)} ADL)")
        return

    # Khóa seed cố định để đảm bảo tính sòng phẳng khi test lại nhóm file cũ
    random.seed(101)
    random.shuffle(fall_v)
    random.shuffle(adl_v)
    
    test_fall = fall_v[:5]
    test_adl = adl_v[:5]
    
    TP, FP, TN, FN = 0, 0, 0, 0
    print("\n⏳ Hệ thống tiến hành quét thực nghiệm chuỗi thời gian...")
    
    print("\n➔ [TESTING] Nhóm video NGÃ (FALL):")
    for idx, v in enumerate(test_fall):
        name = os.path.basename(v)
        if evaluate_single_video(v):
            print(f"   [{idx+1}] Video: {name} ➔ Cảnh báo: Đã kích hoạt chuông đỏ 🚨")
            TP += 1
        else:
            print(f"   [{idx+1}] Video: {name} ➔ Thất bại: Bị lọt lưới (FN) ❌")
            FN += 1
            
    print("\n➔ [TESTING] Nhóm video BÌNH THƯỜNG (ADL):")
    for idx, v in enumerate(test_adl):
        name = os.path.basename(v)
        if evaluate_single_video(v):
            print(f"   [{idx+1}] Video: {name} ➔ Thất bại: Báo động giả (FP) ❌")
            FP += 1
        else:
            print(f"   [{idx+1}] Video: {name} ➔ Cảnh báo: Giữ trạng thái an toàn (Xanh)  ")
            TN += 1

    # Tính toán kết quả Metrics cuối cùng
    acc = (TP + TN) / 10
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    print("\n" + "="*60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ CHUỖI VIDEO THỰC TẾ CUỐI CÙNG")
    print("="*60)
    print(f"● True Positive (Bắt đúng ca ngã):   {TP}/5")
    print(f"● True Negative (Lọc đúng ca ADL):   {TN}/5")
    print(f"● False Positive (Báo động giả):    {FP}/5")
    print(f"● False Negative (Bỏ sót ca ngã):   {FN}/5")
    print("-" * 60)
    print(f"🎯 ACCURACY (Độ chính xác):        {acc*100:.2f}%")
    print(f"🔥 PRECISION (Chống báo nhầm):     {prec*100:.2f}%")
    print(f"⚡ RECALL (Tỷ lệ bắt trúng ngã):    {rec*100:.2f}%")
    print(f"🏆 F1-SCORE (Chỉ số cân bằng):      {f1*100:.2f}%")
    print("="*60)

if __name__ == '__main__':
    path = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2"
    run_experiment_10_videos(path)