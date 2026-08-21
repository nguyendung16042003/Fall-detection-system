import cv2
import time
import collections
import numpy as np
from ultralytics import YOLO

# =========================================================================
# ⚙️ CONFIGURATION PIPELINE: CƠ CHẾ PHÂN TÍCH HÌNH HỌC HỘP CÁT BIẾN ĐỘNG
# =========================================================================
CONFIG = {
    "detector_path": "yolov8n.pt", # Chỉ cần dùng duy nhất 1 mô hình Object Detection gốc
    "video_source": r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2\MCFD\dataset\chute01\cam1.avi",
    
    # Ngưỡng toán học quyết định hình thái hộp cát
    "aspect_ratio_fall_threshold": 1.3, # Nếu Chiều rộng / Chiều cao > 1.3 -> Dáng người đang nằm ngang
    "velocity_fall_threshold": 45,      # Tốc độ dịch chuyển đáy hộp cát (pixels/frame) khi ngã sụp
    "window_size": 15
}

def main():
    print("🎯 [SYSTEM] Đang nạp mô hình YOLOv8-det...")
    model = YOLO(CONFIG["detector_path"])
    
    # Lưu trữ lịch sử tọa độ không gian để tính toán vận tốc rơi tự do
    # Cấu trúc: { person_idx: { "last_y2": val, "aspect_queue": deque, "state": "SAFE" } }
    geometry_trackers = {}

    cap = cv2.VideoCapture(CONFIG["video_source"])
    if not cap.isOpened():
        print(f"❌ Lỗi: Không thể mở video: {CONFIG['video_source']}")
        return

    print("▶️ BẮT ĐẦU LUỒNG KIỂM THỬ CƠ CHẾ PHÂN TÍCH BIẾN ĐỘNG HÌNH HỌC HỘP CÁT.")
    print("-" * 95)

    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.45, verbose=False)
        boxes = results[0].boxes
        
        current_frame_targets = set()

        for idx, box in enumerate(boxes):
            class_id = int(box.cls[0])
            if class_id == 0:  # Thực thể người
                current_frame_targets.add(idx)
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Tính toán kích thước hộp cát thực tế
                box_width = x2 - x1
                box_height = y2 - y1
                
                # 💥 Chỉ số quan trọng 1: Tỷ lệ hình học (Aspect Ratio = W / H)
                aspect_ratio = box_width / float(box_height) if box_height > 0 else 0
                
                # Khởi tạo bộ theo dõi không gian nếu là người mới
                if idx not in geometry_trackers:
                    geometry_trackers[idx] = {
                        "last_y2": y2,
                        "aspect_queue": collections.deque(maxlen=CONFIG["window_size"]),
                        "state": "SAFE",
                        "alarm_lock": False
                    }
                    
                tracker = geometry_trackers[idx]
                tracker["aspect_queue"].append(aspect_ratio)
                
                # 💥 Chỉ số quan trọng 2: Vận tốc rơi tự do (Tốc độ biến động cạnh đáy Y2)
                velocity = y2 - tracker["last_y2"]
                tracker["last_y2"] = y2 # Cập nhật tọa độ nền cho frame sau
                
                # Tính toán tỷ lệ nằm ngang trung bình trong cửa sổ thời gian
                avg_aspect = np.mean(tracker["aspect_queue"])
                
                # --- TOÁN HỌC QUYẾT ĐỊNH TẦNG RULE HÌNH HỌC XUYÊN SUỐT ---
                
                # Điều kiện 1: Phát hiện cú sụp trọng lực nhanh (Vận tốc vượt ngưỡng)
                if velocity > CONFIG["velocity_fall_threshold"]:
                    tracker["state"] = "FALLING"
                    
                # Điều kiện 2: Đã sụp xuống và cơ thể trải dài nằm ngang ổn định
                if avg_aspect >= CONFIG["aspect_ratio_fall_threshold"] or tracker["state"] == "FALLING":
                    # Nếu có vận tốc rơi đi trước hoặc dáng nằm ngang kéo dài -> Kích nổ ALERT
                    tracker["state"] = "ALERT"
                    color = (0, 0, 255) # Đỏ rực nguy hiểm
                    status_text = f"🚨 ALERT: FALL DETECTED (W/H: {avg_aspect:.2f})"
                elif avg_aspect > 0.8 and avg_aspect < CONFIG["aspect_ratio_fall_threshold"]:
                    tracker["state"] = "SITTING/BENDING"
                    color = (255, 0, 0) # Xanh dương theo dõi
                    status_text = f"STATE: SIT/BEND (W/H: {avg_aspect:.2f})"
                else:
                    tracker["state"] = "SAFE"
                    color = (0, 255, 0) # Xanh lá cây an toàn tuyệt đối
                    status_text = f"STATE: STANDING (W/H: {avg_aspect:.2f})"

                # Vẽ khung Bounding Box hình học
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{idx} {status_text}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                
                # Vẽ thêm đường nét vận tốc để dễ biện luận đồ án
                if velocity > 5:
                    cv2.putText(frame, f"V-Drop: {velocity} px/f", (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)

        # Giải phóng các ID đã đi ra khỏi khung hình để tránh rác bộ nhớ
        for idx in list(geometry_trackers.keys()):
            if idx not in current_frame_targets:
                del geometry_trackers[idx]

        # TẦNG HUD CHUNG
        current_time = time.time()
        fps = int(1 / (current_time - prev_time)) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Pure Bounding-Box Geometry Fall Pipeline (Anti-Classification Blur)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()