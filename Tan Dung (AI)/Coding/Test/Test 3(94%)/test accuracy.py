import os
import cv2
import collections
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

def print_separator(character="-", length=85):
    print(character * length)

def main():
    # =========================================================================
    # 1. KHAI BÁO ĐƯỜNG DẪN TRỌNG SỐ VÀ THƯ MỤC GỐC DATASET VIDEO
    # =========================================================================
    MODEL_PATH = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\runs\classify\Fall_Detection_Advanced_Loss\YOLOv8n_AFCL_Balanced-5\weights\best.pt"
    
    # Thư mục cha chứa các thư mục con rẽ nhánh video của bạn (Ví dụ: các folder chứa 8 video kịch bản)
    DATASET_ROOT_DIR = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2"

    # =========================================================================
    # 2. CẤU HÌNH TẬP RULE THỜI GIAN THEO CHUẨN PIPELINE CỦA NHÓM
    # =========================================================================
    CONFIG_RULES = {
        "window_size": 15,              # Kích thước hàng đợi cửa sổ trượt (Frames)
        "fall_ratio_threshold": 0.70,   # Ngưỡng tỉ lệ 'lie' để kích hoạt báo động đỏ
        "half_person_frames": 10        # Số frame liên tiếp bị che khuất để báo động cam
    }

    model_classes = ['bend', 'exercise', 'half_person', 'lie', 'sit', 'stand']
    
    # Định dạng video được chấp nhận (Hệ thống sẽ bốc các file này để phân tích)
    valid_video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')

    # Kiểm tra điều kiện tồn tại vật lý của file trọng số và thư mục
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Lỗi cấu hình Model: Không tìm thấy file tại {MODEL_PATH}")
        return
    if not os.path.exists(DATASET_ROOT_DIR):
        print(f"❌ Lỗi cấu hình Dataset: Thư mục gốc không tồn tại tại {DATASET_ROOT_DIR}")
        return

    print("🤖 [SYSTEM] Đang nạp mô hình YOLOv8n-cls (AFCL Loss) lên RAM...")
    model = YOLO(MODEL_PATH)
    print("✅ [SYSTEM] Nạp mô hình thành công. Chuẩn bị càn quét luồng Video.")

    # Khởi tạo các mảng phẳng để thu thập nhãn nhị phân phục vụ ma trận đối chứng
    y_true_binary = []
    y_pred_binary = []

    # Các biến đếm thống kê hiệu năng chi tiết phục vụ viết báo cáo đồ án
    stats_counter = {
        "total_videos_processed": 0,
        "total_frames_evaluated": 0,
        "urfd_frames": 0,
        "mcfd_frames": 0,
        "skipped_files": 0
    }

    print_separator("=", 85)
    print("📸 BẮT ĐẦU QUÉT TOÀN DIỆN CHUỖI VIDEO THỜI GIAN TRÊN TẬP DATASET BÀI BÁO")
    print_separator("=", 85)

    # =========================================================================
    # 3. BỘ QUÉT ĐỆ QUY LÙNG SỤC FILE VIDEO TRÊN CÂY THƯ MỤC ĐA TẦNG
    # =========================================================================
    for root, dirs, files in os.walk(DATASET_ROOT_DIR):
        
        # Lọc ra tất cả các file video có trong thư mục hiện tại
        video_files = [f for f in files if f.lower().endswith(valid_video_extensions)]
        
        if not video_files:
            continue

        # Xác định nhãn thực tế gốc (Ground Truth) dựa trên chuỗi đường dẫn tuyệt đối
        # Đồng bộ logic bài báo: FALL = 1 (Video có ngã), NO-FALL/ADL = 0 (Video sinh hoạt/không ngã)
        path_lower = root.lower()
        
        if 'fall' in path_lower:
            true_video_label = 1
        elif 'adl' in path_lower or 'confusion' in path_lower or 'normal' in path_lower:
            true_video_label = 0
        else:
            # Nếu đường dẫn không chứa từ khóa nhận dạng nhãn cụ thể, tạm thời bỏ qua file này để tránh sai số
            stats_counter["skipped_files"] += len(video_files)
            continue

        is_urfd = "urfd" in path_lower

        # Duyệt qua từng file Video cụ thể được tìm thấy trong nhánh thư mục này
        for video_name in video_files:
            video_path = os.path.join(root, video_name)
            
            # Sử dụng OpenCV để mở luồng giải nén video trực tiếp vào RAM
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            stats_counter["total_videos_processed"] += 1
            
            # ---------------------------------------------------------------------
            # KHỞI TẠO TẦNG CỬA SỔ TRƯỢT THỜI GIAN & MÁY TRẠNG THÁI CHO FILE VIDEO NÀY
            # ---------------------------------------------------------------------
            # Reset hoàn toàn bộ nhớ đệm khi mở video mới để tránh dữ liệu video trước tràn sang
            window_queue = collections.deque(maxlen=CONFIG_RULES["window_size"])
            current_state = "SAFE"
            last_stable_state = "UNKNOWN"
            half_person_counter = 0

            # Vòng lặp đọc từng Khung hình (Frame) của video cho đến khi hết
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # Hết video, chuyển sang file tiếp theo

                try:
                    # Tiền xử lý: Chuyển đổi hệ màu từ BGR (OpenCV) sang RGB (PIL) để đồng bộ đặc trưng
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    # Chạy Inference qua mô hình YOLOv8n-cls của bạn
                    results = model(pil_img, verbose=False)
                    
                    # Trích xuất nhãn dự đoán đa lớp
                    pred_label_idx = results[0].probs.top1
                    pred_pose = model_classes[pred_label_idx]

                    # Cập nhật trạng thái tư thế ổn định gần nhất
                    if pred_pose in ['stand', 'sit', 'exercise', 'bend']:
                        last_stable_state = pred_pose

                    # Đẩy nhãn của frame hiện tại vào Cửa sổ trượt của Tầng Rule
                    window_queue.append(pred_pose)
                    
                    # Tính toán tần suất xuất hiện nhãn trong bộ đệm
                    counts = collections.Counter(window_queue)
                    lie_ratio = counts['lie'] / len(window_queue)

                    # --- THỰC THI THUẬT TOÁN QUYẾT ĐỊNH CỦA TẦNG RULE CẤU HÌNH ---
                    # [Quy tắc 1]: Kích hoạt lệnh báo động đỏ ngã rõ ràng
                    if lie_ratio >= CONFIG_RULES["fall_ratio_threshold"]:
                        current_state = "ALERT"
                        
                    # [Quy tắc 2]: Xử lý bộ lọc phòng vệ Half-Person khi bị che khuất
                    elif pred_pose == "half_person":
                        half_person_counter += 1
                        if last_stable_state in ['stand', 'bend'] and half_person_counter >= CONFIG_RULES["half_person_frames"]:
                            current_state = "ALERT"
                            
                    # [Quy tắc 3]: Cơ chế Reset khôi phục trạng thái an toàn
                    else:
                        if counts['stand'] > len(window_queue) // 2 or counts['sit'] > len(window_queue) // 2:
                            current_state = "SAFE"
                            half_person_counter = 0

                    # Chuyển đổi quyết định cuối cùng của Tầng Rule về mã nhị phân gửi đến ma trận đối chứng
                    final_system_decision = 1 if current_state == "ALERT" else 0
                    
                    # Đẩy cặp dữ liệu (Thực tế vs Dự đoán sau lọc Rule) vào mảng tổng
                    y_true_binary.append(true_video_label)
                    y_pred_binary.append(final_system_decision)

                    # Cập nhật bộ đếm khung hình
                    stats_counter["total_frames_evaluated"] += 1
                    if is_urfd:
                        stats_counter["urfd_frames"] += 1
                    else:
                        stats_counter["mcfd_frames"] += 1

                except Exception:
                    continue
            
            # Giải phóng vùng nhớ của file video vừa đọc xong
            cap.release()

    # =========================================================================
    # 4. XUẤT BÁO CÁO HIỆU NĂNG VÀ MA TRẬN NHẦM LẪN CHUẨN CHỨNG MINH ĐỒ ÁN
    # =========================================================================
    print_separator("-", 85)
    print(f"● Tổng số tệp Video (.mp4/.avi) quét thành công: {stats_counter['total_videos_processed']}")
    print(f"● Tổng số khung hình trích xuất từ tập URFD: {stats_counter['urfd_frames']} frames.")
    print(f"● Tổng số khung hình trích xuất từ tập MCFD: {stats_counter['mcfd_frames']} frames.")
    print(f"● Tổng số mẫu dữ liệu đưa vào kiểm thử: {stats_counter['total_frames_evaluated']} mẫu.")
    if stats_counter["skipped_files"] > 0:
        print(f"⚠️  Đã bỏ qua {stats_counter['skipped_files']} file video do không xác định được nhãn Fall/ADL từ đường dẫn.")
    print_separator("-", 85)

    if stats_counter["total_frames_evaluated"] == 0:
        print("❌ Lỗi chí mạng: Hệ thống quét được video nhưng không trích xuất được khung hình nào!")
        print("💡 Gợi ý: Hãy kiểm tra lại xem các file video có bị lỗi codec hoặc đường dẫn gốc trống không.")
        return

    print("\n" + "="*75)
    print("📊 BẢNG HIỆU NĂNG ĐỐI CHỨNG (MODEL + RULE TÙY BIẾN) TRÊN CÁC FILE VIDEO BÀI BÁO")
    print("="*75)
    target_names = ['NO-FALL (ADL)', 'FALL (Té Ngã)']
    print(classification_report(y_true_binary, y_pred_binary, target_names=target_names, zero_division=0))
    
    print("="*75)
    print("🧱 MA TRẬN NHẦM LẪN NHỊ PHÂN SAU BỘ LỌC THỜI GIAN (CONFUSION MATRIX)")
    print("="*75)
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    print(f"                      [Dự Đoán NO-FALL]    [Dự Đoán FALL]")
    print(f"Thực tế NO-FALL (ADL):       {cm[0][0]:<20} {cm[0][1]}")
    print(f"Thực tế TÉ NGÃ (FALL):       {cm[1][0]:<20} {cm[1][1]}")
    print("="*75)
    print("🎯 Toàn bộ luồng đọc Video trực tiếp đã được xử lý mượt mà và đóng gói!")
    print("="*75)

if __name__ == "__main__":
    main()