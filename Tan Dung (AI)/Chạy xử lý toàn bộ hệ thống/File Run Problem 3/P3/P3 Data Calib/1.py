import os
import cv2


def trim_video(input_path, output_path, start_sec, end_sec):
    # 1. Kiểm tra file đầu vào
    if not os.path.exists(input_path):
        print(f"❌ Lỗi: Không tìm thấy file đầu vào tại:\n{input_path}")
        return

    # 2. Tự động tạo thư mục chứa file đầu ra nếu chưa có
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Lỗi: Không thể mở file video!")
        return

    # Lấy thông số kĩ thuật
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    if end_frame > total_frames:
        end_frame = total_frames

    print(f"🎬 Thông tin Video:")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Độ phân giải: {width}x{height}")
    print(f"   - Cắt từ frame {start_frame} đến frame {end_frame}")

    # 3. Sử dụng codec 'XVID' và xuất đuôi .avi để tương thích tuyệt đối trên Windows
    if output_path.endswith(".mp4"):
        output_path = output_path.replace(".mp4", ".avi")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        current_frame += 1

        if current_frame % int(fps) == 0:
            print(
                f"⏳ Đang xử lý... {current_frame / fps:.1f}s / {end_sec:.1f}s"
            )

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ ĐÃ CẮT VÀ LƯU VIDEO THÀNH CÔNG TẠI:\n{output_path}")


# ==========================================
# 🎯 CẤU HÌNH VÀ CHẠY
# ==========================================

input_video = r"D:\data\REID\Data đăng ký\18_8_dangki (1).mp4"

# ĐÃ SỬA: Đã thêm tên file "Data_Calib_Cut.avi" ở cuối đường dẫn
output_video = r"D:\data"

start_time = 7.0
end_time = 27.0

trim_video(input_video, output_video, start_time, end_time)