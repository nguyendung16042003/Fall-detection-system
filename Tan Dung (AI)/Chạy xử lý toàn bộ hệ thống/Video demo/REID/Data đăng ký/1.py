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

    if not fps or fps <= 0:
        print("❌ Lỗi: Không đọc được FPS của video (file có thể bị hỏng).")
        cap.release()
        return

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    if end_frame > total_frames:
        end_frame = total_frames

    if start_frame >= end_frame:
        print("❌ Lỗi: start_sec phải nhỏ hơn end_sec (hoặc video quá ngắn).")
        cap.release()
        return

    print(f"🎬 Thông tin Video:")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Độ phân giải: {width}x{height}")
    print(f"   - Cắt từ frame {start_frame} đến frame {end_frame}")

    # 3. Đảm bảo output_path là FILE (có tên + đuôi mở rộng), không phải thư mục
    root, ext = os.path.splitext(output_path)
    if not ext:
        # Nếu người dùng chỉ truyền vào một thư mục -> tự đặt tên file mặc định
        output_path = os.path.join(output_path, "output_cut.avi")
        root, ext = os.path.splitext(output_path)
        # Tạo lại thư mục cha nếu cần (vì output_path vừa đổi)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Dùng codec 'XVID' + đuôi .avi để tương thích tuyệt đối trên Windows
    if ext.lower() != ".avi":
        output_path = root + ".avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Kiểm tra writer có mở được không (rất quan trọng, tránh "thành công giả")
    if not out.isOpened():
        print(f"❌ Lỗi: Không thể tạo file video tại:\n{output_path}")
        print("   (Kiểm tra lại đường dẫn có trùng tên thư mục, hoặc thiếu quyền ghi không)")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        current_frame += 1

        if int(fps) > 0 and current_frame % int(fps) == 0:
            print(
                f"⏳ Đang xử lý... {current_frame / fps:.1f}s / {end_sec:.1f}s"
            )

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Kiểm tra file thật sự đã được tạo ra và có dung lượng > 0
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"\n✅ ĐÃ CẮT VÀ LƯU VIDEO THÀNH CÔNG TẠI:\n{output_path}")
    else:
        print(f"\n❌ Lỗi: File output không được tạo ra hoặc rỗng:\n{output_path}")


# ==========================================
# 🎯 CẤU HÌNH VÀ CHẠY
# ==========================================

input_video = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Chạy xử lý toàn bộ hệ thống\Video demo\Boundary\Fall detect\Cam 2- Scene 3.mp4"

# ĐÃ SỬA: output_video là đường dẫn FILE đầy đủ (có tên file + đuôi .avi)
output_video = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Chạy xử lý toàn bộ hệ thống\Video demo\18_8_reid_3_cut.avi"

# ĐÃ SỬA: bỏ dấu "=" thừa gây lỗi cú pháp
start_time = 3.0
end_time = 14.0

trim_video(input_video, output_video, start_time, end_time)