import cv2
import os

def cut_video(input_path, start_sec, end_sec, output_path=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}, Kích thước: {width}x{height}, Tổng frame: {total_frames}")

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    if output_path is None:
        folder, filename = os.path.split(input_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(folder, f"{name}_cut{ext}")

    # Tự tạo folder chứa output nếu chưa có
    out_folder = os.path.dirname(output_path)
    if out_folder and not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
        print(f"Đã tạo folder: {out_folder}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError("VideoWriter không mở được — thử đổi codec hoặc đuôi file .avi")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    frame_written = 0
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Dừng sớm tại frame {current_frame} (đọc frame thất bại)")
            break
        out.write(frame)
        frame_written += 1
        current_frame += 1

    cap.release()
    out.release()
    print(f"Đã ghi {frame_written} frame")
    print(f"Đã lưu video cắt tại: {output_path}")
    print(f"File tồn tại sau khi ghi: {os.path.exists(output_path)}")


# ==== CHỖ BẠN SỬA ====
input_path = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test P1,2,3\P3\P3-3-CAM 2.mp4"

# Lưu ngay trong cùng folder với video gốc, không phải D:\videos
folder = os.path.dirname(input_path)
output_path = os.path.join(folder, "P1-1(Canh 3)-CAM 1_cut.mp4")

start_sec =22.0
end_sec = 34.0
# =======================

cut_video(input_path, start_sec, end_sec, output_path)