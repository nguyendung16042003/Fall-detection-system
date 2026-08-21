"""
Augment TEST data cho P1 (KHONG PHAI train) -- tang so lan chay danh gia
Test Tang 2 tu 1 video test (2 lan A xuat hien lai) len nhieu bien the.

Khac P3 (bi rang buoc boi homography calib theo pixel goc), Re-ID CHI can
crop nguoi qua detector roi dua vao model trich embedding -- KHONG phu
thuoc toa do pixel tuyet doi, nen co the dung THEM augment hinh hoc nhe
(lat ngang) ngoai cac augment an toan chung (sang/toi/nhieu/nen).

CAC BAN AUGMENT LA BIEN THE CUA CUNG 1 VIDEO TEST THAT (2 su kien A quay
lai), KHONG PHAI su kien doc lap moi -- ghi ro khi bao cao so lieu.
"""
from pathlib import Path

import cv2
import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
P1_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1" / "P1 test"
SRC_VIDEOS = [P1_DIR / "Data 2" / "Scene 5-CAM 1.mp4", P1_DIR / "Data 2" / "Scene 5-CAM 2.mp4"]
DST_DIR = P1_DIR / "Data 2 (augmented)"


def aug_bright(frame, delta):
    return np.clip(frame.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def aug_noise(frame, sigma=12.0):
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_jpeg_compress(frame, quality=15):
    ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def aug_flip(frame):
    return cv2.flip(frame, 1)  # lat ngang (trai-phai)


AUGMENTATIONS = {
    "aug-bright_up": lambda f: aug_bright(f, 45),
    "aug-bright_down": lambda f: aug_bright(f, -45),
    "aug-noise": lambda f: aug_noise(f, 12.0),
    "aug-compress": lambda f: aug_jpeg_compress(f, 15),
    "aug-flip": aug_flip,
}


def process_video(src_path, dst_dir, aug_name, aug_fn):
    cap = cv2.VideoCapture(str(src_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst_path = dst_dir / f"{src_path.stem}_{aug_name}.mp4"
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (w, h))
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(aug_fn(frame))
        n += 1
    cap.release()
    writer.release()
    return dst_path, n


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for src_video in SRC_VIDEOS:
        if not src_video.exists():
            print(f"CANH BAO: khong thay {src_video}")
            continue
        for aug_name, aug_fn in AUGMENTATIONS.items():
            dst_path, n = process_video(src_video, DST_DIR, aug_name, aug_fn)
            print(f"  {src_video.name} -> {dst_path.name} ({n} frame)")
    print(f"\nDa tao {len(AUGMENTATIONS)} bien the x {len(SRC_VIDEOS)} camera goc "
          f"({', '.join(v.name for v in SRC_VIDEOS)}).")
    print("Luu y: day la BIEN THE cua CUNG cac video test that, KHONG phai video doc lap moi.")


if __name__ == "__main__":
    main()
