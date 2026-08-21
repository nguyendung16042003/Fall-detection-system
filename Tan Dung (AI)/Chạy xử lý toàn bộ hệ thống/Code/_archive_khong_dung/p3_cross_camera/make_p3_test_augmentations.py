"""
Augment TEST data cho P3 (KHONG PHAI train) -- tang so "lan chay" danh gia
tu 6 canh that (4 nga + 2 ADL, P3 Data 1/2) len 36 (moi canh x6 bien the:
1 goc + 4 augment anh sang/mau/nen + 1 dao camera tinh o buoc danh gia,
khong tao file rieng).

QUAN TRONG -- CHI dung augment KHONG DOI vi tri pixel (an toan voi homography
da calib theo toa do pixel goc): sang/toi, nhieu Gaussian, nen JPEG chat
luong thap. KHONG xoay/crop/scale/lat/warp -- se lam sai homography ma
khong bao loi.

CAC BAN AUGMENT LA BIEN THE CUA CUNG 1 SU KIEN THAT, KHONG PHAI CASE DOC
LAP MOI -- ghi ro trong ten file + khi bao cao so lieu (khong duoc dem la
"N su kien nga rieng biet" khi N thuc chat la it su kien x nhieu bien the).
"""
from pathlib import Path

import cv2
import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
P3_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 3" / "P3"

SOURCES = [
    (P3_DIR / "P3 Data 1", P3_DIR / "P3 Data 1 (augmented)", [1, 2, 3, 4], "Cam"),
    (P3_DIR / "P3 Data 2", P3_DIR / "P3 Data 2 (augmented)", [1, 2], "Cam"),
]


def aug_bright(frame, delta):
    return np.clip(frame.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def aug_noise(frame, sigma=12.0):
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_jpeg_compress(frame, quality=15):
    ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


AUGMENTATIONS = {
    "aug-bright_up": lambda f: aug_bright(f, 45),
    "aug-bright_down": lambda f: aug_bright(f, -45),
    "aug-noise": lambda f: aug_noise(f, 12.0),
    "aug-compress": lambda f: aug_jpeg_compress(f, 15),
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
    total_files = 0
    for src_folder, dst_folder, scene_ids, cam_prefix in SOURCES:
        dst_folder.mkdir(parents=True, exist_ok=True)
        for scene_id in scene_ids:
            for cam_id in [1, 2]:
                src = src_folder / f"Scene {scene_id}-{cam_prefix} {cam_id}.mp4"
                if not src.exists():
                    print(f"  CANH BAO: khong thay {src}")
                    continue
                for aug_name, aug_fn in AUGMENTATIONS.items():
                    dst_path, n = process_video(src, dst_folder, aug_name, aug_fn)
                    print(f"  {src.name} -> {dst_path.name} ({n} frame)")
                    total_files += 1

    print(f"\nTong: {total_files} file video augmented da tao.")
    print("Luu y: day la BIEN THE cua cac su kien that da co (4 nga + 2 ADL trong P3 Data 1/2),")
    print("KHONG phai su kien doc lap moi -- dung de kiem tra do on dinh (robustness),")
    print("khong dung de tang so 'case nga doc lap' khi tinh Sensitivity chinh thuc.")


if __name__ == "__main__":
    main()
