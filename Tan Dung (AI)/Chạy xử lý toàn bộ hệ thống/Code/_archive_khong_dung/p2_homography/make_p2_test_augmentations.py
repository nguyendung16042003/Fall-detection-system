"""
Augment TEST data cho P2 Data 3 -- MO RONG ca 4 canh (truoc chi co Canh 4).
CHI dung augment AN TOAN voi homography (giong P3, KHONG lat/xoay/crop --
P2 cung phu thuoc toa do pixel goc da calib)."""
from pathlib import Path

import cv2
import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
P2_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 2" / "P2 data test"
SRC_DIR = P2_DIR / "P2 Data 3"
DST_DIR = P2_DIR / "P2 Data 3 (augmented)"
SCENE_IDS = [1, 2, 3, 4]


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
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for scene_id in SCENE_IDS:
        for cam in ["CAM 1", "CAM 2"]:
            src = SRC_DIR / f"Scene {scene_id}-{cam}.mp4"
            if not src.exists():
                print(f"CANH BAO: khong thay {src}")
                continue
            for aug_name, aug_fn in AUGMENTATIONS.items():
                dst_path, n = process_video(src, DST_DIR, aug_name, aug_fn)
                print(f"  {src.name} -> {dst_path.name} ({n} frame)")
    print(f"\nDa tao augment cho P2 Data 3, canh {SCENE_IDS}.")


if __name__ == "__main__":
    main()
