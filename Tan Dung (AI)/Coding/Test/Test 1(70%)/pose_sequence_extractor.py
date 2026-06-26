from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================

MODEL_PATH = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\runs\classify\runs\classify\clsv2_1-14\weights\best.pt"

MCFD_PATH = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2\MCFD"
URFD_PATH = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2\URFD"

OUTPUT_CSV = "pose_sequences.csv"

FRAME_SKIP = 5

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = YOLO(MODEL_PATH)

CLASS_NAMES = [
    "bend",
    "exercise",
    "lie",
    "sit",
    "stand",
    "unknown"
]

# =========================
# FIND VIDEOS
# =========================

video_files = []

for ext in ["*.avi", "*.mp4"]:
    video_files.extend(Path(MCFD_PATH).rglob(ext))
    video_files.extend(Path(URFD_PATH).rglob(ext))

print(f"Found {len(video_files)} videos")

# =========================
# PROCESS
# =========================

rows = []

for idx, video_path in enumerate(video_files, start=1):

    print(f"\n[{idx}/{len(video_files)}] Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    frame_idx = 0
    processed = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        if frame_idx % FRAME_SKIP != 0:
            continue

        processed += 1

        result = model.predict(
            source=frame,
            verbose=False
        )[0]

        cls_id = int(result.probs.top1)

        pose_name = CLASS_NAMES[cls_id]

        rows.append({
            "video_name": str(video_path),
            "frame_id": frame_idx,
            "pose": pose_name
        })

    cap.release()

    print(
        f"Done | Frames={frame_idx} | Processed={processed}"
    )

# =========================
# SAVE
# =========================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", OUTPUT_CSV)
print(df.head())