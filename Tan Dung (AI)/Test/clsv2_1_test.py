from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path
from collections import Counter

# =========================
# CONFIG
# =========================

MODEL_PATH = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\runs\classify\runs\classify\clsv2_1-14\weights\best.pt"

MCFD_PATH = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2\MCFD"
URFD_PATH = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2\URFD"

OUTPUT_CSV = "pose_classification_results.csv"

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
# PROCESS VIDEOS
# =========================

results_list = []

for idx, video_path in enumerate(video_files, start=1):

    print(f"\n[{idx}/{len(video_files)}] Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Cannot open video")
        continue

    pose_counter = Counter()

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame is None:
            continue

        frame_idx += 1

        # Process 1 frame every FRAME_SKIP frames
        if frame_idx % FRAME_SKIP != 0:
            continue

        try:

            result = model.predict(
                source=frame,
                verbose=False
            )[0]

            cls_id = int(result.probs.top1)

            pose_name = CLASS_NAMES[cls_id]

            pose_counter[pose_name] += 1

        except Exception as e:

            print(
                f"Frame error in {video_path.name} "
                f"(frame {frame_idx}): {e}"
            )

            continue

    cap.release()

    total_frames = frame_idx
    processed_frames = sum(pose_counter.values())

    if processed_frames > 0:
        dominant_pose = pose_counter.most_common(1)[0][0]
    else:
        dominant_pose = "unknown"

    row = {
        "video_name": video_path.name,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "bend": pose_counter["bend"],
        "exercise": pose_counter["exercise"],
        "lie": pose_counter["lie"],
        "sit": pose_counter["sit"],
        "stand": pose_counter["stand"],
        "unknown": pose_counter["unknown"],
        "dominant_pose": dominant_pose
    }

    results_list.append(row)

    print(
        f"Done | Frames={total_frames} | "
        f"Processed={processed_frames} | "
        f"Dominant={dominant_pose}"
    )

# =========================
# SAVE CSV
# =========================

df = pd.DataFrame(results_list)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\n====================")
print("FINISHED")
print("====================")

print(df.head())

print(f"\nSaved to: {OUTPUT_CSV}")