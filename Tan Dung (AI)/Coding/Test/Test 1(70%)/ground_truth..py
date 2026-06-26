from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================

MCFD_PATH = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2\MCFD"

URFD_FALL_PATH = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2\URFD\Cam\FALL"

URFD_ADL_PATH = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Test 2\URFD\Cam\ADL"

OUTPUT_CSV = "ground_truth.csv"

# =========================
# BUILD GROUND TRUTH
# =========================

rows = []

# -------------------------
# MCFD = FALL
# -------------------------

for video in Path(MCFD_PATH).rglob("*.avi"):

    rows.append({
        "video_name": str(video).replace("\\", "/"),
        "ground_truth": "Fall"
    })

# -------------------------
# URFD FALL
# -------------------------

for video in Path(URFD_FALL_PATH).rglob("*.mp4"):

    rows.append({
        "video_name": str(video).replace("\\", "/"),
        "ground_truth": "Fall"
    })

# -------------------------
# URFD ADL
# -------------------------

for video in Path(URFD_ADL_PATH).rglob("*.mp4"):

    rows.append({
        "video_name": str(video).replace("\\", "/"),
        "ground_truth": "NoFall"
    })

# =========================
# SAVE
# =========================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(df.head())

print("\n======================")
print("GROUND TRUTH CREATED")
print("======================")

print(f"Videos: {len(df)}")

print(f"Saved: {OUTPUT_CSV}")