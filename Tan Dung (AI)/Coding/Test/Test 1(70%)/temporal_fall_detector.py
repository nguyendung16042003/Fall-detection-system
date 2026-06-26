import pandas as pd

INPUT_CSV = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\pose_sequences.csv"
OUTPUT_CSV = "fall_predictions.csv"

LIE_RATIO_THRESHOLD = 0.40

df = pd.read_csv(INPUT_CSV)

results = []

for video_name, group in df.groupby("video_name"):

    total_frames = len(group)

    lie_frames = (group["pose"] == "lie").sum()

    lie_ratio = lie_frames / total_frames

    if lie_ratio >= LIE_RATIO_THRESHOLD:
        prediction = "Fall"
    else:
        prediction = "NoFall"

    results.append({
        "video_name": video_name,
        "total_frames": total_frames,
        "lie_frames": lie_frames,
        "lie_ratio": round(lie_ratio, 3),
        "prediction": prediction
    })

out_df = pd.DataFrame(results)

out_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(out_df.head())

print(f"\nSaved: {OUTPUT_CSV}")